"""
Assigns optional source spans to parsed expressions.

When the parser runs with ``source_spans=True``, every expression of each parsed
statement is annotated with a ``SourceSpan`` (stored under the ``"span"`` meta key and
exposed via ``Expression.source_span``): a half-open character range together with the
corresponding UTF-8 byte range, so the original input can be sliced back directly.

Ownership rules
---------------
Spans are nested intervals: a parent's span always contains its children's spans.

- A statement's span covers all of its tokens; the semicolons separating statements
  belong to no expression.
- Tokens between siblings (commas, operators, keywords such as ``AS`` or ``OVER``)
  belong to the enclosing expression.
- Clause-leading keywords (``SELECT``, ``WITH``, ``FROM``, ``WHERE``, ...) belong to
  the clause expression they introduce (see ``_LEADING_KEYWORDS``).
- Parentheses belong to the innermost ``Paren`` / ``Subquery`` / ``Tuple`` expression
  they delimit, square brackets to the ``Array`` they delimit; if the parser did not
  materialize such a node, they belong to the enclosing expression.
- A function's name and argument parentheses belong to its ``Func`` expression.
- A comment belongs to the innermost expression whose span covers it. Comments before
  a statement's first token or after its last token belong to that statement.

Spans always refer to the original parsed text. ``copy()`` preserves them, while
``transform()``/``replace()`` never fabricate spans for newly created nodes.
"""

from __future__ import annotations

from sqlglot import exp
from sqlglot.expressions.core import SourceSpan
from sqlglot.tokens import Token, TokenType

# Clause-leading keyword tokens claimed by the expression types that introduce them.
# Anything not claimed here (or by the delimiter rules below) simply falls through to
# the enclosing expression's span.
_LEADING_KEYWORDS: dict[type, frozenset[TokenType]] = {
    exp.Select: frozenset({TokenType.SELECT}),
    exp.With: frozenset({TokenType.WITH}),
    exp.From: frozenset({TokenType.FROM}),
    exp.Where: frozenset({TokenType.WHERE}),
    exp.Group: frozenset({TokenType.GROUP_BY}),
    exp.Having: frozenset({TokenType.HAVING}),
    exp.Order: frozenset({TokenType.ORDER_BY}),
    exp.Limit: frozenset({TokenType.LIMIT}),
    exp.Offset: frozenset({TokenType.OFFSET}),
    exp.Qualify: frozenset({TokenType.QUALIFY}),
    exp.Fetch: frozenset({TokenType.FETCH}),
    exp.Values: frozenset({TokenType.VALUES}),
    exp.Lateral: frozenset({TokenType.LATERAL}),
    exp.Distinct: frozenset({TokenType.DISTINCT}),
    exp.Case: frozenset({TokenType.CASE}),
    exp.When: frozenset({TokenType.WHEN}),
    exp.If: frozenset({TokenType.WHEN}),
    exp.Array: frozenset({TokenType.ARRAY}),
    exp.WindowSpec: frozenset({TokenType.ROWS, TokenType.RANGE, TokenType.BETWEEN}),
    exp.Join: frozenset(
        {
            TokenType.JOIN,
            TokenType.LEFT,
            TokenType.RIGHT,
            TokenType.FULL,
            TokenType.INNER,
            TokenType.CROSS,
            TokenType.OUTER,
            TokenType.NATURAL,
            TokenType.SEMI,
            TokenType.ANTI,
            TokenType.ASOF,
        }
    ),
}

# Expression types that own the parentheses wrapping their content. exp.CTE is included
# because the parenthesized CTE body does not always have a dedicated Subquery node.
_PAREN_OWNERS = (exp.Paren, exp.Subquery, exp.Tuple, exp.CTE)

# Temporary meta key used by the parser to record the char offset (inclusive) of the
# last token consumed when an expression was created. Consumed by the span assignment.
SPAN_END_META_KEY = "_span_end"


def utf8_byte_offsets(sql: str) -> list[int]:
    """Returns a prefix table mapping char offsets in `sql` to UTF-8 byte offsets."""
    offsets = [0] * (len(sql) + 1)
    total = 0
    for i, char in enumerate(sql):
        codepoint = ord(char)
        if codepoint < 0x80:
            total += 1
        elif codepoint < 0x800:
            total += 2
        elif codepoint < 0x10000:
            total += 3
        else:
            total += 4
        offsets[i + 1] = total
    return offsets


def assign_statement_spans(
    root: exp.Expr, tokens: list[Token], byte_offsets: list[int]
) -> None:
    """
    Computes and assigns source spans for every expression of a single parsed statement.

    Args:
        root: the statement's root expression.
        tokens: the statement's tokens (statement separators excluded).
        byte_offsets: char offset -> UTF-8 byte offset prefix table for the source SQL.
    """
    if not tokens:
        return

    size = len(tokens)
    index_by_start = {}
    index_by_end = {}
    for index, token in enumerate(tokens):
        index_by_start[token.start] = index
        index_by_end[token.end] = index

    # Matching delimiter lookup, e.g. for `(` at index i, rmatch[i] is the index of `)`
    rmatch: dict[int, int] = {}
    lmatch: dict[int, int] = {}
    delimiter_stack: list[int] = []
    pairs = {
        TokenType.L_PAREN: TokenType.R_PAREN,
        TokenType.L_BRACKET: TokenType.R_BRACKET,
    }
    for index, token in enumerate(tokens):
        token_type = token.token_type
        if token_type in pairs:
            delimiter_stack.append(index)
        elif delimiter_stack and token_type is pairs[tokens[delimiter_stack[-1]].token_type]:
            opening = delimiter_stack.pop()
            rmatch[opening] = index
            lmatch[index] = opening

    claimed = [False] * size
    ranges: dict[int, tuple[int, int]] = {}

    # Post-order traversal: children are processed before their parents
    stack = [(root, False)]
    while stack:
        node, expanded = stack.pop()
        if not isinstance(node, exp.Expr):
            continue
        if expanded:
            _assign_node_span(
                node,
                tokens,
                index_by_start,
                index_by_end,
                rmatch,
                lmatch,
                claimed,
                ranges,
                byte_offsets,
            )
            continue
        stack.append((node, True))
        for value in node.args.values():
            if isinstance(value, exp.Expr):
                stack.append((value, False))
            elif type(value) is list:
                for item in value:
                    if isinstance(item, exp.Expr):
                        stack.append((item, False))

    # The statement itself owns all of its tokens, plus any comments attached to its
    # boundary tokens (leading comments precede the first token, trailing comments
    # follow the last one).
    first, last = tokens[0], tokens[-1]
    start = first.start
    end = last.end + 1
    for comment_start, comment_end in first.comment_ranges:
        start = min(start, comment_start)
    for comment_start, comment_end in last.comment_ranges:
        end = max(end, comment_end)

    root.meta["span"] = SourceSpan(start, end, byte_offsets[start], byte_offsets[end])


def _claim_wrapping_delimiters(
    tokens: list[Token],
    start: int,
    end: int,
    size: int,
    claimed: list[bool],
    rmatch: dict[int, int],
    lmatch: dict[int, int],
    opening_type: TokenType,
) -> tuple[int, int]:
    # Claim wrapping `(...)` / `[...]` pairs around the range. Either both delimiters are
    # boundary tokens, or one of them is already inside the range, e.g.
    # `(SELECT ...) alias` (closer inside, alias outside) or
    # `cte AS (SELECT ...)` (opener inside, trailing closer outside)
    closing_type = {
        TokenType.L_PAREN: TokenType.R_PAREN,
        TokenType.L_BRACKET: TokenType.R_BRACKET,
    }[opening_type]
    while True:
        if (
            start > 0
            and end + 1 < size
            and not claimed[start - 1]
            and not claimed[end + 1]
            and tokens[start - 1].token_type is opening_type
            and rmatch.get(start - 1, -1) == end + 1
        ):
            start -= 1
            end += 1
        elif (
            start > 0
            and not claimed[start - 1]
            and tokens[start - 1].token_type is opening_type
            and start <= rmatch.get(start - 1, -1) <= end
        ):
            start -= 1
        elif (
            end + 1 < size
            and not claimed[end + 1]
            and tokens[end + 1].token_type is closing_type
            and start <= lmatch.get(end + 1, -1) <= end
        ):
            end += 1
        else:
            break
    return start, end


def _assign_node_span(
    node: exp.Expr,
    tokens: list[Token],
    index_by_start: dict[int, int],
    index_by_end: dict[int, int],
    rmatch: dict[int, int],
    lmatch: dict[int, int],
    claimed: list[bool],
    ranges: dict[int, tuple[int, int]],
    byte_offsets: list[int],
) -> None:
    size = len(tokens)
    start: int | None = None
    end: int | None = None

    meta = node._meta
    if meta:
        token_start = meta.get("start")
        token_end = meta.get("end")
        if token_start is not None and token_end is not None:
            start = index_by_start.get(token_start)
            end = index_by_end.get(token_end)

    for value in node.args.values():
        values = value if type(value) is list else (value,)
        for item in values:
            child_range = ranges.get(id(item))
            if child_range is not None:
                child_start, child_end = child_range
                start = child_start if start is None else min(start, child_start)
                end = child_end if end is None else max(end, child_end)

    # The parser records the offset of the last token consumed when the expression was
    # created, so keyword-only tails (e.g. CURRENT ROW) are anchored as well
    span_end_index = None
    if meta is not None:
        span_end_index = index_by_end.get(meta.pop(SPAN_END_META_KEY, None))

    if start is None or end is None:
        if span_end_index is None:
            return
        start = end = span_end_index

    # Claim wrapping parentheses, e.g. for exp.Subquery
    if isinstance(node, _PAREN_OWNERS):
        start, end = _claim_wrapping_delimiters(
            tokens, start, end, size, claimed, rmatch, lmatch, TokenType.L_PAREN
        )

    # Claim wrapping square brackets, e.g. for exp.Array
    if isinstance(node, exp.Array):
        start, end = _claim_wrapping_delimiters(
            tokens, start, end, size, claimed, rmatch, lmatch, TokenType.L_BRACKET
        )

    # Claim a closing parenthesis when its matching opening parenthesis is already in
    # range. This covers function calls (Func) as well as constructs that consume
    # parenthesized content without a dedicated child node, e.g. `x IN (1, 2, 3)`
    while (
        end + 1 < size
        and not claimed[end + 1]
        and tokens[end + 1].token_type is TokenType.R_PAREN
        and lmatch.get(end + 1, -1) >= start
    ):
        end += 1

    node_type = type(node)
    keywords = None
    for klass in node_type.__mro__:
        keywords = _LEADING_KEYWORDS.get(klass)
        if keywords is not None:
            break

    # Claim adjacent clause-leading keywords, e.g. SELECT for exp.Select
    if keywords is not None:
        while start > 0 and not claimed[start - 1] and tokens[start - 1].token_type in keywords:
            start -= 1

    # Apply the recorded tail last, so it can't prevent the delimiter rules from
    # claiming wrapping parentheses (e.g. when the parser consumed the closing `)`)
    if span_end_index is not None and span_end_index > end:
        end = span_end_index

    for index in range(start, end + 1):
        claimed[index] = True
    ranges[id(node)] = (start, end)

    first, last = tokens[start], tokens[end]
    char_start = first.start
    char_end = last.end + 1
    node.meta["span"] = SourceSpan(
        char_start, char_end, byte_offsets[char_start], byte_offsets[char_end]
    )
