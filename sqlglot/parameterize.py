"""
Replaces literal values in SQL expressions with bind placeholders.

The `parameterize` function turns a concrete query into a parameterized one, returning a
new AST together with the extracted parameter values (in execution order) and their
inferred types. The input AST is never modified.

Only a well-defined set of literals is parameterized:

- String and numeric literals (`exp.Literal`), including negative numbers (`exp.Neg`
  wrapping a numeric literal), which are extracted as a single parameter.
- Booleans (`exp.Boolean`) and `NULL` (`exp.Null`).
- Hex / byte strings (`exp.HexString`, `exp.ByteString`), extracted as `int` or `bytes`.

Literals are intentionally left untouched in positions where placeholders are not
generally valid or change the meaning of the query:

- Type parameters, e.g. the `10` and `2` in `CAST(x AS DECIMAL(10, 2))`.
- Interval values and units, e.g. `INTERVAL 3 DAY`.
- Ordinal references, i.e. integer literals directly under `ORDER BY` / `GROUP BY`.
- `LIMIT` / `OFFSET` for dialects that render limits as `TOP` (e.g. T-SQL, Teradata),
  as well as `FETCH FIRST` counts.
- DDL properties, e.g. `WITH (fillfactor = 70)`.
- Bit strings (`exp.BitString`) and escaped unicode strings, whose textual form carries
  semantics that a bound value cannot preserve.

Values are extracted losslessly: integers stay `int`, exact decimals become
`decimal.Decimal`, binary values become `bytes` -- nothing is round-tripped through
floats. Existing placeholders / parameters are preserved as-is and new placeholder
names are chosen so they never collide with them.
"""

from __future__ import annotations

import typing as t

from dataclasses import dataclass
from decimal import Decimal

from sqlglot import expressions as exp
from sqlglot.dialects.dialect import Dialect, DialectType
from sqlglot.helper import find_new_name

PLACEHOLDER_STYLES = ("named", "numbered", "qmark")

_INT32_MIN, _INT32_MAX = -(2**31), 2**31 - 1
_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1

# Subtrees in which literals must never be replaced by placeholders.
_PRUNE_TYPES = (exp.DataType, exp.Interval, exp.Fetch, exp.Property)

# Placeholders are first created with unique sentinel names; their execution order is
# then derived from their position in the generated SQL and they are renamed to their
# final (style-dependent) names.
_SENTINEL_PREFIX = "sqlglot_param_"


@dataclass(frozen=True)
class Parameter:
    """A single extracted parameter.

    Args:
        name: the placeholder name used in the generated SQL, or `None` for anonymous
            `?` (qmark style) placeholders.
        value: the extracted Python value (`int`, `Decimal`, `str`, `bytes`, `bool` or
            `None`), extracted without lossy float conversion.
        type: the inferred SQL type of the value. When the literal was directly cast
            (e.g. `CAST('2024-01-01' AS DATE)`), the cast's target type is used.
        statement: index of the statement (within `Parameterized.expressions`) the
            parameter belongs to.
    """

    name: t.Optional[str]
    value: t.Any
    type: exp.DataType
    statement: int = 0


@dataclass
class Parameterized:
    """The result of parameterizing one or more SQL statements.

    Args:
        expressions: the new ASTs, with supported literals replaced by placeholders.
        parameters: the extracted parameters, in execution (textual) order across all
            statements.
        dialect: the target dialect used for SQL generation.
        style: the placeholder style that was used.
    """

    expressions: list[exp.Expression]
    parameters: list[Parameter]
    dialect: DialectType = None
    style: str = "named"

    @property
    def expression(self) -> exp.Expression:
        """The first (usually only) parameterized expression."""
        return self.expressions[0]

    @property
    def values(self) -> list[t.Any]:
        """The parameter values in execution order."""
        return [p.value for p in self.parameters]

    def sqls(self, **opts: t.Any) -> list[str]:
        """Generates SQL for every parameterized statement."""
        return [e.sql(dialect=self.dialect, **opts) for e in self.expressions]

    def sql(self, **opts: t.Any) -> str:
        """Generates SQL for the first parameterized statement."""
        return self.expression.sql(dialect=self.dialect, **opts)


def parameterize(
    expression: str | exp.Expression | t.Sequence[exp.Expression],
    dialect: DialectType = None,
    style: str = "named",
    dedupe: bool = True,
) -> Parameterized:
    """
    Replaces supported literals with placeholders and collects their values.

    Args:
        expression: a SQL string (possibly containing multiple statements), a single
            expression, or a sequence of expressions.
        dialect: the target dialect. It affects both parsing (for string input) and
            which positions are considered parameterizable (e.g. `LIMIT` for
            `TOP`-style dialects).
        style: the placeholder style:
            - "named": named placeholders such as `:p1` / `@p1` / `$p1`, depending on
              the dialect.
            - "numbered": numerically named placeholders (`1`, `2`, ...), rendered
              e.g. as `$1`, `$2` by dialects like DuckDB.
            - "qmark": anonymous `?` placeholders; parameters are purely positional.
        dedupe: if True, identical literals (same value and kind) share a single
            parameter. Literals with the same text but different kinds (e.g. the
            string `'1'` vs the number `1`) are never merged. Ignored for the "qmark"
            style, where every placeholder is a separate positional slot.

    Returns:
        A `Parameterized` result containing the new expression(s) and the parameters
        in execution order. The input AST is not modified.
    """
    if style not in PLACEHOLDER_STYLES:
        raise ValueError(
            f"Unknown placeholder style: {style!r}. Expected one of {PLACEHOLDER_STYLES}."
        )

    if isinstance(expression, str):
        expressions: list[exp.Expression] = [
            e for e in Dialect.get_or_raise(dialect).parse(expression) if e is not None
        ]
    elif isinstance(expression, exp.Expression):
        expressions = [expression]
    else:
        expressions = list(expression)

    parameterizer = _Parameterizer(dialect, style, dedupe)

    for e in expressions:
        parameterizer.collect_existing_names(e)

    expressions = [parameterizer.run(e, i) for i, e in enumerate(expressions)]

    return Parameterized(
        expressions=expressions,
        parameters=parameterizer.finalize(expressions),
        dialect=dialect,
        style=style,
    )


class _Parameterizer:
    def __init__(self, dialect: DialectType, style: str, dedupe: bool) -> None:
        self.dialect = dialect
        self.keep_limit_literals = Dialect.get_or_raise(dialect).generator_class.LIMIT_IS_TOP
        self.style = style
        self.dedupe = dedupe and style != "qmark"
        self.taken: set[str] = set()
        self.counter = 0
        self._statement = 0
        self._sentinel_counter = 0
        self._skip: set[int] = set()
        self._seen: dict[tuple[str, type, t.Any], str] = {}
        # sentinel name -> [value, inferred type, statement], in creation order
        self._extracted: dict[str, list[t.Any]] = {}
        self.parameters: list[Parameter] = []

    def collect_existing_names(self, expression: exp.Expression) -> None:
        for node in expression.walk():
            if isinstance(node, exp.Placeholder):
                if node.this:
                    self.taken.add(node.name)
            elif isinstance(node, exp.Parameter):
                self.taken.add(node.text("this"))

    def run(self, expression: exp.Expression, statement: int) -> exp.Expression:
        self._statement = statement
        root = expression.copy()

        for node in root.dfs(prune=self._prune):
            if id(node) in self._skip:
                continue

            placeholder = self._placeholder_for(node)
            if placeholder is not None:
                if node is root:
                    root = placeholder
                else:
                    node.replace(placeholder)

        return root

    def finalize(self, expressions: list[exp.Expression]) -> list[Parameter]:
        """Orders parameters by placeholder position in the generated SQL and renames
        the placeholders to their final, style-dependent names."""
        positions: dict[str, tuple[int, int]] = {}
        rendered: dict[str, str] = {}

        for statement, expression in enumerate(expressions):
            sql = expression.sql(dialect=self.dialect)
            for sentinel in self._extracted:
                if sentinel in positions:
                    continue
                token = rendered.setdefault(
                    sentinel, exp.Placeholder(this=sentinel).sql(dialect=self.dialect)
                )
                index = sql.find(token)
                if index != -1:
                    positions[sentinel] = (statement, index)

        ordered = sorted(
            self._extracted,
            key=lambda s: positions.get(s, (len(expressions), 0)),
        )

        renames: dict[str, t.Optional[str]] = {}
        parameters: list[Parameter] = []

        for sentinel in ordered:
            value, data_type, statement = self._extracted[sentinel]
            name = renames[sentinel] = self._final_name()
            parameters.append(
                Parameter(name=name, value=value, type=data_type, statement=statement)
            )

        for expression in expressions:
            for node in expression.walk():
                if isinstance(node, exp.Placeholder):
                    sentinel = node.text("this")
                    if sentinel in renames:
                        node.set("this", renames[sentinel])

        return parameters

    def _final_name(self) -> t.Optional[str]:
        if self.style == "qmark":
            return None

        while True:
            self.counter += 1
            base = f"p{self.counter}" if self.style == "named" else str(self.counter)
            name = find_new_name(self.taken, base)
            self.taken.add(name)
            if name == base:
                return name

    def _prune(self, node: exp.Expression) -> bool:
        if isinstance(node, _PRUNE_TYPES):
            return True
        return self.keep_limit_literals and isinstance(node, exp.Limit)

    def _placeholder_for(self, node: exp.Expression) -> t.Optional[exp.Placeholder]:
        if (
            isinstance(node, exp.Neg)
            and isinstance(node.this, exp.Literal)
            and node.this.is_number
        ):
            self._skip.add(id(node.this))
            return self._add(node, -node.this.to_py(), "number")
        if isinstance(node, exp.Literal):
            if node.is_number and isinstance(node.parent, (exp.Ordered, exp.Group)):
                return None
            return self._add(node, node.to_py(), "string" if node.is_string else "number")
        if isinstance(node, exp.Boolean):
            return self._add(node, node.this, "boolean")
        if isinstance(node, exp.Null):
            return self._add(node, None, "null")
        if isinstance(node, exp.HexString):
            if node.args.get("is_integer"):
                return self._add(node, int(node.name, 16), "number")
            try:
                return self._add(node, bytes.fromhex(node.name), "bytes")
            except ValueError:
                return None
        if isinstance(node, exp.ByteString):
            return self._add(node, node.name.encode("utf-8"), "bytes")
        return None

    def _add(self, node: exp.Expression, value: t.Any, kind: str) -> exp.Placeholder:
        key = (kind, type(value), value)

        if self.dedupe and key in self._seen:
            sentinel = self._seen[key]
        else:
            while True:
                sentinel = f"{_SENTINEL_PREFIX}{self._sentinel_counter}"
                self._sentinel_counter += 1
                if sentinel not in self.taken:
                    break
            if self.dedupe:
                self._seen[key] = sentinel
            self._extracted[sentinel] = [
                value,
                _parameter_type(node, value, kind),
                self._statement,
            ]

        return exp.Placeholder(this=sentinel)


def _parameter_type(node: exp.Expression, value: t.Any, kind: str) -> exp.DataType:
    parent = node.parent
    if isinstance(parent, exp.Cast):
        to = parent.args.get("to")
        if isinstance(to, exp.DataType) and not to.expressions:
            return to.copy()

    if kind == "string":
        return exp.DataType.build("VARCHAR")
    if kind == "boolean":
        return exp.DataType.build("BOOLEAN")
    if kind == "bytes":
        return exp.DataType.build("VARBINARY")
    if kind == "null":
        return exp.DataType.build("UNKNOWN")

    if isinstance(value, int):
        if _INT32_MIN <= value <= _INT32_MAX:
            return exp.DataType.build("INT")
        if _INT64_MIN <= value <= _INT64_MAX:
            return exp.DataType.build("BIGINT")
        return exp.DataType.build(f"DECIMAL({len(str(abs(value)))}, 0)")

    decimal = t.cast(Decimal, value)
    _, digits, exponent = decimal.as_tuple()
    precision = len(digits)
    scale = 0
    if exponent < 0:
        scale = -exponent
    else:
        precision += exponent
    return exp.DataType.build(f"DECIMAL({max(precision, scale)}, {scale})")
