"""
.. include:: ../posts/parameterization.md

----
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass, field
from decimal import Decimal
from itertools import count

from sqlglot import expressions as exp
from sqlglot.dialects.dialect import Dialect

if t.TYPE_CHECKING:
    from sqlglot.dialects.dialect import DialectType

NAMED = "named"
"""Generate named placeholders, e.g. `:p1` / `$p1` / `%(p1)s`, depending on the dialect."""

NUMBERED = "numbered"
"""Generate numbered parameters, e.g. `$1` / `@1`, depending on the dialect."""

QMARK = "qmark"
"""Generate positional `?` placeholders."""

STYLES = (NAMED, NUMBERED, QMARK)

LIMIT_PARAMETERIZABLE_DIALECTS = ("duckdb", "mysql", "postgres", "sqlite")
"""Dialects whose LIMIT / OFFSET / FETCH clauses accept bound parameters."""

_NON_PARAMETERIZABLE_ANCESTORS = (exp.DataType, exp.Interval)
_LIMIT_ANCESTORS = (exp.Limit, exp.Offset, exp.Fetch)

_DATE_TYPES = {exp.DataType.Type.DATE, exp.DataType.Type.DATE32}
_TIME_TYPES = {exp.DataType.Type.TIME, exp.DataType.Type.TIMETZ}


@dataclass
class Parameter:
    """A single parameter extracted from a literal, in execution (text) order."""

    name: str
    """The placeholder name, e.g. `p1`. For `qmark` style this is a bookkeeping name only."""

    value: str | int | Decimal
    """The exact Python value to bind. Never a lossy float."""

    type: exp.DataType
    """The inferred SQL type of the parameter."""


@dataclass
class ParameterizedQuery:
    """The result of `parameterize`: new expression(s) plus their ordered parameters."""

    expressions: list[exp.Expression]
    parameters: list[Parameter] = field(default_factory=list)
    dialect: DialectType = None
    style: str = NAMED

    @property
    def expression(self) -> exp.Expression:
        """The single parameterized expression. Raises if there are multiple statements."""
        if len(self.expressions) != 1:
            raise ValueError(
                f"Expected a single statement, found {len(self.expressions)}. "
                "Use `expressions` instead."
            )
        return self.expressions[0]

    @property
    def values(self) -> list[str | int | Decimal]:
        """The parameter values in execution order."""
        return [parameter.value for parameter in self.parameters]

    def sql(self, dialect: DialectType = None, **opts: t.Any) -> list[str]:
        """Generates SQL for each parameterized expression, one string per statement."""
        return [expression.sql(dialect or self.dialect, **opts) for expression in self.expressions]


def parameterize(
    expression: exp.Expression | t.Sequence[exp.Expression],
    dialect: DialectType = None,
    style: str = NAMED,
    prefix: str = "p",
    parameterize_limit: bool | None = None,
) -> ParameterizedQuery:
    """
    Replaces literals in one or more expressions with placeholders and returns the
    extracted parameters in execution (SQL text) order.

    Only string and numeric literals (`exp.Literal`) are parameterized. The following
    positions keep their literals, since they are not bind-compatible in general:

    - Data type parameters, e.g. the `10` in `CAST(x AS VARCHAR(10))`.
    - `INTERVAL` values and units, e.g. `INTERVAL '1' DAY`.
    - `ORDER BY` ordinals, e.g. the `1` in `ORDER BY 1`.
    - `LIMIT` / `OFFSET` / `FETCH` counts, unless `parameterize_limit` is enabled
      (enabled by default for `LIMIT_PARAMETERIZABLE_DIALECTS` only).

    Booleans, `NULL` and hex/bit strings are also preserved. Existing `exp.Placeholder`
    and `exp.Parameter` nodes are left untouched and new names never collide with them.
    Literals with identical text *and* kind (string vs. number) share one parameter;
    e.g. `1` and `'1'` yield two distinct parameters. Values are extracted exactly:
    integers stay `int`, decimals become `Decimal`, dates and binary-looking text stay
    strings -- nothing is routed through a lossy `float`.

    The input expression(s) are never mutated; the result contains copies.

    Args:
        expression: a single expression or a sequence of statements.
        dialect: the target dialect, used to decide LIMIT support and as the default
            dialect when generating SQL from the result.
        style: one of `"named"`, `"numbered"` or `"qmark"`.
        prefix: name prefix for generated placeholders (`named` style only).
        parameterize_limit: whether to parameterize LIMIT / OFFSET / FETCH counts.
            Defaults to True for `LIMIT_PARAMETERIZABLE_DIALECTS`, False otherwise.

    Returns:
        A `ParameterizedQuery` with the new expression(s) and ordered parameters.
    """
    if style not in STYLES:
        raise ValueError(f"Unknown placeholder style: {style!r}. Expected one of {STYLES}.")

    if parameterize_limit is None:
        name = type(Dialect.get_or_raise(dialect)).__name__.lower()
        parameterize_limit = name in LIMIT_PARAMETERIZABLE_DIALECTS

    statements = [expression] if isinstance(expression, exp.Expression) else list(expression)
    for statement in statements:
        if not isinstance(statement, exp.Expression):
            raise TypeError(
                f"Expected an Expression or a sequence of Expressions, got: {statement!r}."
            )

    result = ParameterizedQuery(expressions=[], dialect=dialect, style=style)
    taken: set[str] = set()
    for statement in statements:
        for node in statement.walk():
            if isinstance(node, exp.Placeholder):
                if node.this:
                    taken.add(node.name)
            elif isinstance(node, exp.Parameter) and node.this is not None:
                taken.add(node.this.name)

    sequence = count(1)
    seen: dict[tuple[bool, str], str] = {}

    def next_name() -> str:
        while True:
            index = next(sequence)
            name = str(index) if style == NUMBERED else f"{prefix}{index}"
            if name not in taken:
                taken.add(name)
                return name

    for statement in statements:
        parameterized = statement.copy()
        literals = [
            node
            for node in parameterized.dfs()
            if isinstance(node, exp.Literal) and _is_parameterizable(node, parameterize_limit)
        ]
        for literal in literals:
            key = (literal.is_string, literal.this)
            name = seen.get(key)
            if name is None:
                name = next_name()
                seen[key] = name
                result.parameters.append(
                    Parameter(name=name, value=literal.to_py(), type=_infer_type(literal))
                )
            literal.replace(_placeholder(style, name))
        result.expressions.append(parameterized)

    return result


def _placeholder(style: str, name: str) -> exp.Expression:
    if style == QMARK:
        return exp.Placeholder()
    if style == NUMBERED:
        return exp.Parameter(this=exp.Literal.number(name))
    return exp.Placeholder(this=name)


def _is_parameterizable(literal: exp.Literal, parameterize_limit: bool) -> bool:
    parent = literal.parent
    if isinstance(parent, exp.Ordered) and literal.is_number:
        return False

    node = parent
    while node is not None:
        if isinstance(node, _NON_PARAMETERIZABLE_ANCESTORS):
            return False
        if not parameterize_limit and isinstance(node, _LIMIT_ANCESTORS):
            return False
        node = node.parent
    return True


def _infer_type(literal: exp.Literal) -> exp.DataType:
    parent = literal.parent
    if isinstance(parent, exp.Cast) and parent.to:
        dtype = parent.to.this
        if dtype in _DATE_TYPES:
            return exp.DataType.build("DATE")
        if dtype in _TIME_TYPES:
            return exp.DataType.build("TIME")
        if dtype in exp.DataType.TEMPORAL_TYPES:
            return exp.DataType.build("TIMESTAMP")

    if literal.is_string:
        return exp.DataType.build("VARCHAR")

    value = literal.to_py()
    if isinstance(value, int):
        return exp.DataType.build("INT" if -(2**31) <= value < 2**31 else "BIGINT")

    _, digits, exponent = value.as_tuple()
    scale = max(-exponent, 0)
    precision = max(len(digits), 1) + max(exponent, 0)
    return exp.DataType.build(f"DECIMAL({precision}, {scale})")
