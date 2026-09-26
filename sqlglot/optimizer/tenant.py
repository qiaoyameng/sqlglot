from __future__ import annotations

import typing as t

from sqlglot import exp
from sqlglot.errors import OptimizeError
from sqlglot.optimizer.scope import Scope, traverse_scope

if t.TYPE_CHECKING:
    from sqlglot.dialects.dialect import DialectType


def add_tenant_filters(
    expression: exp.Expression,
    tables: t.Collection[str],
    tenant: str | exp.Expression,
    column: str = "tenant_id",
    dialect: DialectType = None,
    copy: bool = True,
) -> exp.Expression:
    """
    Add tenant predicates to every physical table reference in a query.

    For each allowlisted table, a predicate ``<alias>.<column> = <tenant>`` is
    injected based on the scope the table is referenced in:

    - Tables on the preserved side of their joins get the predicate in WHERE.
    - Tables on the nullable side of a LEFT/RIGHT (or SEMI/ANTI) join get the
      predicate in that join's ON clause, so outer joins are not turned into
      inner joins.
    - CTE references and derived tables are resolved through scope sources, so
      a CTE that shadows a physical table name is never filtered by mistake;
      the physical table is filtered inside the CTE body instead.
    - Existing equivalent predicates (either operand order) are not duplicated,
      which also makes the transformation idempotent.

    Structures that cannot be rewritten safely (FULL JOIN, NATURAL/USING outer
    joins, pivoted tables, tables on the nullable side of multiple joins,
    ambiguous aliases, non-query statements) raise OptimizeError instead of
    silently skipping a table.

    Args:
        expression: the query to transform (Select, Union, ...).
        tables: allowlist of physical table names. Entries are matched against
            the bare table name or the dotted ``db.table``/``catalog.db.table`` name.
        tenant: the tenant value expression (e.g. exp.Placeholder, exp.Parameter
            or a literal). A string is parsed as an expression.
        column: the tenant column name.
        dialect: dialect used to parse ``tenant`` when it is a string.
        copy: if True, transform a copy of the expression.

    Returns:
        The transformed expression.
    """
    if isinstance(tenant, str):
        tenant = exp.maybe_parse(tenant, dialect=dialect)
    if not isinstance(tenant, exp.Expression):
        raise ValueError(f"Invalid tenant expression: {tenant!r}")
    if not isinstance(expression, exp.Query):
        raise OptimizeError(
            f"Cannot add tenant filters to {type(expression).__name__}, expected a query"
        )

    root = expression.copy() if copy else expression
    names = set(tables)

    # Plan everything before mutating, so a rejection leaves the tree untouched.
    plans: list[tuple[exp.Select | exp.Join, exp.EQ]] = []
    for scope in traverse_scope(root):
        select = scope.expression
        if isinstance(select, exp.Select):
            _plan_scope(scope, select, names, tenant, column, plans)

    for target, condition in plans:
        if isinstance(target, exp.Join):
            target.on(condition, copy=False)
        else:
            target.where(condition, copy=False)

    return root


def _plan_scope(
    scope: Scope,
    select: exp.Select,
    names: t.Set[str],
    tenant: exp.Expression,
    column: str,
    plans: list[tuple[exp.Select | exp.Join, exp.EQ]],
) -> None:
    from_ = select.args.get("from_")
    if not from_:
        return

    records: list[tuple[exp.Join, list[exp.Table], list[exp.Table]]] = []
    tables = _expand(from_.this, records)
    for join in select.args.get("joins") or []:
        right = _expand(join.this, records)
        records.append((join, tables, right))
        tables = tables + right

    # Raises OptimizeError on ambiguous duplicate aliases in this scope.
    selected = scope.selected_sources
    semi_anti = scope.semi_or_anti_join_tables

    for table in tables:
        if not _matches(table, names):
            continue

        entry = selected.get(table.alias_or_name)
        if entry is not None:
            if entry[1] is not table:
                # A reference to a CTE or parent source, not the physical table.
                continue
        elif not (
            table.alias_or_name in semi_anti and scope.sources.get(table.alias_or_name) is table
        ):
            continue

        if table.args.get("pivots"):
            raise OptimizeError(
                f"Cannot add tenant filter to pivoted table {table.alias_or_name!r}"
            )

        nullable = [
            (join, left, right)
            for join, left, right in records
            if _is_nullable(join, table, left, right)
        ]

        if not nullable:
            where = select.args.get("where")
            existing = where.this if where else None
            condition = _condition(table, column, tenant)
            if not _has_equivalent(existing, condition):
                plans.append((select, condition))
        elif len(nullable) == 1:
            join, left, right = nullable[0]
            side = join.side
            on_right = _contains(right, table)
            if (on_right and (side == "LEFT" or join.is_semi_or_anti_join)) or (
                not on_right and side == "RIGHT" and _contains(left, table)
            ):
                _check_join_target(join, table)
                condition = _condition(table, column, tenant)
                if not _has_equivalent(join.args.get("on"), condition):
                    plans.append((join, condition))
            else:
                raise OptimizeError(
                    f"Cannot safely add tenant filter to table {table.alias_or_name!r} "
                    f"because of a {side or 'FULL'} JOIN"
                )
        else:
            raise OptimizeError(
                f"Cannot safely add tenant filter to table {table.alias_or_name!r}: "
                "the table is on the nullable side of multiple joins"
            )


def _process_group(
    head: exp.Expression,
    joins: list[exp.Join],
    records: list[tuple[exp.Join, list[exp.Table], list[exp.Table]]],
) -> list[exp.Table]:
    """
    Collect the physical tables of a join group (a FROM head plus its joins),
    recursing into nested join groups like ``FROM (a JOIN b)``. For every join,
    record the tables of its left and right operands.
    """
    left = _plain_tables(head)
    for join in joins:
        right = _expand(join.this, records)
        records.append((join, left, right))
        left = left + right
    return left


def _expand(
    container: exp.Expression,
    records: list[tuple[exp.Join, list[exp.Table], list[exp.Table]]],
) -> list[exp.Table]:
    node = container
    while isinstance(node, (exp.Subquery, exp.Final)) and not isinstance(node.this, exp.Query):
        node = node.this
    if isinstance(node, exp.Table) and node.args.get("joins"):
        return _process_group(node, node.args["joins"], records)
    return _plain_tables(container)


def _plain_tables(node: exp.Expression) -> list[exp.Table]:
    # Tables directly under node, without crossing into subqueries (those are
    # separate scopes) or nested joins (handled by _process_group).
    return [
        n
        for n in node.walk(
            bfs=False, prune=lambda n: n is not node and isinstance(n, (exp.Query, exp.Join))
        )
        if isinstance(n, exp.Table)
    ]


def _matches(table: exp.Table, names: t.Set[str]) -> bool:
    return table.name in names or ".".join(part.name for part in table.parts) in names


def _contains(tables: list[exp.Table], table: exp.Table) -> bool:
    return any(t is table for t in tables)


def _is_nullable(
    join: exp.Join, table: exp.Table, left: list[exp.Table], right: list[exp.Table]
) -> bool:
    side = join.side
    if side == "LEFT" or join.is_semi_or_anti_join:
        return _contains(right, table)
    if side == "RIGHT":
        return _contains(left, table)
    if side == "FULL":
        return True
    return False


def _check_join_target(join: exp.Join, table: exp.Table) -> None:
    if join.method == "NATURAL" or join.args.get("natural"):
        raise OptimizeError(
            f"Cannot add tenant filter to table {table.alias_or_name!r} through a NATURAL join"
        )
    for arg, description in (("using", "a USING"), ("match_condition", "an ASOF")):
        if join.args.get(arg):
            raise OptimizeError(
                f"Cannot add tenant filter to table {table.alias_or_name!r} "
                f"through {description} join"
            )


def _condition(table: exp.Table, column: str, tenant: exp.Expression) -> exp.EQ:
    return exp.column(column, table=table.alias_or_name).eq(tenant.copy())


def _has_equivalent(clause: exp.Expression | None, condition: exp.EQ) -> bool:
    if clause is None:
        return False
    conjuncts = clause.flatten() if isinstance(clause, exp.And) else (clause,)
    this, other = condition.this, condition.expression
    return any(
        isinstance(e, exp.EQ)
        and (
            (e.this == this and e.expression == other)
            or (e.this == other and e.expression == this)
        )
        for e in conjuncts
    )
