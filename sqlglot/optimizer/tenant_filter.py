from __future__ import annotations

import typing as t

from sqlglot import exp
from sqlglot.errors import TenantFilterError
from sqlglot.optimizer.scope import Scope, traverse_scope


def add_tenant_filter(
    expression: exp.Expr,
    tables: t.Collection[str],
    tenant_value: t.Any,
    tenant_column: str = "tenant_id",
    copy: bool = True,
) -> exp.Expr:
    """
    Restrict physical tables to a tenant by adding ``tenant_column = tenant_value``
    predicates to every scope that references them.

    The traversal is scope-aware: CTEs and derived tables that shadow a physical
    table name are left untouched, each UNION branch and (correlated) subquery is
    handled in its own scope, and predicates for tables on the null-supplying side
    of an outer join are added to that join's ``ON`` clause instead of ``WHERE``,
    so outer joins are never silently turned into inner joins.

    Structures that cannot be rewritten safely (e.g. a ``FULL OUTER JOIN`` involving
    a filtered table, or a null-supplying table joined with ``NATURAL``/``USING``)
    raise a ``TenantFilterError`` instead of returning a partially filtered query.

    Args:
        expression: the query to rewrite. Must be a ``SELECT``/set operation;
            anything else is rejected.
        tables: names of the physical tables to filter. Entries are either bare
            table names (``"orders"``) or qualified names (``"main.orders"``).
        tenant_value: the tenant value expression supplied by the caller (e.g.
            ``exp.Placeholder(this="tenant_id")``). Plain Python values are
            converted to literals with ``exp.convert``.
        tenant_column: name of the tenant column on every filtered table.
        copy: whether to copy the input expression before rewriting.

    Returns:
        The rewritten expression.

    Examples:
        >>> import sqlglot
        >>> from sqlglot import exp
        >>> from sqlglot.optimizer.tenant_filter import add_tenant_filter
        >>> sql = "SELECT * FROM a LEFT JOIN b ON a.id = b.id"
        >>> add_tenant_filter(sqlglot.parse_one(sql), ["a", "b"], exp.convert(1)).sql()
        'SELECT * FROM a LEFT JOIN b ON a.id = b.id AND b.tenant_id = 1 WHERE a.tenant_id = 1'
    """
    if not isinstance(expression, exp.Query):
        raise TenantFilterError(
            f"Cannot apply tenant filters to {type(expression).__name__}, "
            "only queries are supported"
        )

    if not isinstance(tenant_value, exp.Expression):
        tenant_value = exp.convert(tenant_value)

    table_names = {tuple(name.split(".")) for name in tables}
    root = expression.copy() if copy else expression

    for scope in traverse_scope(root):
        if isinstance(scope.expression, exp.Select):
            _add_scope_filters(scope, table_names, tenant_column, tenant_value)

    return root


def _add_scope_filters(
    scope: Scope,
    table_names: t.Set[t.Tuple[str, ...]],
    tenant_column: str,
    tenant_value: exp.Expression,
) -> None:
    select = scope.expression
    from_ = select.args.get("from_")
    if not from_:
        return

    joins = select.args.get("joins") or []
    chain = [_unwrap(from_.this)] + [_unwrap(join.this) for join in joins]
    join_kinds = [_join_kind(join) for join in joins]

    for node in chain:
        if isinstance(node, exp.Subquery) and not _is_derived_table(node):
            raise TenantFilterError(
                f"Cannot apply tenant filters to {select.sql()!r}: "
                "parenthesized join groups are not supported"
            )

    table_source_count = sum(
        1 for node in chain if isinstance(node, exp.Table) and _is_physical_table(scope, node)
    )

    for position, node in enumerate(chain):
        if (
            not isinstance(node, exp.Table)
            or not _is_physical_table(scope, node)
            or not _matches(node, table_names)
        ):
            continue

        if node.args.get("joins") or node.args.get("pivots"):
            raise TenantFilterError(
                f"Cannot apply tenant filters to {select.sql()!r}: "
                "joins or pivots attached to table sources are not supported"
            )

        if "unknown" in join_kinds:
            raise TenantFilterError(
                f"Cannot apply tenant filters to {select.sql()!r}: unsupported join type"
            )

        if any(
            kind == "full" and position <= join_position
            for join_position, kind in enumerate(join_kinds, start=1)
        ):
            raise TenantFilterError(
                f"Cannot apply tenant filters to {select.sql()!r}: "
                f"table {node.alias_or_name!r} participates in a FULL OUTER JOIN"
            )

        null_supplying = [
            index
            for index, kind in enumerate(join_kinds)
            if (kind == "left" and position == index + 1)
            or (kind == "right" and position <= index)
        ]

        source_name = node.alias_or_name
        predicate = exp.EQ(
            this=exp.column(tenant_column, table=source_name),
            expression=tenant_value.copy(),
        )

        if null_supplying:
            join = joins[min(null_supplying)]
            if join.method == "NATURAL" or join.args.get("using"):
                raise TenantFilterError(
                    f"Cannot apply tenant filters to {select.sql()!r}: table {source_name!r} is on "
                    "the null-supplying side of a NATURAL/USING join, so its tenant predicate "
                    "cannot be added to the join's ON clause"
                )
            unqualified_ok = table_source_count == 1
            if not _has_equivalent(
                join.args.get("on"),
                source_name,
                tenant_column,
                tenant_value,
                allow_unqualified=unqualified_ok,
            ) and not _has_equivalent(
                select.args.get("where"),
                source_name,
                tenant_column,
                tenant_value,
                allow_unqualified=unqualified_ok,
            ):
                on = join.args.get("on")
                join.set("on", exp.and_(on, predicate) if on else predicate)
        elif not _has_equivalent(
            select.args.get("where"),
            source_name,
            tenant_column,
            tenant_value,
            allow_unqualified=table_source_count == 1,
        ):
            where = select.args.get("where")
            if where:
                where.set("this", exp.and_(where.this, predicate))
            else:
                select.set("where", exp.Where(this=predicate))


def _unwrap(node: exp.Expr) -> exp.Expr:
    while isinstance(node, exp.Final):
        node = node.this
    return node


def _is_derived_table(node: exp.Subquery) -> bool:
    return bool(node.alias) or isinstance(node.this, exp.Query)


def _is_physical_table(scope: Scope, table: exp.Table) -> bool:
    # CTEs and other subquery sources shadow physical table names; they resolve
    # to a Scope instead of the exp.Table node itself
    return not isinstance(scope.sources.get(table.alias_or_name), Scope)


def _matches(table: exp.Table, table_names: t.Set[t.Tuple[str, ...]]) -> bool:
    parts = tuple(part for part in (table.catalog, table.db, table.name) if part)
    return parts in table_names or (table.name,) in table_names


def _join_kind(join: exp.Join) -> str:
    side = join.side
    kind = join.kind

    if side == "FULL" or kind == "FULL":
        return "full"
    if side == "LEFT":
        return "left"
    if side == "RIGHT":
        return "right"
    if kind in ("SEMI", "ANTI"):
        # The right-hand side of SEMI/ANTI joins is never emitted, so filtering
        # it in the ON clause is always safe
        return "left"
    if (
        not side
        and kind in ("", "INNER", "CROSS", "STRAIGHT_JOIN")
        and join.method in ("", "NATURAL")
    ):
        return "inner"
    return "unknown"


def _has_equivalent(
    clause: t.Optional[exp.Expr],
    source_name: str,
    tenant_column: str,
    tenant_value: exp.Expression,
    allow_unqualified: bool = True,
) -> bool:
    if clause is None:
        return False

    expression = clause.this if isinstance(clause, exp.Where) else clause

    for conjunct in _conjuncts(expression):
        conjunct = conjunct.unnest()
        if not isinstance(conjunct, exp.EQ):
            continue
        for column, other in ((conjunct.left, conjunct.right), (conjunct.right, conjunct.left)):
            if (
                isinstance(column, exp.Column)
                and column.name == tenant_column
                and (column.table == source_name if column.table else allow_unqualified)
                and other == tenant_value
            ):
                return True
    return False


def _conjuncts(expression: exp.Expr) -> t.Iterator[exp.Expr]:
    if isinstance(expression, exp.And):
        yield from _conjuncts(expression.this)
        yield from _conjuncts(expression.expression)
    else:
        yield expression
