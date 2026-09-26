import unittest

import sqlglot
from sqlglot import exp
from sqlglot.errors import TenantFilterError
from sqlglot.executor import execute
from sqlglot.optimizer import add_tenant_filter


def filter_sql(sql, tables, tenant_value=1, **kwargs):
    return add_tenant_filter(sqlglot.parse_one(sql), tables, tenant_value, **kwargs).sql()


class TestTenantFilter(unittest.TestCase):
    def test_simple_select(self):
        self.assertEqual(
            filter_sql("SELECT id, amount FROM orders", ["orders"]),
            "SELECT id, amount FROM orders WHERE orders.tenant_id = 1",
        )

    def test_only_listed_tables_are_filtered(self):
        self.assertEqual(
            filter_sql("SELECT * FROM a JOIN b ON a.id = b.id", ["a"]),
            "SELECT * FROM a JOIN b ON a.id = b.id WHERE a.tenant_id = 1",
        )

    def test_qualified_table_names(self):
        self.assertEqual(
            filter_sql("SELECT * FROM main.orders AS o", ["main.orders"]),
            "SELECT * FROM main.orders AS o WHERE o.tenant_id = 1",
        )
        self.assertEqual(
            filter_sql("SELECT * FROM main.orders AS o", ["orders"]),
            "SELECT * FROM main.orders AS o WHERE o.tenant_id = 1",
        )
        self.assertEqual(
            filter_sql("SELECT * FROM main.orders AS o", ["other.orders"]),
            "SELECT * FROM main.orders AS o",
        )

    def test_custom_tenant_column(self):
        self.assertEqual(
            filter_sql("SELECT * FROM orders", ["orders"], tenant_column="org_id"),
            "SELECT * FROM orders WHERE orders.org_id = 1",
        )

    def test_left_join_goes_to_on_clause(self):
        self.assertEqual(
            filter_sql("SELECT * FROM a LEFT JOIN b ON a.id = b.id", ["a", "b"]),
            "SELECT * FROM a LEFT JOIN b ON a.id = b.id AND b.tenant_id = 1 WHERE a.tenant_id = 1",
        )

    def test_right_join_goes_to_on_clause(self):
        self.assertEqual(
            filter_sql("SELECT * FROM a RIGHT JOIN b ON a.id = b.id", ["a", "b"]),
            "SELECT * FROM a RIGHT JOIN b ON a.id = b.id AND a.tenant_id = 1 WHERE b.tenant_id = 1",
        )

    def test_left_join_chain(self):
        self.assertEqual(
            filter_sql(
                "SELECT * FROM a LEFT JOIN b ON a.id = b.id LEFT JOIN c ON b.id = c.id",
                ["a", "b", "c"],
            ),
            "SELECT * FROM a "
            "LEFT JOIN b ON a.id = b.id AND b.tenant_id = 1 "
            "LEFT JOIN c ON b.id = c.id AND c.tenant_id = 1 "
            "WHERE a.tenant_id = 1",
        )

    def test_inner_join_uses_where(self):
        self.assertEqual(
            filter_sql("SELECT * FROM a JOIN b ON a.id = b.id", ["a", "b"]),
            "SELECT * FROM a JOIN b ON a.id = b.id WHERE a.tenant_id = 1 AND b.tenant_id = 1",
        )
        self.assertEqual(
            filter_sql("SELECT * FROM a, b", ["a", "b"]),
            "SELECT * FROM a, b WHERE a.tenant_id = 1 AND b.tenant_id = 1",
        )

    def test_full_join_is_rejected(self):
        with self.assertRaises(TenantFilterError):
            filter_sql("SELECT * FROM a FULL OUTER JOIN b ON a.id = b.id", ["b"])
        with self.assertRaises(TenantFilterError):
            filter_sql("SELECT * FROM a FULL OUTER JOIN b ON a.id = b.id", ["a"])

    def test_unsupported_join_is_rejected(self):
        with self.assertRaises(TenantFilterError):
            filter_sql("SELECT * FROM a LEFT JOIN b USING (id)", ["b"])
        with self.assertRaises(TenantFilterError):
            filter_sql("SELECT * FROM a NATURAL LEFT JOIN b", ["b"])
        with self.assertRaises(TenantFilterError):
            filter_sql("SELECT * FROM a ASOF JOIN b ON a.id = b.id", ["a"])

    def test_inner_join_with_using_uses_where(self):
        self.assertEqual(
            filter_sql("SELECT * FROM a JOIN b USING (id)", ["b"]),
            "SELECT * FROM a JOIN b USING (id) WHERE b.tenant_id = 1",
        )

    def test_non_query_is_rejected(self):
        for sql in ("DELETE FROM a", "UPDATE a SET x = 1"):
            with self.assertRaises(TenantFilterError):
                filter_sql(sql, ["a"])

    def test_cte_shadowing(self):
        self.assertEqual(
            filter_sql("WITH orders AS (SELECT * FROM orders) SELECT * FROM orders", ["orders"]),
            "WITH orders AS (SELECT * FROM orders WHERE orders.tenant_id = 1) SELECT * FROM orders",
        )

    def test_cte_reference_with_alias(self):
        self.assertEqual(
            filter_sql(
                "WITH c AS (SELECT * FROM a) SELECT * FROM c AS x JOIN b ON x.id = b.id",
                ["a", "b", "c"],
            ),
            "WITH c AS (SELECT * FROM a WHERE a.tenant_id = 1) "
            "SELECT * FROM c AS x JOIN b ON x.id = b.id WHERE b.tenant_id = 1",
        )

    def test_derived_table(self):
        self.assertEqual(
            filter_sql("SELECT * FROM (SELECT * FROM a) AS s JOIN b ON s.id = b.id", ["a", "b"]),
            "SELECT * FROM (SELECT * FROM a WHERE a.tenant_id = 1) AS s "
            "JOIN b ON s.id = b.id WHERE b.tenant_id = 1",
        )

    def test_correlated_subquery(self):
        self.assertEqual(
            filter_sql(
                "SELECT * FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b.x = a.x)", ["a", "b"]
            ),
            "SELECT * FROM a WHERE EXISTS(SELECT 1 FROM b WHERE b.x = a.x AND b.tenant_id = 1) "
            "AND a.tenant_id = 1",
        )

    def test_alias_reuse_across_scopes(self):
        self.assertEqual(
            filter_sql(
                "SELECT (SELECT MAX(t.y) FROM b AS t) FROM a AS t",
                ["a", "b"],
            ),
            "SELECT (SELECT MAX(t.y) FROM b AS t WHERE t.tenant_id = 1) "
            "FROM a AS t WHERE t.tenant_id = 1",
        )

    def test_union_branches(self):
        self.assertEqual(
            filter_sql("SELECT x FROM a UNION ALL SELECT x FROM b", ["a", "b"]),
            "SELECT x FROM a WHERE a.tenant_id = 1 UNION ALL SELECT x FROM b WHERE b.tenant_id = 1",
        )

    def test_union_with_cte_and_outer_reference(self):
        self.assertEqual(
            filter_sql(
                "WITH c AS (SELECT x FROM a) SELECT * FROM c UNION ALL SELECT x FROM b",
                ["a", "b", "c"],
            ),
            "WITH c AS (SELECT x FROM a WHERE a.tenant_id = 1) "
            "SELECT * FROM c UNION ALL SELECT x FROM b WHERE b.tenant_id = 1",
        )

    def test_existing_equivalent_condition_is_not_duplicated(self):
        self.assertEqual(
            filter_sql("SELECT * FROM a WHERE a.tenant_id = 1", ["a"]),
            "SELECT * FROM a WHERE a.tenant_id = 1",
        )
        self.assertEqual(
            filter_sql("SELECT * FROM a WHERE a.tenant_id = 1 AND a.x > 0", ["a"]),
            "SELECT * FROM a WHERE a.tenant_id = 1 AND a.x > 0",
        )
        # a different tenant value is not equivalent
        self.assertEqual(
            filter_sql("SELECT * FROM a WHERE a.tenant_id = 2", ["a"]),
            "SELECT * FROM a WHERE a.tenant_id = 2 AND a.tenant_id = 1",
        )
        # an equivalent condition on another table does not count
        self.assertEqual(
            filter_sql("SELECT * FROM a JOIN b ON a.id = b.id WHERE b.tenant_id = 1", ["a", "b"]),
            "SELECT * FROM a JOIN b ON a.id = b.id WHERE b.tenant_id = 1 AND a.tenant_id = 1",
        )

    def test_idempotent(self):
        sql = "SELECT * FROM a LEFT JOIN b ON a.id = b.id WHERE a.x > 0"
        once = add_tenant_filter(sqlglot.parse_one(sql), ["a", "b"], 1)
        twice = add_tenant_filter(once, ["a", "b"], 1)
        self.assertEqual(once.sql(), twice.sql())

    def test_tenant_value_expressions(self):
        self.assertEqual(
            filter_sql("SELECT * FROM orders", ["orders"], "acme"),
            "SELECT * FROM orders WHERE orders.tenant_id = 'acme'",
        )
        self.assertEqual(
            filter_sql("SELECT * FROM orders", ["orders"], exp.Placeholder(this="tenant_id")),
            "SELECT * FROM orders WHERE orders.tenant_id = :tenant_id",
        )

    def test_multiple_dialects(self):
        expression = add_tenant_filter(
            sqlglot.parse_one("SELECT * FROM a LEFT JOIN b ON a.id = b.id"),
            ["a", "b"],
            exp.Placeholder(this="tenant_id"),
        )
        for dialect in ("duckdb", "postgres", "snowflake", "mysql", "tsql"):
            sql = expression.sql(dialect=dialect)
            reparsed = sqlglot.parse_one(sql, read=dialect)
            self.assertEqual(len(list(reparsed.find_all(exp.EQ))), 3, sql)
            self.assertIn("tenant_id", sql)

    def test_copy_default(self):
        expression = sqlglot.parse_one("SELECT * FROM a")
        add_tenant_filter(expression, ["a"], 1)
        self.assertEqual(expression.sql(), "SELECT * FROM a")


class TestTenantFilterExecution(unittest.TestCase):
    """Compare results before/after rewriting on a small dataset.

    DuckDB is not installable in this environment, so these tests run the
    queries with sqlglot's own execution engine; the expected results are
    computed by executing the original query against tenant-filtered tables.
    """

    tables = {
        "users": [
            {"id": 1, "tenant_id": 1, "name": "a"},
            {"id": 2, "tenant_id": 2, "name": "b"},
            {"id": 3, "tenant_id": 1, "name": "c"},
        ],
        "orders": [
            {"id": 10, "user_id": 1, "tenant_id": 1, "amt": 5},
            {"id": 11, "user_id": 1, "tenant_id": 2, "amt": 7},
            {"id": 12, "user_id": 9, "tenant_id": 1, "amt": 3},
            {"id": 13, "user_id": 2, "tenant_id": 2, "amt": 4},
        ],
    }

    def assert_same_results(self, sql, tables, physical_tables):
        sort_key = lambda row: tuple((value is None, value) for value in row)
        for tenant in (1, 2):
            filtered = {
                name: [row for row in rows if row["tenant_id"] == tenant]
                for name, rows in tables.items()
            }
            expected = sorted(map(tuple, execute(sql, tables=filtered).rows), key=sort_key)
            rewritten = add_tenant_filter(sqlglot.parse_one(sql), physical_tables, tenant).sql()
            actual = sorted(map(tuple, execute(rewritten, tables=tables).rows), key=sort_key)
            self.assertEqual(expected, actual, f"tenant={tenant}: {rewritten}")

    def test_left_join_results(self):
        self.assert_same_results(
            "SELECT users.id, orders.amt FROM users "
            "LEFT JOIN orders ON users.id = orders.user_id",
            self.tables,
            ["users", "orders"],
        )

    def test_right_join_results(self):
        self.assert_same_results(
            "SELECT users.id, orders.amt FROM users "
            "RIGHT JOIN orders ON users.id = orders.user_id",
            self.tables,
            ["users", "orders"],
        )

    def test_inner_join_results(self):
        self.assert_same_results(
            "SELECT users.id, orders.amt FROM users JOIN orders ON users.id = orders.user_id",
            self.tables,
            ["users", "orders"],
        )

    def test_cte_results(self):
        self.assert_same_results(
            "WITH o AS (SELECT user_id, amt FROM orders) "
            "SELECT users.id, o.amt FROM users LEFT JOIN o ON users.id = o.user_id",
            self.tables,
            ["users", "orders"],
        )

    def test_union_results(self):
        self.assert_same_results(
            "SELECT id, amt FROM orders UNION ALL SELECT id, tenant_id AS amt FROM users",
            self.tables,
            ["users", "orders"],
        )

    def test_correlated_subquery_results(self):
        self.assert_same_results(
            "SELECT users.id FROM users WHERE EXISTS "
            "(SELECT 1 FROM orders WHERE orders.user_id = users.id)",
            self.tables,
            ["users", "orders"],
        )


if __name__ == "__main__":
    unittest.main()
