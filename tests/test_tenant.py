import unittest

from sqlglot import exp, parse_one
from sqlglot.errors import OptimizeError
from sqlglot.executor import execute
from sqlglot.optimizer.tenant import add_tenant_filters

try:
    import duckdb
except ImportError:
    duckdb = None

TABLES = ["users", "events", "orders"]
TENANT = exp.Placeholder(this="tenant")


def transform(sql, tables=TABLES, tenant=TENANT, **kwargs):
    return add_tenant_filters(parse_one(sql), tables, tenant, **kwargs).sql()


class TestTenantFilters(unittest.TestCase):
    def test_simple_select(self):
        self.assertEqual(
            transform("SELECT * FROM users"),
            "SELECT * FROM users WHERE users.tenant_id = :tenant",
        )

    def test_appends_to_existing_where(self):
        self.assertEqual(
            transform("SELECT * FROM users u WHERE u.active = 1"),
            "SELECT * FROM users AS u WHERE u.active = 1 AND u.tenant_id = :tenant",
        )

    def test_custom_column_and_tenant_expression(self):
        self.assertEqual(
            transform("SELECT * FROM users", tenant=exp.Literal.number(7), column="org_id"),
            "SELECT * FROM users WHERE users.org_id = 7",
        )

    def test_string_tenant_is_parsed(self):
        self.assertEqual(
            transform("SELECT * FROM users", tenant="42"),
            "SELECT * FROM users WHERE users.tenant_id = 42",
        )

    def test_qualified_table_allowlist(self):
        self.assertEqual(
            transform("SELECT * FROM db.users", tables=["db.users"]),
            "SELECT * FROM db.users WHERE users.tenant_id = :tenant",
        )
        self.assertEqual(
            transform("SELECT * FROM other.users", tables=["db.users"]),
            "SELECT * FROM other.users",
        )

    def test_non_allowlisted_tables_untouched(self):
        self.assertEqual(
            transform("SELECT * FROM users u JOIN logs l ON u.id = l.uid"),
            "SELECT * FROM users AS u JOIN logs AS l ON u.id = l.uid "
            "WHERE u.tenant_id = :tenant",
        )

    def test_left_join_goes_to_on_clause(self):
        self.assertEqual(
            transform("SELECT * FROM users u LEFT JOIN events e ON u.id = e.uid"),
            "SELECT * FROM users AS u LEFT JOIN events AS e "
            "ON u.id = e.uid AND e.tenant_id = :tenant WHERE u.tenant_id = :tenant",
        )

    def test_right_join_goes_to_on_clause(self):
        self.assertEqual(
            transform("SELECT * FROM users u RIGHT JOIN events e ON u.id = e.uid"),
            "SELECT * FROM users AS u RIGHT JOIN events AS e "
            "ON u.id = e.uid AND u.tenant_id = :tenant WHERE e.tenant_id = :tenant",
        )

    def test_left_join_chain(self):
        self.assertEqual(
            transform(
                "SELECT * FROM users u "
                "LEFT JOIN events e ON u.id = e.uid "
                "LEFT JOIN orders o ON o.uid = u.id"
            ),
            "SELECT * FROM users AS u "
            "LEFT JOIN events AS e ON u.id = e.uid AND e.tenant_id = :tenant "
            "LEFT JOIN orders AS o ON o.uid = u.id AND o.tenant_id = :tenant "
            "WHERE u.tenant_id = :tenant",
        )

    def test_inner_join_after_left_join(self):
        self.assertEqual(
            transform(
                "SELECT * FROM users u "
                "LEFT JOIN events e ON u.id = e.uid "
                "JOIN orders o ON o.uid = u.id"
            ),
            "SELECT * FROM users AS u "
            "LEFT JOIN events AS e ON u.id = e.uid AND e.tenant_id = :tenant "
            "JOIN orders AS o ON o.uid = u.id "
            "WHERE u.tenant_id = :tenant AND o.tenant_id = :tenant",
        )

    def test_semi_join_rhs_goes_to_on_clause(self):
        self.assertEqual(
            transform("SELECT * FROM users u LEFT SEMI JOIN events e ON u.id = e.uid"),
            "SELECT * FROM users AS u LEFT SEMI JOIN events AS e "
            "ON u.id = e.uid AND e.tenant_id = :tenant WHERE u.tenant_id = :tenant",
        )

    def test_nested_join_group_in_from(self):
        self.assertEqual(
            transform("SELECT * FROM (users LEFT JOIN events ON users.id = events.uid)"),
            "SELECT * FROM (users LEFT JOIN events "
            "ON users.id = events.uid AND events.tenant_id = :tenant) "
            "WHERE users.tenant_id = :tenant",
        )

    def test_nested_join_group_left_of_right_join(self):
        self.assertEqual(
            transform(
                "SELECT * FROM (users JOIN events ON users.id = events.uid) "
                "RIGHT JOIN orders ON events.uid = orders.uid"
            ),
            "SELECT * FROM (users JOIN events ON users.id = events.uid) "
            "RIGHT JOIN orders ON (events.uid = orders.uid AND users.tenant_id = :tenant) "
            "AND events.tenant_id = :tenant WHERE orders.tenant_id = :tenant",
        )

    def test_nested_join_group_right_of_left_join(self):
        self.assertEqual(
            transform(
                "SELECT * FROM orders "
                "LEFT JOIN (users JOIN events ON users.id = events.uid) ON orders.uid = users.id"
            ),
            "SELECT * FROM orders "
            "LEFT JOIN (users JOIN events ON users.id = events.uid) "
            "ON (orders.uid = users.id AND users.tenant_id = :tenant) "
            "AND events.tenant_id = :tenant WHERE orders.tenant_id = :tenant",
        )

    def test_cte_shadowing_physical_table(self):
        # The CTE named "users" shadows the physical table: the outer reference
        # must not be filtered, the physical table inside the CTE body must be.
        self.assertEqual(
            transform("WITH users AS (SELECT id FROM orders) SELECT * FROM users"),
            "WITH users AS (SELECT id FROM orders WHERE orders.tenant_id = :tenant) "
            "SELECT * FROM users",
        )

    def test_cte_reference_joined_with_physical_table(self):
        self.assertEqual(
            transform(
                "WITH users AS (SELECT id FROM orders) "
                "SELECT * FROM users u LEFT JOIN events e ON u.id = e.uid"
            ),
            "WITH users AS (SELECT id FROM orders WHERE orders.tenant_id = :tenant) "
            "SELECT * FROM users AS u LEFT JOIN events AS e "
            "ON u.id = e.uid AND e.tenant_id = :tenant",
        )

    def test_correlated_subquery(self):
        self.assertEqual(
            transform(
                "SELECT * FROM users u WHERE EXISTS (SELECT 1 FROM events e WHERE e.uid = u.id)"
            ),
            "SELECT * FROM users AS u WHERE EXISTS("
            "SELECT 1 FROM events AS e WHERE e.uid = u.id AND e.tenant_id = :tenant"
            ") AND u.tenant_id = :tenant",
        )

    def test_alias_reused_across_scopes(self):
        self.assertEqual(
            transform(
                "SELECT * FROM users u WHERE u.id IN "
                "(SELECT uid FROM events u WHERE u.tenant_id IS NOT NULL)"
            ),
            "SELECT * FROM users AS u WHERE u.id IN "
            "(SELECT uid FROM events AS u WHERE NOT u.tenant_id IS NULL "
            "AND u.tenant_id = :tenant) AND u.tenant_id = :tenant",
        )

    def test_derived_table(self):
        self.assertEqual(
            transform("SELECT * FROM (SELECT * FROM users) u LEFT JOIN events e ON u.id = e.uid"),
            "SELECT * FROM (SELECT * FROM users WHERE users.tenant_id = :tenant) AS u "
            "LEFT JOIN events AS e ON u.id = e.uid AND e.tenant_id = :tenant",
        )

    def test_union_branches_filtered_separately(self):
        self.assertEqual(
            transform("SELECT * FROM users UNION ALL SELECT * FROM users"),
            "SELECT * FROM users WHERE users.tenant_id = :tenant "
            "UNION ALL SELECT * FROM users WHERE users.tenant_id = :tenant",
        )

    def test_union_with_outer_join_branches(self):
        self.assertEqual(
            transform(
                "SELECT * FROM users u LEFT JOIN events e ON u.id = e.uid "
                "UNION ALL SELECT * FROM users"
            ),
            "SELECT * FROM users AS u LEFT JOIN events AS e "
            "ON u.id = e.uid AND e.tenant_id = :tenant WHERE u.tenant_id = :tenant "
            "UNION ALL SELECT * FROM users WHERE users.tenant_id = :tenant",
        )

    def test_existing_equivalent_condition_not_duplicated(self):
        self.assertEqual(
            transform("SELECT * FROM users u WHERE u.tenant_id = :tenant AND u.active = 1"),
            "SELECT * FROM users AS u WHERE u.tenant_id = :tenant AND u.active = 1",
        )

    def test_existing_equivalent_condition_flipped_operands(self):
        self.assertEqual(
            transform("SELECT * FROM users u WHERE :tenant = u.tenant_id"),
            "SELECT * FROM users AS u WHERE :tenant = u.tenant_id",
        )

    def test_existing_equivalent_condition_in_join_on(self):
        self.assertEqual(
            transform(
                "SELECT * FROM users u LEFT JOIN events e "
                "ON u.id = e.uid AND e.tenant_id = :tenant"
            ),
            "SELECT * FROM users AS u LEFT JOIN events AS e "
            "ON u.id = e.uid AND e.tenant_id = :tenant WHERE u.tenant_id = :tenant",
        )

    def test_idempotent(self):
        sql = (
            "WITH users AS (SELECT id FROM orders) "
            "SELECT * FROM users u LEFT JOIN events e ON u.id = e.uid"
        )
        once = add_tenant_filters(parse_one(sql), TABLES, TENANT)
        twice = add_tenant_filters(once, TABLES, TENANT)
        self.assertEqual(once.sql(), twice.sql())

    def test_input_not_mutated_by_default(self):
        query = parse_one("SELECT * FROM users u LEFT JOIN events e ON u.id = e.uid")
        before = query.sql()
        add_tenant_filters(query, TABLES, TENANT)
        self.assertEqual(query.sql(), before)

    def test_multiple_dialects(self):
        transformed = add_tenant_filters(
            parse_one(
                "WITH users AS (SELECT id FROM orders) "
                "SELECT * FROM users u LEFT JOIN events e ON u.id = e.uid"
            ),
            TABLES,
            TENANT,
        )
        expected = (
            "WITH users AS (SELECT id FROM orders WHERE orders.tenant_id = {p}) "
            "SELECT * FROM users AS u LEFT JOIN events AS e "
            "ON u.id = e.uid AND e.tenant_id = {p}"
        )
        for dialect, placeholder in (
            ("duckdb", "$tenant"),
            ("postgres", "%(tenant)s"),
            ("mysql", ":tenant"),
            ("snowflake", ":tenant"),
        ):
            sql = transformed.sql(dialect)
            self.assertEqual(sql, expected.format(p=placeholder))
            # The generated SQL must remain parseable in the target dialect.
            self.assertEqual(parse_one(sql, dialect=dialect).sql(dialect), sql)

        # TSQL explicitly aliases projections; just check it round-trips.
        sql = transformed.sql("tsql")
        self.assertIn(":tenant", sql)
        self.assertEqual(parse_one(sql, dialect="tsql").sql("tsql"), sql)

    def test_full_join_rejected(self):
        with self.assertRaises(OptimizeError):
            transform("SELECT * FROM users u FULL JOIN events e ON u.id = e.uid")

    def test_natural_join_rejected(self):
        with self.assertRaises(OptimizeError):
            transform("SELECT * FROM users u NATURAL LEFT JOIN events e")

    def test_using_join_rejected(self):
        with self.assertRaises(OptimizeError):
            transform("SELECT * FROM users u LEFT JOIN events e USING (id)")

    def test_nullable_side_of_multiple_joins_rejected(self):
        with self.assertRaises(OptimizeError):
            transform(
                "SELECT * FROM users u "
                "LEFT JOIN events e ON u.id = e.uid "
                "RIGHT JOIN orders o ON o.uid = e.uid"
            )

    def test_nested_nullable_side_of_multiple_joins_rejected(self):
        with self.assertRaises(OptimizeError):
            transform(
                "SELECT * FROM (users u LEFT JOIN events e ON u.id = e.uid) "
                "RIGHT JOIN orders o ON o.uid = u.id"
            )

    def test_pivoted_table_rejected(self):
        with self.assertRaises(OptimizeError):
            transform(
                "SELECT * FROM users PIVOT (sum(id) FOR tenant_id IN (1, 2))",
            )

    def test_ambiguous_alias_rejected(self):
        with self.assertRaises(OptimizeError):
            transform("SELECT * FROM users JOIN users ON 1 = 1")

    def test_non_query_rejected(self):
        with self.assertRaises(OptimizeError):
            transform("DELETE FROM users WHERE id = 1")

    def test_rejection_leaves_tree_untouched(self):
        query = parse_one("SELECT * FROM users u FULL JOIN events e ON u.id = e.uid")
        before = query.sql()
        with self.assertRaises(OptimizeError):
            add_tenant_filters(query, TABLES, TENANT, copy=False)
        self.assertEqual(query.sql(), before)


class TestTenantExecution(unittest.TestCase):
    schema = {
        "users": {"id": "INT", "tenant_id": "INT"},
        "events": {"id": "INT", "uid": "INT", "tenant_id": "INT"},
    }
    data = {
        "users": [
            {"id": 1, "tenant_id": 1},
            {"id": 2, "tenant_id": 1},
            {"id": 3, "tenant_id": 2},
        ],
        "events": [
            {"id": 10, "uid": 1, "tenant_id": 1},
            {"id": 11, "uid": 1, "tenant_id": 2},
            {"id": 12, "uid": 3, "tenant_id": 1},
            {"id": 13, "uid": 99, "tenant_id": 1},
        ],
    }

    def run_tenant(self, sql, tenant):
        transformed = add_tenant_filters(
            parse_one(sql), TABLES, exp.Literal.number(tenant)
        ).sql("duckdb")
        return set(
            execute(transformed, schema=self.schema, tables=self.data).rows
        )

    def test_left_join_results(self):
        sql = "SELECT u.id, e.id FROM users u LEFT JOIN events e ON u.id = e.uid"
        # Tenant 1 keeps users 1 and 2; event 11 belongs to tenant 2 and must
        # not filter out user 1's row entirely (LEFT JOIN preserved).
        self.assertEqual(self.run_tenant(sql, 1), {(1, 10), (2, None)})
        # User 3's only event (12) belongs to tenant 1, so the row survives
        # with NULLs instead of being dropped.
        self.assertEqual(self.run_tenant(sql, 2), {(3, None)})

    def test_right_join_results(self):
        sql = "SELECT u.id, e.id FROM users u RIGHT JOIN events e ON u.id = e.uid"
        self.assertEqual(self.run_tenant(sql, 1), {(1, 10), (None, 12), (None, 13)})
        self.assertEqual(self.run_tenant(sql, 2), {(None, 11)})

    def test_correlated_subquery_results(self):
        sql = (
            "SELECT u.id FROM users u "
            "WHERE EXISTS (SELECT 1 FROM events e WHERE e.uid = u.id)"
        )
        self.assertEqual(self.run_tenant(sql, 1), {(1,)})
        self.assertEqual(self.run_tenant(sql, 2), set())

    def test_cte_results(self):
        sql = (
            "WITH e AS (SELECT * FROM events) "
            "SELECT u.id, e.id FROM users u JOIN e ON u.id = e.uid"
        )
        self.assertEqual(self.run_tenant(sql, 1), {(1, 10)})
        self.assertEqual(self.run_tenant(sql, 2), set())


@unittest.skipIf(duckdb is None, "duckdb is not installed")
class TestTenantDuckDB(unittest.TestCase):
    def setUp(self):
        self.con = duckdb.connect()
        self.con.execute(
            "CREATE TABLE users AS SELECT * FROM "
            "VALUES (1, 1), (2, 1), (3, 2) AS t(id, tenant_id)"
        )
        self.con.execute(
            "CREATE TABLE events AS SELECT * FROM "
            "VALUES (10, 1, 1), (11, 1, 2), (12, 3, 1), (13, 99, 1) AS t(id, uid, tenant_id)"
        )

    def tearDown(self):
        self.con.close()

    def run_tenant(self, sql, tenant):
        transformed = add_tenant_filters(
            parse_one(sql), TABLES, exp.Placeholder(this="tenant")
        ).sql("duckdb")
        return set(self.con.execute(transformed, {"tenant": tenant}).fetchall())

    def test_left_join_results(self):
        sql = "SELECT u.id, e.id FROM users u LEFT JOIN events e ON u.id = e.uid"
        self.assertEqual(self.run_tenant(sql, 1), {(1, 10), (2, None)})
        self.assertEqual(self.run_tenant(sql, 2), {(3, None)})

    def test_right_join_results(self):
        sql = "SELECT u.id, e.id FROM users u RIGHT JOIN events e ON u.id = e.uid"
        self.assertEqual(self.run_tenant(sql, 1), {(1, 10), (None, 12), (None, 13)})
        self.assertEqual(self.run_tenant(sql, 2), {(None, 11)})

    def test_union_results(self):
        sql = "SELECT id FROM users UNION ALL SELECT uid FROM events"
        self.assertEqual(self.run_tenant(sql, 1), {(1,), (2,), (3,), (99,)})
        self.assertEqual(self.run_tenant(sql, 2), {(1,), (3,)})

    def test_cte_and_correlated_subquery_results(self):
        sql = (
            "WITH u2 AS (SELECT * FROM users) "
            "SELECT id FROM u2 WHERE EXISTS "
            "(SELECT 1 FROM events e WHERE e.uid = u2.id)"
        )
        self.assertEqual(self.run_tenant(sql, 1), {(1,)})
        self.assertEqual(self.run_tenant(sql, 2), set())
