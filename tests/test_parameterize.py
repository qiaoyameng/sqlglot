import sqlite3
import unittest
from decimal import Decimal

from sqlglot import exp, parse, parse_one, parameterize
from sqlglot.parameterize import NAMED, NUMBERED, QMARK


class TestParameterize(unittest.TestCase):
    maxDiff = None

    def test_named_style(self):
        result = parameterize(
            parse_one("SELECT * FROM t WHERE a = 1 AND b = 'x' AND c = 1", read="duckdb"),
            dialect="duckdb",
        )
        self.assertEqual(
            result.sql(), ["SELECT * FROM t WHERE a = $p1 AND b = $p2 AND c = $p1"]
        )
        self.assertEqual([p.name for p in result.parameters], ["p1", "p2"])
        self.assertEqual(result.values, [1, "x"])
        self.assertEqual([p.type.sql() for p in result.parameters], ["INT", "VARCHAR"])

    def test_qmark_style(self):
        result = parameterize(parse_one("SELECT 1, 'a'"), style=QMARK)
        self.assertEqual(result.sql(), ["SELECT ?, ?"])
        self.assertEqual(result.values, [1, "a"])

    def test_numbered_style(self):
        result = parameterize(
            parse_one("SELECT 1, 'a', 1", read="duckdb"), dialect="duckdb", style=NUMBERED
        )
        self.assertEqual(result.sql(), ["SELECT $1, $2, $1"])
        self.assertEqual([p.name for p in result.parameters], ["1", "2"])

    def test_same_text_different_types_not_merged(self):
        result = parameterize(parse_one("SELECT 1, '1', 1.0, 1"))
        self.assertEqual(len(result.parameters), 3)
        self.assertEqual(result.values, [1, "1", Decimal("1.0")])
        self.assertEqual(
            [p.type.sql() for p in result.parameters], ["INT", "VARCHAR", "DECIMAL(2, 1)"]
        )

    def test_existing_parameters_preserved(self):
        result = parameterize(
            parse_one("SELECT * FROM t WHERE a = :p1 AND b = ? AND c = @p3 AND d = 2"),
            style=NAMED,
        )
        self.assertEqual(
            result.sql(), ["SELECT * FROM t WHERE a = :p1 AND b = ? AND c = @p3 AND d = :p2"]
        )
        self.assertEqual([p.name for p in result.parameters], ["p2"])
        self.assertEqual(result.values, [2])

    def test_existing_numbered_parameters_not_reused(self):
        result = parameterize(
            parse_one("SELECT $1, 5", read="duckdb"), dialect="duckdb", style=NUMBERED
        )
        self.assertEqual(result.sql(), ["SELECT $1, $2"])
        self.assertEqual([p.name for p in result.parameters], ["2"])

    def test_non_parameterizable_positions(self):
        result = parameterize(
            parse_one(
                "SELECT CAST(x AS VARCHAR(10)), CAST(y AS DECIMAL(18, 2)), INTERVAL '1' DAY "
                "FROM t ORDER BY 1 LIMIT 5 OFFSET 3"
            )
        )
        self.assertEqual(
            result.sql(),
            [
                "SELECT CAST(x AS VARCHAR(10)), CAST(y AS DECIMAL(18, 2)), INTERVAL '1' DAY "
                "FROM t ORDER BY 1 LIMIT 5 OFFSET 3"
            ],
        )
        self.assertEqual(result.parameters, [])

    def test_parameterize_limit_default_dialects(self):
        sql = "SELECT * FROM t LIMIT 5 OFFSET 3"
        expected = {
            "duckdb": "SELECT * FROM t LIMIT ? OFFSET ?",
            "mysql": "SELECT * FROM t LIMIT ? OFFSET ?",
            "postgres": "SELECT * FROM t LIMIT %s OFFSET %s",
            "sqlite": "SELECT * FROM t LIMIT ? OFFSET ?",
        }
        for dialect, expected_sql in expected.items():
            result = parameterize(parse_one(sql, read=dialect), dialect=dialect, style=QMARK)
            self.assertEqual(result.sql(), [expected_sql])
            self.assertEqual(result.values, [5, 3])

        result = parameterize(parse_one(sql), style=QMARK)
        self.assertEqual(result.sql(), ["SELECT * FROM t LIMIT 5 OFFSET 3"])

    def test_parameterize_limit_override(self):
        sql = "SELECT * FROM t LIMIT 5"
        result = parameterize(parse_one(sql), parameterize_limit=True, style=QMARK)
        self.assertEqual(result.sql(), ["SELECT * FROM t LIMIT ?"])

        result = parameterize(
            parse_one(sql, read="duckdb"), dialect="duckdb", parameterize_limit=False
        )
        self.assertEqual(result.sql(), ["SELECT * FROM t LIMIT 5"])

    def test_exact_values(self):
        result = parameterize(parse_one("SELECT -3.14, 123456789012345, 0.1, -7"))
        decimal_value, big_value, small_value, neg_value = result.values
        self.assertIsInstance(decimal_value, Decimal)
        self.assertEqual(decimal_value, Decimal("3.14"))
        self.assertIsInstance(big_value, int)
        self.assertEqual(big_value, 123456789012345)
        self.assertIsInstance(small_value, Decimal)
        self.assertNotIsInstance(small_value, float)
        self.assertEqual(neg_value, 7)
        self.assertEqual(
            [p.type.sql() for p in result.parameters],
            ["DECIMAL(3, 2)", "BIGINT", "DECIMAL(1, 1)", "INT"],
        )
        # negative numbers keep the sign in the SQL, the bound value stays exact
        self.assertEqual(result.sql(), ["SELECT -:p1, :p2, :p3, -:p4"])

    def test_temporal_cast_types(self):
        result = parameterize(
            parse_one("SELECT DATE '2024-01-01', TIMESTAMP '2024-01-01 10:00:00'", read="duckdb"),
            dialect="duckdb",
        )
        self.assertEqual(result.values, ["2024-01-01", "2024-01-01 10:00:00"])
        self.assertEqual([p.type.sql() for p in result.parameters], ["DATE", "TIMESTAMP"])

    def test_original_not_mutated(self):
        expression = parse_one("SELECT 1 AS x, 'a' AS y FROM t WHERE z = 2")
        before = repr(expression)
        result = parameterize(expression)
        self.assertEqual(repr(expression), before)
        self.assertIsNot(result.expression, expression)

    def test_multiple_statements_order(self):
        result = parameterize(parse("SELECT 1; SELECT 'a', 1; SELECT 2.5"))
        self.assertEqual(result.sql(), ["SELECT :p1", "SELECT :p2, :p1", "SELECT :p3"])
        self.assertEqual(result.values, [1, "a", Decimal("2.5")])
        with self.assertRaises(ValueError):
            result.expression

    def test_nested_queries_order(self):
        result = parameterize(
            parse_one(
                "SELECT (SELECT MAX(c) FROM u WHERE u.d = 7) FROM "
                "(SELECT * FROM t WHERE x = 1) s WHERE y = 'a' AND z = 2"
            )
        )
        self.assertEqual(result.values, [7, 1, "a", 2])

    def test_generated_sql_reparses(self):
        sql = "SELECT a, 'x' AS b FROM t WHERE c = 1 AND d = 2.5 ORDER BY 1 LIMIT 3"
        for dialect in (None, "sqlite", "duckdb", "mysql"):
            for style in (NAMED, QMARK):
                result = parameterize(
                    parse_one(sql, read=dialect), dialect=dialect, style=style
                )
                for generated in result.sql():
                    reparsed = parse_one(generated, read=dialect)
                    self.assertEqual(reparsed.sql(dialect), generated)

    def test_generated_sql_reparses_numbered(self):
        result = parameterize(
            parse_one("SELECT 1, 'a'", read="duckdb"), dialect="duckdb", style=NUMBERED
        )
        generated = result.sql()[0]
        self.assertEqual(generated, "SELECT $1, $2")
        self.assertEqual(parse_one(generated, read="duckdb").sql("duckdb"), generated)

    def test_invalid_style(self):
        with self.assertRaises(ValueError):
            parameterize(parse_one("SELECT 1"), style="unknown")

    def test_invalid_input(self):
        with self.assertRaises(TypeError):
            parameterize("SELECT 1")

    def test_sqlite_execution_round_trip(self):
        connection = sqlite3.connect(":memory:")
        connection.execute("CREATE TABLE t (a INTEGER, b TEXT, c REAL)")

        insert = parameterize(
            parse_one("INSERT INTO t VALUES (1, 'x', 2.5)", read="sqlite"),
            dialect="sqlite",
            style=QMARK,
        )
        connection.execute(
            insert.sql()[0],
            [float(v) if isinstance(v, Decimal) else v for v in insert.values],
        )
        connection.execute("INSERT INTO t VALUES (2, 'y', 3.5)")

        query = parse_one("SELECT a, b, c FROM t WHERE a = 1 AND b = 'x'", read="sqlite")
        expected = connection.execute(query.sql("sqlite")).fetchall()

        parameterized = parameterize(query, dialect="sqlite", style=QMARK)
        actual = connection.execute(parameterized.sql()[0], parameterized.values).fetchall()
        self.assertEqual(actual, expected)
        self.assertEqual(actual, [(1, "x", 2.5)])

        named = parameterize(query, dialect="sqlite", style=NAMED)
        bound = connection.execute(
            named.sql()[0], {p.name: p.value for p in named.parameters}
        ).fetchall()
        self.assertEqual(bound, expected)
        connection.close()


if __name__ == "__main__":
    unittest.main()
