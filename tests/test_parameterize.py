import sqlite3
import unittest
from decimal import Decimal

import sqlglot
from sqlglot import exp, parameterize
from sqlglot.parameterize import Parameterized


class TestParameterize(unittest.TestCase):
    maxDiff = None

    def test_basic_named(self):
        result = parameterize(
            "SELECT * FROM t WHERE a = 1 AND b = 'x' AND c = '1' AND d = -2.50",
            dialect="duckdb",
        )
        self.assertIsInstance(result, Parameterized)
        self.assertEqual(
            result.sql(), "SELECT * FROM t WHERE a = $p1 AND b = $p2 AND c = $p3 AND d = $p4"
        )
        self.assertEqual(result.values, [1, "x", "1", Decimal("-2.50")])
        self.assertEqual(
            [p.type.sql() for p in result.parameters],
            ["INT", "VARCHAR", "VARCHAR", "DECIMAL(3, 2)"],
        )
        self.assertEqual([p.name for p in result.parameters], ["p1", "p2", "p3", "p4"])

    def test_lossless_values(self):
        result = parameterize(
            "SELECT -5, 1.50, 123456789012345678901234567890, 0.1", dialect="sqlite"
        )
        values = result.values
        self.assertEqual(values[0], -5)
        self.assertIsInstance(values[0], int)
        self.assertEqual(values[1], Decimal("1.50"))
        self.assertIsInstance(values[1], Decimal)
        self.assertEqual(values[2], 123456789012345678901234567890)
        self.assertIsInstance(values[2], int)
        self.assertEqual(values[3], Decimal("0.1"))
        self.assertNotIsInstance(values[3], float)
        self.assertEqual(
            [p.type.sql() for p in result.parameters],
            ["INT", "DECIMAL(3, 2)", "DECIMAL(30, 0)", "DECIMAL(1, 1)"],
        )

    def test_same_text_different_types_not_merged(self):
        result = parameterize("SELECT '1', 1, '1', 1.0, TRUE, NULL", dialect="sqlite")
        self.assertEqual(result.values, ["1", 1, Decimal("1.0"), True, None])
        self.assertEqual(
            result.sql(), "SELECT :p1, :p2, :p1, :p3, :p4, :p5"
        )

    def test_dedupe_can_be_disabled(self):
        result = parameterize("SELECT 'a', 'a'", dialect="sqlite", dedupe=False)
        self.assertEqual(result.sql(), "SELECT :p1, :p2")
        self.assertEqual(result.values, ["a", "a"])

    def test_excluded_positions_keep_literals(self):
        result = parameterize(
            "SELECT CAST(x AS DECIMAL(10, 2)), CAST(y AS VARCHAR(10)) FROM t "
            "GROUP BY 1, 2 ORDER BY 1",
            dialect="postgres",
        )
        self.assertEqual(
            result.sql(),
            "SELECT CAST(x AS DECIMAL(10, 2)), CAST(y AS VARCHAR(10)) FROM t "
            "GROUP BY 1, 2 ORDER BY 1",
        )
        self.assertEqual(result.parameters, [])

    def test_interval_is_not_parameterized(self):
        result = parameterize("SELECT INTERVAL 3 DAY, INTERVAL '1' YEAR", dialect="duckdb")
        self.assertEqual(result.sql(), "SELECT INTERVAL '3' DAY, INTERVAL '1' YEAR")
        self.assertEqual(result.parameters, [])

    def test_limit_parameterized_unless_top_dialect(self):
        result = parameterize("SELECT a FROM t ORDER BY a LIMIT 5 OFFSET 2", dialect="sqlite")
        self.assertEqual(result.sql(), "SELECT a FROM t ORDER BY a LIMIT :p1 OFFSET :p2")
        self.assertEqual(result.values, [5, 2])

        result = parameterize("SELECT TOP 5 a FROM t WHERE b = 1", dialect="tsql")
        self.assertEqual(result.sql(), "SELECT TOP 5 a FROM t WHERE b = :p1")
        self.assertEqual(result.values, [1])

    def test_date_cast_type_is_inferred(self):
        result = parameterize("SELECT DATE '2024-01-01'", dialect="duckdb")
        self.assertEqual(result.sql(), "SELECT CAST($p1 AS DATE)")
        self.assertEqual(result.values, ["2024-01-01"])
        self.assertEqual(result.parameters[0].type.sql(), "DATE")

    def test_binary_values(self):
        result = parameterize("SELECT x'0A0B'", dialect="sqlite")
        self.assertEqual(result.values, [b"\n\x0b"])
        self.assertEqual(result.parameters[0].type.sql(), "VARBINARY")

    def test_existing_placeholders_are_preserved(self):
        result = parameterize("SELECT :x, 1", style="named")
        self.assertEqual(result.sql(), "SELECT :x, :p1")
        self.assertEqual([p.name for p in result.parameters], ["p1"])

        result = parameterize("SELECT $1, 2", dialect="duckdb", style="numbered")
        self.assertEqual(result.sql(), "SELECT $1, $2")
        self.assertEqual([p.name for p in result.parameters], ["2"])

        result = parameterize("SELECT @existing, 1", dialect="bigquery", style="named")
        self.assertEqual(result.sql(), "SELECT @existing, @p1")

    def test_original_ast_is_not_modified(self):
        expression = sqlglot.parse_one("SELECT a, 1, 'x' FROM t WHERE b = -2.5")
        before = expression.sql()
        result = parameterize(expression, dialect="sqlite")
        self.assertEqual(expression.sql(), before)
        self.assertIsNone(next(expression.find_all(exp.Placeholder), None))
        self.assertEqual(result.sql(), "SELECT a, :p1, :p2 FROM t WHERE b = :p3")

    def test_multiple_statements_share_numbering(self):
        result = parameterize("SELECT 1; SELECT 'a', 2", dialect="sqlite", style="qmark")
        self.assertEqual(result.sqls(), ["SELECT ?", "SELECT ?, ?"])
        self.assertEqual(result.values, [1, "a", 2])
        self.assertEqual([p.statement for p in result.parameters], [0, 1, 1])

        result = parameterize("SELECT 1; SELECT 2", dialect="duckdb")
        self.assertEqual(result.sqls(), ["SELECT $p1", "SELECT $p2"])

    def test_nested_query_parameter_order(self):
        result = parameterize(
            "SELECT (SELECT MAX(b) FROM s WHERE s.y = 10) FROM t "
            "WHERE t.a = 'x' AND t.c IN (SELECT c FROM u WHERE u.d = 2.5)",
            dialect="sqlite",
            style="qmark",
        )
        self.assertEqual(
            result.sql(),
            "SELECT (SELECT MAX(b) FROM s WHERE s.y = ?) FROM t "
            "WHERE t.a = ? AND t.c IN (SELECT c FROM u WHERE u.d = ?)",
        )
        self.assertEqual(result.values, [10, "x", Decimal("2.5")])

    def test_generated_sql_reparses(self):
        sql = "SELECT a, 'x' AS b FROM t WHERE c = 1 AND d = -2.5 ORDER BY 1 LIMIT 3"
        for dialect in ("sqlite", "duckdb", "bigquery", "mysql", None):
            with self.subTest(dialect=dialect):
                result = parameterize(sql, dialect=dialect)
                reparsed = sqlglot.parse_one(result.sql(), read=dialect)
                placeholders = sum(
                    1
                    for _ in reparsed.find_all(exp.Placeholder, exp.Parameter)
                )
                self.assertEqual(placeholders, len(result.parameters))

    def test_invalid_style_raises(self):
        with self.assertRaises(ValueError):
            parameterize("SELECT 1", style="unknown")

    def test_sqlite_execution_matches_literal_query(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE t (a INTEGER, b TEXT, c REAL, d BLOB)")
        conn.executemany(
            "INSERT INTO t VALUES (?, ?, ?, ?)",
            [
                (1, "x", 0.5, b"\x01"),
                (2, "y", 1.5, b"\x02"),
                (3, "x", 2.5, b"\x03"),
            ],
        )

        sql = (
            "SELECT a, b, c, d FROM t "
            "WHERE a > -1 AND b = 'x' AND c <= 2.5 AND d <> x'01' "
            "ORDER BY 1 LIMIT 10 OFFSET 0"
        )
        result = parameterize(sql, dialect="sqlite", style="qmark")

        def adapt(value):
            return float(value) if isinstance(value, Decimal) else value

        expected = conn.execute(sql).fetchall()
        actual = conn.execute(result.sql(), [adapt(v) for v in result.values]).fetchall()
        self.assertEqual(actual, expected)
        self.assertEqual(actual, [(3, "x", 2.5, b"\x03")])

    def test_sqlite_execution_multiple_statements(self):
        conn = sqlite3.connect(":memory:")
        result = parameterize(
            "CREATE TABLE t (a INTEGER, b TEXT); "
            "INSERT INTO t VALUES (1, 'x'), (2, 'y'); "
            "SELECT b FROM t WHERE a = 2",
            dialect="sqlite",
            style="qmark",
        )
        self.assertEqual([p.statement for p in result.parameters], [1, 1, 1, 1, 2])
        for statement, sql in enumerate(result.sqls()):
            params = [p.value for p in result.parameters if p.statement == statement]
            cursor = conn.execute(sql, params)
        self.assertEqual(cursor.fetchall(), [("y",)])


if __name__ == "__main__":
    unittest.main()
