import unittest

from sqlglot import exp, parse, parse_one


class TestSourceSpans(unittest.TestCase):
    def assert_slice(self, sql, expression, expected, msg=None):
        span = expression.source_span
        self.assertIsNotNone(span, msg)
        self.assertEqual(sql[span.start : span.end], expected, msg)
        # the UTF-8 byte span must slice the encoded input to the same text
        self.assertEqual(
            sql.encode("utf-8")[span.byte_start : span.byte_end].decode("utf-8"),
            expected,
            msg,
        )
        self.assertEqual(expression.source_slice(sql), expected, msg)

    def test_function_call(self):
        sql = "SELECT MAX(price), CONCAT(a, b, 1) FROM t"
        ast = parse_one(sql, source_spans=True)

        functions = list(ast.find_all(exp.Func))
        self.assert_slice(sql, functions[0], "MAX(price)")
        self.assert_slice(sql, functions[1], "CONCAT(a, b, 1)")
        self.assert_slice(sql, ast, sql)

    def test_window_expression(self):
        sql = "SELECT ROW_NUMBER() OVER (PARTITION BY x ORDER BY y), SUM(v) OVER () FROM t"
        ast = parse_one(sql, source_spans=True)

        windows = list(ast.find_all(exp.Window))
        self.assert_slice(sql, windows[0], "ROW_NUMBER() OVER (PARTITION BY x ORDER BY y)")
        self.assert_slice(sql, windows[1], "SUM(v) OVER ()")
        self.assert_slice(sql, ast, sql)

    def test_cte(self):
        sql = "WITH cte AS (SELECT a FROM t1) SELECT * FROM cte JOIN t2 ON cte.a = t2.a"
        ast = parse_one(sql, source_spans=True)

        self.assert_slice(sql, ast.find(exp.CTE), "cte AS (SELECT a FROM t1)")
        self.assert_slice(sql, ast.find(exp.With), "WITH cte AS (SELECT a FROM t1)")
        self.assert_slice(sql, ast, sql)

    def test_commented_subquery(self):
        sql = "SELECT * FROM (SELECT a /* keep me */ FROM t -- trailing\n) AS s WHERE x = 1"
        ast = parse_one(sql, source_spans=True)

        subquery = ast.find(exp.Subquery)
        self.assert_slice(sql, subquery, "(SELECT a /* keep me */ FROM t -- trailing\n) AS s")
        # the trailing comment is attached to the closing parenthesis, which is
        # owned by the subquery, so the inner select's span ends before it
        self.assert_slice(sql, subquery.this, "SELECT a /* keep me */ FROM t")
        self.assert_slice(sql, ast, sql)

    def test_alias_and_parens(self):
        sql = "SELECT (a + 1) AS b FROM t WHERE x IN (1, 2)"
        ast = parse_one(sql, source_spans=True)

        self.assert_slice(sql, ast.find(exp.Alias), "(a + 1) AS b")
        self.assert_slice(sql, ast.find(exp.Paren), "(a + 1)")
        self.assert_slice(sql, ast.find(exp.In), "x IN (1, 2)")

    def test_case_expression(self):
        sql = "SELECT CASE WHEN x > 1 THEN 'a' ELSE 'b' END, CASE x WHEN 1 THEN 2 END FROM t"
        ast = parse_one(sql, source_spans=True)

        cases = list(ast.find_all(exp.Case))
        self.assert_slice(sql, cases[0], "CASE WHEN x > 1 THEN 'a' ELSE 'b' END")
        self.assert_slice(sql, cases[1], "CASE x WHEN 1 THEN 2 END")
        self.assert_slice(sql, ast, sql)

    def test_ddl_and_values(self):
        sql = "CREATE TABLE t (a INT, b TEXT)"
        ast = parse_one(sql, source_spans=True)
        self.assert_slice(sql, ast.find(exp.Schema), "t (a INT, b TEXT)")
        self.assert_slice(sql, ast, sql)

        sql = "INSERT INTO t VALUES (1, 2), (3, 4)"
        ast = parse_one(sql, source_spans=True)
        self.assert_slice(sql, ast.find(exp.Values), "(1, 2), (3, 4)")
        tuples = list(ast.find_all(exp.Tuple))
        self.assert_slice(sql, tuples[0], "(1, 2)")
        self.assert_slice(sql, tuples[1], "(3, 4)")
        self.assert_slice(sql, ast, sql)

    def test_multiple_statements(self):
        sql = "SELECT 1; SELECT 2 ;\nSELECT 3"
        statements = parse(sql, source_spans=True)

        self.assertEqual(len(statements), 3)
        self.assert_slice(sql, statements[0], "SELECT 1")
        self.assert_slice(sql, statements[1], "SELECT 2")
        self.assert_slice(sql, statements[2], "SELECT 3")

    def test_unicode_and_crlf(self):
        sql = "SELECT '你好' AS 名, '🎉🎉' /* 表情 */,\r\n  1 AS one\r\nFROM 表"
        ast = parse_one(sql, source_spans=True)

        aliases = list(ast.find_all(exp.Alias))
        self.assert_slice(sql, aliases[0], "'你好' AS 名")
        self.assert_slice(sql, aliases[1], "1 AS one")
        self.assert_slice(sql, ast, sql)

        literals = {sql[l.source_span.start : l.source_span.end] for l in ast.find_all(exp.Literal)}
        self.assertEqual(literals, {"'你好'", "'🎉🎉'", "1"})

    def test_byte_offsets_with_emoji(self):
        sql = "SELECT '🎉' /* x */ + '🎉🎉'"
        ast = parse_one(sql, source_spans=True)

        self.assert_slice(sql, ast.find(exp.Add), "'🎉' /* x */ + '🎉🎉'")
        self.assert_slice(sql, ast, sql)

        literals = list(ast.find_all(exp.Literal))
        self.assert_slice(sql, literals[0], "'🎉'")
        self.assert_slice(sql, literals[1], "'🎉🎉'")

        # '🎉' is 1 character but 4 UTF-8 bytes, so byte offsets diverge
        self.assertEqual(literals[0].source_span.end - literals[0].source_span.start, 3)
        self.assertEqual(literals[0].source_span.byte_end - literals[0].source_span.byte_start, 6)

    def test_disabled_by_default(self):
        sql = "SELECT MAX(price) AS m FROM t"
        ast = parse_one(sql)

        for node in ast.walk():
            self.assertIsNone(node.source_span)
            self.assertIsNone(node.source_slice(sql))
            meta = node.meta
            self.assertNotIn("span", meta)
            self.assertNotIn("byte_span", meta)

    def test_source_text_not_retained(self):
        # spans are stored as integer offsets, so no node retains a reference
        # to the source text, whether or not source_spans is enabled
        sql = "SELECT MAX(price) AS m FROM t"
        for flag in (False, True):
            ast = parse_one(sql, source_spans=flag)
            for node in ast.walk():
                for key, value in (node._meta or {}).items():
                    self.assertNotIsInstance(value, str, (flag, key))
                    if flag and key in ("span", "byte_span"):
                        self.assertTrue(all(isinstance(v, int) for v in value))

    def test_positions_still_compatible(self):
        sql = "SELECT a, b FROM test_schema.test_table_a"
        ast = parse_one(sql, source_spans=True)

        table = ast.args["from_"].this
        self.assertEqual(
            {k: table.this.meta[k] for k in ("line", "col", "start", "end")},
            {"line": 1, "col": 41, "start": 29, "end": 40},
        )
        self.assertEqual(sql[table.this.meta["start"] : table.this.meta["end"] + 1], "test_table_a")
        self.assert_slice(sql, table, "test_schema.test_table_a")

    def test_copy_preserves_span(self):
        sql = "SELECT MAX(x) FROM t"
        ast = parse_one(sql, source_spans=True)
        copied = ast.copy()

        self.assertEqual(copied.find(exp.AggFunc).source_span, ast.find(exp.AggFunc).source_span)
        self.assert_slice(sql, copied, sql)

    def test_transform_drops_stale_spans(self):
        sql = "SELECT a, b FROM t"
        ast = parse_one(sql, source_spans=True)

        transformed = ast.transform(
            lambda node: (
                exp.column("z") if isinstance(node, exp.Column) and node.name == "a" else node
            )
        )
        self.assertEqual(transformed.sql(), "SELECT z, b FROM t")

        # the replacement node must not inherit a stale range
        replacement = transformed.expressions[0]
        self.assertIsNone(replacement.source_span)

        # ancestors of the replaced node no longer have a (now stale) span
        self.assertIsNone(transformed.source_span)

        # untouched subtrees keep their spans
        self.assert_slice(sql, transformed.expressions[1], "b")

    def test_replace_drops_ancestor_spans(self):
        sql = "SELECT a, b FROM t"
        ast = parse_one(sql, source_spans=True)

        column = ast.find(exp.Column)
        column.replace(exp.column("z"))

        self.assertIsNone(ast.source_span)
        self.assertIsNone(ast.expressions[0].source_span)
        # the sibling projection is untouched
        self.assert_slice(sql, ast.expressions[1], "b")

    def test_programmatic_nodes_have_no_span(self):
        ast = exp.select("x").from_("t")
        for node in ast.walk():
            self.assertIsNone(node.source_span)
