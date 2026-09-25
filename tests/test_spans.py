import unittest

from sqlglot import exp, parse, parse_one


class TestSourceSpans(unittest.TestCase):
    def _assert_span(self, sql, node, expected):
        span = node.source_span
        self.assertIsNotNone(span, f"{type(node).__name__} has no span")
        self.assertEqual(span.slice(sql), expected)
        # UTF-8 byte offsets must slice the exact same source text
        self.assertEqual(span.slice_bytes(sql.encode("utf-8")).decode("utf-8"), expected)
        return span

    def test_spans_disabled_by_default(self):
        sql = "SELECT a + 1 AS b FROM t WHERE a > 0"
        tree = parse_one(sql)
        self.assertIsNone(tree.source_span)
        for node in tree.walk():
            if isinstance(node, exp.Expression) and node._meta is not None:
                self.assertNotIn("span", node._meta)
                self.assertNotIn("_span_end", node._meta)
                # The source text itself must never be retained on the AST
                for value in node._meta.values():
                    self.assertNotIsInstance(value, str)

    def test_spans_are_integer_offsets_only(self):
        sql = "SELECT '汉😀' FROM t"
        tree = parse_one(sql, source_spans=True)
        for node in tree.walk():
            if isinstance(node, exp.Expression) and node.source_span is not None:
                self.assertTrue(all(isinstance(i, int) for i in node.source_span))
                self.assertNotIn("_span_end", node.meta)

    def test_function_call(self):
        sql = "SELECT CONCAT(first_name, ' ', last_name) FROM users"
        tree = parse_one(sql, source_spans=True)
        self._assert_span(sql, tree.find(exp.Concat), "CONCAT(first_name, ' ', last_name)")
        self._assert_span(sql, tree.find(exp.Column), "first_name")

    def test_window_expression(self):
        sql = (
            "SELECT SUM(x) OVER (PARTITION BY y ORDER BY z "
            "ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS s FROM t"
        )
        tree = parse_one(sql, source_spans=True)
        window = tree.find(exp.Window)
        self._assert_span(
            sql,
            window,
            "SUM(x) OVER (PARTITION BY y ORDER BY z "
            "ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)",
        )
        self._assert_span(
            sql, window.find(exp.WindowSpec), "ROWS BETWEEN 1 PRECEDING AND CURRENT ROW"
        )
        self._assert_span(sql, window.find(exp.Sum), "SUM(x)")
        self._assert_span(sql, window.find(exp.Order), "ORDER BY z")

    def test_cte(self):
        sql = "WITH cte AS (SELECT x FROM t1), cte2 AS (SELECT y FROM t2) SELECT * FROM cte"
        tree = parse_one(sql, source_spans=True)
        self._assert_span(
            sql,
            tree.find(exp.With),
            "WITH cte AS (SELECT x FROM t1), cte2 AS (SELECT y FROM t2)",
        )
        ctes = list(tree.find_all(exp.CTE))
        self._assert_span(sql, ctes[0], "cte AS (SELECT x FROM t1)")
        self._assert_span(sql, ctes[1], "cte2 AS (SELECT y FROM t2)")
        self._assert_span(sql, ctes[0].find(exp.Select), "SELECT x FROM t1")

    def test_commented_subquery(self):
        sql = "SELECT * FROM ( -- subquery\n SELECT a /* keep a */, b FROM t) AS s"
        tree = parse_one(sql, source_spans=True)
        subquery = tree.find(exp.Subquery)
        self._assert_span(
            sql, subquery, "( -- subquery\n SELECT a /* keep a */, b FROM t) AS s"
        )
        inner_select = subquery.this
        self._assert_span(sql, inner_select, "SELECT a /* keep a */, b FROM t")

    def test_leading_and_trailing_comments_belong_to_statement(self):
        sql = "/* header */ SELECT a -- tail\n"
        tree = parse_one(sql, source_spans=True)
        self._assert_span(sql, tree, "/* header */ SELECT a -- tail")

    def test_multiple_statements_and_separators(self):
        sql = "SELECT 1;\nSELECT 2;"
        statements = parse(sql, source_spans=True)
        self._assert_span(sql, statements[0], "SELECT 1")
        self._assert_span(sql, statements[1], "SELECT 2")
        # Block is built outside the parser and carries no span
        block = parse_one(sql, source_spans=True)
        self.assertIsInstance(block, exp.Block)
        self.assertIsNone(block.source_span)

    def test_alias_ownership(self):
        sql = "SELECT b + 1 AS c, d AS e FROM t"
        tree = parse_one(sql, source_spans=True)
        aliases = list(tree.find_all(exp.Alias))
        self._assert_span(sql, aliases[0], "b + 1 AS c")
        self._assert_span(sql, aliases[1], "d AS e")
        self._assert_span(sql, aliases[0].find(exp.Add), "b + 1")

    def test_parentheses_ownership(self):
        sql = "SELECT (a + b) * c, x IN (1, 2, 3) FROM t"
        tree = parse_one(sql, source_spans=True)
        self._assert_span(sql, tree.find(exp.Paren), "(a + b)")
        self._assert_span(sql, tree.find(exp.In), "x IN (1, 2, 3)")

    def test_nested_spans_are_nested_and_half_open(self):
        sql = "SELECT a FROM t"
        tree = parse_one(sql, source_spans=True)
        for node in tree.dfs():
            span = node.source_span if isinstance(node, exp.Expression) else None
            if span is not None:
                self.assertLessEqual(span.start, span.end)
                self.assertLessEqual(span.byte_start, span.byte_end)
                self.assertEqual(
                    span.byte_end - span.byte_start,
                    len(sql[span.start : span.end].encode("utf-8")),
                )
        for parent in tree.dfs():
            if not isinstance(parent, exp.Expression):
                continue
            for child in parent.iter_expressions():
                if child.source_span is not None and parent.source_span is not None:
                    self.assertGreaterEqual(child.source_span.start, parent.source_span.start)
                    self.assertLessEqual(child.source_span.end, parent.source_span.end)

    def test_utf8_crlf_emoji(self):
        sql = "SELECT '你好,世界' AS 问候, '😀' AS emoji\r\nFROM t\r\nWHERE a = '汉'"
        encoded = sql.encode("utf-8")
        tree = parse_one(sql, source_spans=True)
        root_span = tree.source_span
        self.assertEqual(root_span.slice(sql), sql)
        self.assertEqual(root_span.slice_bytes(encoded).decode("utf-8"), sql)
        self.assertEqual(root_span.byte_start, 0)
        self.assertEqual(root_span.byte_end, len(encoded))

        literal = tree.find(exp.Literal)
        span = literal.source_span
        self.assertEqual(span.slice(sql), "'你好,世界'")
        self.assertEqual(span.byte_end - span.byte_start, len("'你好,世界'".encode("utf-8")))
        self.assertGreater(span.byte_end - span.byte_start, span.end - span.start)
        self.assertEqual(
            encoded[span.byte_start : span.byte_end].decode("utf-8"), "'你好,世界'"
        )

        where = tree.find(exp.Where)
        self._assert_span(sql, where, "WHERE a = '汉'")

    def test_copy_preserves_span(self):
        sql = "SELECT a + 1 FROM t"
        tree = parse_one(sql, source_spans=True)
        column = tree.find(exp.Column)
        self.assertEqual(column.copy().source_span, column.source_span)

    def test_transform_does_not_fabricate_spans(self):
        sql = "SELECT a + 1 FROM t"
        tree = parse_one(sql, source_spans=True)
        transformed = tree.transform(
            lambda node: exp.Literal.number(2) if isinstance(node, exp.Literal) else node
        )
        for node in transformed.find_all(exp.Literal):
            self.assertIsNone(node.source_span)
        # Unchanged nodes retain their original span
        self.assertEqual(transformed.find(exp.Column).source_span.slice(sql), "a")

    def test_replace_and_update_positions_do_not_transfer_span(self):
        sql = "SELECT a + 1 FROM t"
        tree = parse_one(sql, source_spans=True)
        literal = tree.find(exp.Literal)
        replacement = exp.Literal.number(9)
        self.assertIsNone(replacement.source_span)
        literal.replace(replacement)
        self.assertIsNone(replacement.source_span)

        node = exp.column("z").update_positions(literal)
        self.assertIsNone(node.source_span)

    def test_existing_positions_unchanged(self):
        sql = "SELECT a, 1 FROM t"
        disabled = parse_one(sql)
        enabled = parse_one(sql, source_spans=True)
        for a, b in zip(disabled.walk(), enabled.walk()):
            self.assertEqual(type(a), type(b))
            meta_a = a._meta or {}
            meta_b = b._meta or {}
            for key in ("line", "col", "start", "end"):
                self.assertEqual(meta_a.get(key), meta_b.get(key))

    def test_parse_into_with_spans(self):
        sql = "(SELECT a FROM t)"
        subquery = parse_one(sql, source_spans=True)
        self._assert_span(sql, subquery, "(SELECT a FROM t)")


if __name__ == "__main__":
    unittest.main()
