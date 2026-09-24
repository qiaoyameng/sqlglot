import unittest

from sqlglot import parse_one, expressions as exp
from sqlglot.named_args import (
    NamedArgumentError,
    Parameter,
    SignatureRegistry,
    named_args_to_positional,
)


class TestNamedArgs(unittest.TestCase):
    maxDiff = None

    def transform(self, sql, signatures, dialect=None, read=None):
        registry = SignatureRegistry(dialect=dialect)
        for name, parameters in signatures.items():
            registry.register(name, parameters)
        return named_args_to_positional(parse_one(sql, read=read), registry)

    def test_reorders_named_arguments(self):
        result = self.transform(
            "SELECT my_func(c => 3, a => 1, b => 2)",
            {"my_func": ["a", "b", "c"]},
        )
        self.assertEqual(result.sql(), "SELECT MY_FUNC(1, 2, 3)")

    def test_trailing_default_is_dropped(self):
        result = self.transform(
            "SELECT my_func(a => 1)",
            {"my_func": ["a", Parameter("b", default="0")]},
        )
        self.assertEqual(result.sql(), "SELECT MY_FUNC(1)")

    def test_middle_default_is_filled(self):
        result = self.transform(
            "SELECT my_func(a => 1, c => 3)",
            {"my_func": ["a", Parameter("b", default="'x'"), "c"]},
        )
        self.assertEqual(result.sql(), "SELECT MY_FUNC(1, 'x', 3)")

    def test_mixed_positional_and_named(self):
        result = self.transform(
            "SELECT my_func(1, c => 3)",
            {"my_func": ["a", Parameter("b", default="'x'"), "c"]},
        )
        self.assertEqual(result.sql(), "SELECT MY_FUNC(1, 'x', 3)")

    def test_expression_default(self):
        result = self.transform(
            "SELECT my_func(a => 1, c => 3)",
            {"my_func": ["a", Parameter("b", default=exp.null()), "c"]},
        )
        self.assertEqual(result.sql(), "SELECT MY_FUNC(1, NULL, 3)")

    def test_unknown_argument(self):
        with self.assertRaises(NamedArgumentError):
            self.transform("SELECT my_func(nope => 1)", {"my_func": ["a"]})

    def test_duplicate_named_argument(self):
        with self.assertRaises(NamedArgumentError):
            self.transform("SELECT my_func(a => 1, a => 2)", {"my_func": ["a"]})

    def test_positional_after_named(self):
        with self.assertRaises(NamedArgumentError):
            self.transform("SELECT my_func(a => 1, 2)", {"my_func": ["a", "b"]})

    def test_positional_named_conflict(self):
        with self.assertRaises(NamedArgumentError):
            self.transform("SELECT my_func(1, a => 2)", {"my_func": ["a"]})

    def test_too_many_positional_arguments(self):
        with self.assertRaises(NamedArgumentError):
            self.transform("SELECT my_func(1, 2, c => 3)", {"my_func": ["a"]})

    def test_missing_required_argument(self):
        with self.assertRaises(NamedArgumentError):
            self.transform(
                "SELECT my_func(c => 3)",
                {"my_func": ["a", Parameter("b", default="0"), "c"]},
            )

    def test_same_name_across_schemas(self):
        result = self.transform(
            "SELECT s1.f(a => 1, c => 9), s2.f(a => 1, c => 9)",
            {
                "s1.f": ["a", Parameter("b", default="'s1'"), "c"],
                "s2.f": ["a", Parameter("b", default="'s2'"), "c"],
            },
        )
        self.assertEqual(result.sql(), "SELECT s1.f(1, 's1', 9), s2.f(1, 's2', 9)")

    def test_unregistered_qualified_call_untouched(self):
        result = self.transform(
            "SELECT other.f(a => 1)",
            {"s1.f": ["a"]},
        )
        self.assertEqual(result.sql(), "SELECT other.f(a => 1)")

    def test_three_part_qualified_name(self):
        result = self.transform(
            "SELECT cat.s.f(b => 2, a => 1)",
            {"cat.s.f": ["a", "b"]},
        )
        self.assertEqual(result.sql(), "SELECT cat.s.f(1, 2)")

    def test_dialect_case_insensitive_matching(self):
        result = self.transform(
            "SELECT My_Func(X => 1)",
            {"my_func": ["x"]},
            dialect="snowflake",
        )
        self.assertEqual(result.sql("snowflake"), "SELECT MY_FUNC(1)")

    def test_quoted_name_is_case_sensitive(self):
        # Quoted "my_func" must not match the unquoted registration my_func in
        # Snowflake (which normalizes to MY_FUNC), so the call is left untouched.
        result = self.transform(
            'SELECT "my_func"(x => 1)',
            {"my_func": ["x"]},
            dialect="snowflake",
            read="snowflake",
        )
        self.assertIsNotNone(result.find(exp.Kwarg))

        registry = SignatureRegistry(dialect="snowflake")
        registry.register('"my_func"', ["x"])
        result = named_args_to_positional(
            parse_one('SELECT "my_func"(x => 1)', read="snowflake"), registry
        )
        self.assertIsNone(result.find(exp.Kwarg))
        self.assertEqual(
            result.find(exp.Anonymous).expressions, [exp.Literal.number(1)]
        )

    def test_nested_calls(self):
        result = self.transform(
            "SELECT g(b => h(y => 2), a => 1)",
            {"g": ["a", "b"], "h": ["y"]},
        )
        self.assertEqual(result.sql(), "SELECT G(1, H(2))")

    def test_original_ast_unchanged(self):
        expression = parse_one("SELECT my_func(c => 3, a => 1)")
        before = expression.sql()
        result = named_args_to_positional(expression, {"my_func": ["a", Parameter("b", default="0"), "c"]})
        self.assertEqual(expression.sql(), before)
        self.assertIsNot(result, expression)
        self.assertEqual(result.sql(), "SELECT MY_FUNC(1, 0, 3)")

    def test_defaults_not_shared_between_calls(self):
        registry = SignatureRegistry()
        registry.register("f", ["a", Parameter("b", default="[1, 2]"), "c"])
        result = named_args_to_positional(
            parse_one("SELECT f(a => 1, c => 3), f(a => 4, c => 6)"), registry
        )
        defaults = [
            node
            for node in result.find_all(exp.Array)
        ]
        self.assertEqual(len(defaults), 2)
        self.assertIsNot(defaults[0], defaults[1])
        defaults[0].set("expressions", [])
        self.assertEqual(
            result.sql(), "SELECT F(1, ARRAY(), 3), F(4, ARRAY(1, 2), 6)"
        )

    def test_volatile_reorder_rejected(self):
        with self.assertRaises(NamedArgumentError):
            self.transform("SELECT f(b => 2, a => rand())", {"f": ["a", "b"]})

    def test_volatile_in_order_allowed(self):
        result = self.transform("SELECT f(a => rand(), b => 2)", {"f": ["a", "b"]})
        self.assertEqual(result.sql(), "SELECT F(RAND(), 2)")

    def test_volatile_default_interleaved_rejected(self):
        with self.assertRaises(NamedArgumentError):
            self.transform(
                "SELECT f(a => random(), c => 1)",
                {"f": ["a", Parameter("b", default="rand()"), "c"]},
            )

    def test_volatile_default_without_volatile_args_allowed(self):
        result = self.transform(
            "SELECT f(a => 1, c => 3)",
            {"f": ["a", Parameter("b", default="uuid()"), "c"]},
        )
        self.assertEqual(result.sql(), "SELECT F(1, UUID(), 3)")

    def test_sequence_nextval_reorder_rejected(self):
        with self.assertRaises(NamedArgumentError):
            self.transform(
                "SELECT f(b => 2, a => NEXT VALUE FOR my_seq)",
                {"f": ["a", "b"]},
                read="tsql",
            )

    def test_generates_target_dialect_sql(self):
        result = self.transform(
            "SELECT my_func(c => 3, a => 1)",
            {"my_func": ["a", Parameter("b", default="0"), "c"]},
        )
        self.assertEqual(result.sql("duckdb"), "SELECT MY_FUNC(1, 0, 3)")


if __name__ == "__main__":
    unittest.main()
