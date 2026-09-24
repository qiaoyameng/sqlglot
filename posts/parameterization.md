# Parameterization

`sqlglot.parameterize` turns literal values in one or more parsed statements into
placeholders and returns the extracted parameters in execution (SQL text) order, so a
query agent can bind them instead of interpolating values into SQL text.

```python
import sqlglot

result = sqlglot.parameterize(
    sqlglot.parse_one("SELECT * FROM t WHERE a = 1 AND b = 'x'", read="duckdb"),
    dialect="duckdb",
    style="named",
)

result.sql()
# ["SELECT * FROM t WHERE a = $p1 AND b = $p2"]

[(p.name, p.value, p.type.sql()) for p in result.parameters]
# [('p1', 1, 'INT'), ('p2', 'x', 'VARCHAR')]
```

## Placeholder styles

- `"named"` (default): dialect-specific named placeholders, e.g. `:p1`, `$p1`, `%(p1)s`.
- `"numbered"`: numbered parameters, e.g. `$1`, `@1`.
- `"qmark"`: positional `?` placeholders.

## Supported scope

Only string and numeric literals are parameterized. The following positions keep their
literals because they are not bind-compatible in general:

- data type parameters, e.g. the `10` in `CAST(x AS VARCHAR(10))`
- `INTERVAL` values and units, e.g. `INTERVAL '1' DAY`
- `ORDER BY` ordinals, e.g. the `1` in `ORDER BY 1`
- `LIMIT` / `OFFSET` / `FETCH` counts, unless `parameterize_limit=True` is passed
  (it defaults to `True` only for dialects known to accept bound limits: DuckDB,
  MySQL, Postgres and SQLite)

Booleans, `NULL` and hex/bit strings are preserved as well. Existing placeholders and
parameters in the input are left untouched, and generated names never collide with them.

## Value fidelity

Literals with identical text *and* kind share a single parameter, so `1` and `'1'` are
never merged. Values are extracted exactly: integers stay `int`, decimals become
`Decimal`, and date/timestamp/binary-looking text stays a string -- nothing is routed
through a lossy `float`. Each parameter carries an inferred `exp.DataType`
(`INT`/`BIGINT`, `DECIMAL(p, s)`, `VARCHAR`, or `DATE`/`TIME`/`TIMESTAMP` when the
literal is the operand of a temporal cast).

The input expressions are never mutated; the result contains copies.
