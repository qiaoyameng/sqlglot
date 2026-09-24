"""Value vs. control dependencies in column lineage.

Value dependencies (Node.downstream) answer "where does this output's data come
from?". Control dependencies (Node.control) answer "which columns decide whether
this row appears at all?". Control edges are typed by the clause they came from:
where / join / having / qualify / group / case.

Run: python examples/control_dependencies.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlglot.lineage import lineage  # noqa: E402

SCHEMA = {
    "orders": {"order_id": "int", "amount": "decimal", "status": "text", "region": "text"},
}

SQL = """
WITH big_orders AS (
    SELECT order_id, amount, status, region
    FROM orders
    WHERE amount > 100
)
SELECT
    region,
    SUM(amount) AS total,
    CASE WHEN SUM(amount) > 5000 THEN 'busy' ELSE 'quiet' END AS volume
FROM big_orders
GROUP BY region
HAVING SUM(amount) > 1000
"""


def show(node, depth=0, seen=None):
    seen = seen if seen is not None else set()
    if id(node) in seen:
        return
    seen.add(id(node))
    print("  " * depth + f"{node.name}    [{node.source.sql(dialect='snowflake')[:70]}]")
    for child in node.downstream:
        print("  " * (depth + 1) + "(value)")
        show(child, depth + 2, seen)
    for edge in node.control:
        print("  " * (depth + 1) + f"({edge.kind})")
        show(edge.node, depth + 2, seen)


for column in ("region", "total", "volume"):
    print(f"=== lineage of {column!r} ===")
    show(lineage(column, SQL, schema=SCHEMA, include_control=True))
    print()
