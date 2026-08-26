"""Read-only guard for model-generated SQL.

The agent writes SQL and then runs it. That is the point of the project and also the
obvious way to get hurt, especially once the demo is publicly reachable. So every
statement the model produces goes through here first, and ClickHouse is *also* told
to run it read-only with a time limit — belt and braces, because a validator that is
the only line of defence eventually meets a query it did not anticipate.
"""

from __future__ import annotations

import re

MAX_ROWS = 200
MAX_EXECUTION_SECONDS = 10

_FORBIDDEN = (
    "insert", "alter", "drop", "create", "attach", "detach", "truncate", "rename",
    "optimize", "grant", "revoke", "kill", "system", "set", "use", "delete",
    "update", "exchange", "restore", "backup",
)
_COMMENT = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
_STRING = re.compile(r"'(?:[^'\\]|\\.)*'")


class UnsafeSQL(ValueError):
    """Raised when generated SQL is not a plain read."""


def _strip(sql: str) -> str:
    """Remove comments and string literals so keyword checks cannot be smuggled past."""
    return _STRING.sub("''", _COMMENT.sub(" ", sql))


def validate(sql: str) -> str:
    """Return the statement if it is a single read-only SELECT, else raise `UnsafeSQL`."""
    if not sql or not sql.strip():
        raise UnsafeSQL("empty statement")

    bare = _strip(sql).strip().rstrip(";").strip()
    if ";" in bare:
        raise UnsafeSQL("only a single statement is allowed")

    first = bare.split(None, 1)[0].lower() if bare.split() else ""
    if first not in {"select", "with"}:
        raise UnsafeSQL(f"statement must start with SELECT or WITH, got {first.upper() or '?'}")

    words = set(re.findall(r"[a-z_]+", bare.lower()))
    hits = sorted(words & set(_FORBIDDEN))
    if hits:
        raise UnsafeSQL(f"statement contains forbidden keyword(s): {', '.join(hits)}")

    return sql.strip().rstrip(";").strip()


READONLY_SETTINGS = {
    "readonly": 1,
    "max_execution_time": MAX_EXECUTION_SECONDS,
    "max_result_rows": MAX_ROWS,
    "result_overflow_mode": "break",
}
