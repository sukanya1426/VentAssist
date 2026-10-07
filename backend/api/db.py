"""PostgreSQL layer — clinician accounts, the patient roster and its history.

Five tables, defined in ``schema.sql`` and created on startup:

  * ``clinician``       one row per clinician who signed up: username, display
                        name and the PBKDF2 hash of their password (never the
                        password itself). ``username`` is UNIQUE, so "this name is
                        taken" is a database guarantee rather than a race between
                        two simultaneous signups.
  * ``action``          the 125-element (ΔPEEP × ΔTV × ΔFiO₂) action space,
                        seeded from ``backend.mdp.action_space`` so the table can
                        never drift from the model's own encoding.
  * ``patient``         one row per bed on the roster. The id is the primary key,
                        so re-uploading the same id updates in place rather than
                        duplicating. The 12 clinical and 6 waveform fields are
                        flat columns with range checks.
  * ``recommendation``  append-only. Every ``POST /api/recommend`` writes one row
                        holding the state and settings it was asked with *and* the
                        action it chose, so changing a setting and asking again
                        files a second record under the same patient. It stores
                        ``action_idx`` only — ΔPEEP/ΔTV/ΔFiO₂ are read back by
                        joining ``action``, which is why a stored recommendation
                        cannot contradict the action space.
  * ``safety_flag``     one row per flag raised, child of ``recommendation``.

Deleting a patient is a single delete: ``recommendation`` cascades from
``patient``, and ``safety_flag`` cascades from ``recommendation``. Nothing is
retained afterwards. Deleting a *clinician* nulls the history's ``user_id``
instead of cascading — the clinical record has to outlive the account.

The pool is created lazily on first use and shared for the process lifetime;
connection failures surface as HTTP 503 at the route rather than killing startup,
so the recommendation endpoint keeps working when the database is unreachable.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Optional

import asyncpg

from backend.pipeline import config  # noqa: F401  — imports load backend/.env

log = logging.getLogger(__name__)

# A libpq URI: postgresql://user:password@host:port/dbname
DATABASE_URL = os.getenv("DATABASE_URL", "")

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

_pool: Optional[asyncpg.Pool] = None
# The loop the pool's sockets are bound to. A pool cannot be used from a
# different event loop: asyncpg schedules its timeouts on the loop that created
# it, so a stale pool fails with "Event loop is closed" rather than reconnecting.
# Under uvicorn there is one loop for the process and this never changes, but
# anything that runs requests across loops (Starlette's TestClient creates one
# per call unless used as a context manager) would otherwise wedge permanently.
_pool_loop: Optional[asyncio.AbstractEventLoop] = None


class DatabaseUnavailable(RuntimeError):
    """Raised when PostgreSQL is not configured or cannot be reached."""


# The driver errors a route should translate into a 503 rather than a 500: the API
# itself is fine, its store is not. Kept as a tuple so routes can name one thing.
DB_ERRORS = (DatabaseUnavailable, asyncpg.PostgresError, OSError)


def is_configured() -> bool:
    return bool(DATABASE_URL)


async def get_pool() -> asyncpg.Pool:
    global _pool, _pool_loop
    if not DATABASE_URL:
        raise DatabaseUnavailable(
            "DATABASE_URL is not set — add it to backend/.env (see .env.example)."
        )
    loop = asyncio.get_running_loop()
    if _pool is not None and _pool_loop is not loop:
        # Belongs to a different (usually finished) loop — drop it and reconnect
        # on this one rather than raising "Event loop is closed" forever.
        stale, _pool, _pool_loop = _pool, None, None
        try:
            # terminate(), not close(): close() awaits on the old loop, which is
            # exactly the loop that is no longer running.
            stale.terminate()
        except Exception:
            pass
    if _pool is None:
        try:
            # Fail fast: a hung roster request is worse than a clear error.
            _pool = await asyncpg.create_pool(
                DATABASE_URL, min_size=1, max_size=10,
                command_timeout=8, timeout=8,
                server_settings={"application_name": "VentAssist"},
            )
        except Exception as e:
            raise DatabaseUnavailable(f"could not connect to PostgreSQL: {e}") from e
        _pool_loop = loop
    return _pool


# --------------------------------------------------------------------------- #
# Query helpers — thin wrappers so routes read as SQL, not as pool plumbing
# --------------------------------------------------------------------------- #
async def fetch(sql: str, *args: Any) -> list[asyncpg.Record]:
    pool = await get_pool()
    async with pool.acquire() as con:
        return await con.fetch(sql, *args)


async def fetchrow(sql: str, *args: Any) -> Optional[asyncpg.Record]:
    pool = await get_pool()
    async with pool.acquire() as con:
        return await con.fetchrow(sql, *args)


async def fetchval(sql: str, *args: Any) -> Any:
    pool = await get_pool()
    async with pool.acquire() as con:
        return await con.fetchval(sql, *args)


async def execute(sql: str, *args: Any) -> str:
    pool = await get_pool()
    async with pool.acquire() as con:
        return await con.execute(sql, *args)


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #
async def ping() -> bool:
    """True when the database answers. Never raises — callers use it as a probe."""
    if not is_configured():
        return False
    try:
        return await fetchval("SELECT 1") == 1
    except Exception as e:  # network, auth, DNS — all mean "no database right now"
        log.warning("PostgreSQL ping failed: %s", e)
        return False


async def ensure_schema() -> None:
    """Apply ``schema.sql`` and seed the action space. Idempotent.

    Runs on every startup rather than through a migration tool: the DDL is written
    to be re-runnable, and a single statement per object keeps "is the schema
    current?" out of the deployment checklist for a prototype.
    """
    pool = await get_pool()
    async with pool.acquire() as con:
        await con.execute(SCHEMA_PATH.read_text())
    await seed_actions()


async def seed_actions() -> int:
    """Fill ``action`` from the model's own encoding. Returns rows inserted.

    The action space is reference data that MUST match
    ``backend.mdp.action_space``: ``recommendation.action_idx`` is a foreign key
    into it and the reported deltas are derived from it, so a mismatch would
    silently change what a stored recommendation means. Deriving the rows from the
    encoder instead of hard-coding them in SQL is what prevents that.
    """
    from backend.mdp import action_space

    rows = [(idx, *action_space.decode_action(idx))
            for idx in range(action_space.N_ACTIONS)]
    pool = await get_pool()
    async with pool.acquire() as con:
        before = await con.fetchval("SELECT COUNT(*) FROM action")
        await con.executemany(
            """
            INSERT INTO action (action_idx, delta_peep, delta_tv, delta_fio2)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (action_idx) DO UPDATE
                SET delta_peep = EXCLUDED.delta_peep,
                    delta_tv   = EXCLUDED.delta_tv,
                    delta_fio2 = EXCLUDED.delta_fio2
            """,
            rows,
        )
        after = await con.fetchval("SELECT COUNT(*) FROM action")
    if after != before:
        log.info("Seeded action space: %d → %d rows", before, after)
    return after - before


async def close() -> None:
    global _pool, _pool_loop
    if _pool is not None:
        await _pool.close()
        _pool = None
        _pool_loop = None
