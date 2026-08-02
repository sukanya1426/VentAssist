"""MongoDB layer — the clinician accounts, the patient roster and its history.

Three collections. Two are keyed on the roster patient id (``patient-a``,
``upload-…``), the third on a generated user id:

  * ``patients``          one document per bed on the roster. ``_id`` is the
                          patient id, so an upload of the same id upserts rather
                          than duplicating.
  * ``recommendations``   append-only. Every ``POST /api/recommend`` writes one
                          document holding the state and settings it was asked
                          with *and* the answer it gave, so changing a setting and
                          asking again files a second record under the same
                          patient instead of overwriting the first.
  * ``users``             one document per clinician who signed up: username,
                          display name and the PBKDF2 hash of their password
                          (never the password itself). A unique index on
                          ``username`` is what makes "this name is taken" a
                          database guarantee rather than a race between two
                          simultaneous signups.

Deleting a patient is a plain delete of both: the ``patients`` document and every
``recommendations`` document carrying its id. Nothing is retained afterwards.

The client is created lazily on first use and shared for the process lifetime;
connection failures surface as HTTP 503 at the route rather than killing startup,
so the recommendation endpoint keeps working when Atlas is unreachable.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection, AsyncIOMotorDatabase

from backend.pipeline import config  # noqa: F401  — imports load backend/.env

log = logging.getLogger(__name__)

MONGODB_URI = os.getenv("MONGODB_URI", "")
MONGODB_DB = os.getenv("MONGODB_DB", "ventassist")

PATIENTS = "patients"
RECOMMENDATIONS = "recommendations"
USERS = "users"

_client: Optional[AsyncIOMotorClient] = None


class DatabaseUnavailable(RuntimeError):
    """Raised when Mongo is not configured or cannot be reached."""


def is_configured() -> bool:
    return bool(MONGODB_URI)


def get_client() -> AsyncIOMotorClient:
    global _client
    if not MONGODB_URI:
        raise DatabaseUnavailable(
            "MONGODB_URI is not set — add it to backend/.env (see .env.example)."
        )
    if _client is None:
        # Fail fast: a hung roster request is worse than a clear error.
        _client = AsyncIOMotorClient(MONGODB_URI, serverSelectionTimeoutMS=8000,
                                     connectTimeoutMS=8000, appname="VentAssist")
    return _client


def get_db() -> AsyncIOMotorDatabase:
    return get_client()[MONGODB_DB]


def patients() -> AsyncIOMotorCollection:
    return get_db()[PATIENTS]


def recommendations() -> AsyncIOMotorCollection:
    return get_db()[RECOMMENDATIONS]


def users() -> AsyncIOMotorCollection:
    return get_db()[USERS]


async def ping() -> bool:
    """True when the cluster answers. Never raises — callers use it as a health probe."""
    if not is_configured():
        return False
    try:
        await get_client().admin.command("ping")
        return True
    except Exception as e:  # network, auth, DNS — all mean "no database right now"
        log.warning("MongoDB ping failed: %s", e)
        return False


async def ensure_indexes() -> None:
    """History is always read patient-scoped and newest-first; index it that way."""
    await recommendations().create_index([("patient_id", 1), ("created_at", -1)])
    await patients().create_index([("created_at", 1)])
    # Unique, so two people signing up as the same clinician at the same moment
    # produce a duplicate-key error (→ HTTP 409) rather than two live accounts.
    await users().create_index("username", unique=True)


async def close() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
