"""/api/auth — clinician sign-up, sign-in and session check.

The three endpoints a clinician passes through before the roster is reachable:

  * ``POST /api/auth/signup``  create the account (username must be free)
  * ``POST /api/auth/login``   check the password against the stored PBKDF2 hash
  * ``GET  /api/auth/me``      re-validate a stored token on page load

Signup and login both return a token, so registering signs you straight in rather
than bouncing you back to a login form you just filled in.

The password itself is never stored, logged or returned — only its hash reaches
the database (see :mod:`backend.api.auth`). A wrong username and a wrong password give
the *same* 401 message on purpose: distinguishing them would let anyone probe the
endpoint for which clinicians have accounts.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import asyncpg
from fastapi import APIRouter, Depends, HTTPException

from backend.api import auth as A
from backend.api import db
from backend.api import models as M

router = APIRouter(prefix="/auth", tags=["auth"])
log = logging.getLogger(__name__)

BAD_CREDENTIALS = HTTPException(
    status_code=401,
    detail="Incorrect username or password.",
    headers={"WWW-Authenticate": "Bearer"},
)


def _guard(e: Exception) -> HTTPException:
    """Accounts live in the database, so no database means no sign-in — say so."""
    return HTTPException(
        status_code=503,
        detail=f"Account database unavailable — cannot sign in right now: {e}",
    )


def _iso(v: Any) -> Optional[str]:
    if isinstance(v, datetime):
        return v.replace(tzinfo=v.tzinfo or timezone.utc).isoformat()
    return v if isinstance(v, str) else None


def _to_user(row) -> M.AuthUser:
    return M.AuthUser(
        id=row["user_id"],
        username=row["username"],
        full_name=row["full_name"],
        role=row["role"],
        created_at=_iso(row["created_at"]),
        last_login_at=_iso(row["last_login_at"]),
    )


def _session(row) -> M.AuthResponse:
    token = A.create_token(row["user_id"], row["username"])
    return M.AuthResponse(token=token, expires_at=A.token_expiry(token), user=_to_user(row))


@router.post("/signup", response_model=M.AuthResponse, status_code=201)
async def signup(body: M.SignupRequest) -> M.AuthResponse:
    """Register a clinician and sign them in.

    Uniqueness is enforced by the UNIQUE constraint on ``username``, not by a
    pre-check: a SELECT-then-INSERT would let two signups racing for the same name
    both pass the check. The INSERT is the authority, and
    ``UniqueViolationError`` is the one case it rejects.
    """
    username = A.normalise_username(body.username)
    now = datetime.now(timezone.utc)
    user_id = f"user-{uuid.uuid4().hex[:12]}"

    try:
        row = await db.fetchrow(
            """
            INSERT INTO clinician (user_id, username, password_hash, full_name,
                                   role, created_at, last_login_at)
            VALUES ($1, $2, $3, $4, $5, $6, $6)
            RETURNING *
            """,
            user_id, username, A.hash_password(body.password),
            (body.full_name or "").strip() or None,
            (body.role or "").strip() or None, now,
        )
    except asyncpg.UniqueViolationError:
        raise HTTPException(status_code=409,
                            detail=f"The username {username!r} is already taken.")
    except db.DB_ERRORS as e:
        raise _guard(e)

    log.info("New clinician account: %s", username)
    return _session(row)


@router.post("/login", response_model=M.AuthResponse)
async def login(body: M.LoginRequest) -> M.AuthResponse:
    username = A.normalise_username(body.username)
    try:
        row = await db.fetchrow(
            "SELECT * FROM clinician WHERE username = $1", username)
    except db.DB_ERRORS as e:
        raise _guard(e)

    if row is None or not A.verify_password(body.password, row["password_hash"] or ""):
        # Same answer either way — see the module docstring.
        raise BAD_CREDENTIALS

    now = datetime.now(timezone.utc)
    try:
        row = await db.fetchrow(
            "UPDATE clinician SET last_login_at = $2 WHERE user_id = $1 RETURNING *",
            row["user_id"], now)
    except db.DB_ERRORS as e:
        # A missing "last seen" stamp is not worth failing a valid sign-in over.
        log.warning("Could not record last_login_at for %s: %s", username, e)
    return _session(row)


@router.get("/me", response_model=M.AuthUser)
async def me(user: A.CurrentUser = Depends(A.current_user)) -> M.AuthUser:
    """Who the held token belongs to — the frontend calls this on page load.

    Unlike the other protected routes this *does* read the database, so a deleted
    account cannot keep browsing on a token that has not expired yet.
    """
    try:
        doc = await db.fetchrow("SELECT * FROM clinician WHERE user_id = $1", user.id)
    except db.DB_ERRORS as e:
        raise _guard(e)
    if doc is None:
        raise HTTPException(status_code=401, detail="This account no longer exists.",
                            headers={"WWW-Authenticate": "Bearer"})
    return _to_user(doc)
