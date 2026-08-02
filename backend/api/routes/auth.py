"""/api/auth — clinician sign-up, sign-in and session check.

The three endpoints a clinician passes through before the roster is reachable:

  * ``POST /api/auth/signup``  create the account (username must be free)
  * ``POST /api/auth/login``   check the password against the stored PBKDF2 hash
  * ``GET  /api/auth/me``      re-validate a stored token on page load

Signup and login both return a token, so registering signs you straight in rather
than bouncing you back to a login form you just filled in.

The password itself is never stored, logged or returned — only its hash reaches
Mongo (see :mod:`backend.api.auth`). A wrong username and a wrong password give
the *same* 401 message on purpose: distinguishing them would let anyone probe the
endpoint for which clinicians have accounts.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pymongo.errors import DuplicateKeyError, PyMongoError

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
    """Accounts live in Mongo, so no database means no sign-in — say which it is."""
    return HTTPException(
        status_code=503,
        detail=f"Account database unavailable — cannot sign in right now: {e}",
    )


def _iso(v: Any) -> Optional[str]:
    if isinstance(v, datetime):
        return v.replace(tzinfo=v.tzinfo or timezone.utc).isoformat()
    return v if isinstance(v, str) else None


def _to_user(doc: dict) -> M.AuthUser:
    return M.AuthUser(
        id=str(doc["_id"]),
        username=doc["username"],
        full_name=doc.get("full_name"),
        role=doc.get("role"),
        created_at=_iso(doc.get("created_at")),
        last_login_at=_iso(doc.get("last_login_at")),
    )


def _session(doc: dict) -> M.AuthResponse:
    token = A.create_token(str(doc["_id"]), doc["username"])
    return M.AuthResponse(token=token, expires_at=A.token_expiry(token), user=_to_user(doc))


@router.post("/signup", response_model=M.AuthResponse, status_code=201)
async def signup(body: M.SignupRequest) -> M.AuthResponse:
    """Register a clinician and sign them in.

    The uniqueness of the username is enforced by the index, not by the pre-check:
    the ``find_one`` below only exists to turn the common case into a clear message,
    while ``DuplicateKeyError`` covers two signups racing for the same name.
    """
    username = A.normalise_username(body.username)
    now = datetime.now(timezone.utc)
    doc = {
        "_id": f"user-{uuid.uuid4().hex[:12]}",
        "username": username,
        "full_name": (body.full_name or "").strip() or None,
        "role": (body.role or "").strip() or None,
        "password_hash": A.hash_password(body.password),
        "created_at": now,
        "last_login_at": now,
    }

    try:
        if await db.users().find_one({"username": username}) is not None:
            raise HTTPException(status_code=409,
                                detail=f"The username {username!r} is already taken.")
        await db.users().insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=409,
                            detail=f"The username {username!r} is already taken.")
    except (db.DatabaseUnavailable, PyMongoError) as e:
        raise _guard(e)

    log.info("New clinician account: %s", username)
    return _session(doc)


@router.post("/login", response_model=M.AuthResponse)
async def login(body: M.LoginRequest) -> M.AuthResponse:
    username = A.normalise_username(body.username)
    try:
        doc = await db.users().find_one({"username": username})
    except (db.DatabaseUnavailable, PyMongoError) as e:
        raise _guard(e)

    if doc is None or not A.verify_password(body.password, doc.get("password_hash", "")):
        # Same answer either way — see the module docstring.
        raise BAD_CREDENTIALS

    now = datetime.now(timezone.utc)
    try:
        await db.users().update_one({"_id": doc["_id"]}, {"$set": {"last_login_at": now}})
    except PyMongoError as e:
        # A missing "last seen" stamp is not worth failing a valid sign-in over.
        log.warning("Could not record last_login_at for %s: %s", username, e)
    doc["last_login_at"] = now
    return _session(doc)


@router.get("/me", response_model=M.AuthUser)
async def me(user: A.CurrentUser = Depends(A.current_user)) -> M.AuthUser:
    """Who the held token belongs to — the frontend calls this on page load.

    Unlike the other protected routes this *does* read the database, so an account
    deleted from Mongo cannot keep browsing on a token that has not expired yet.
    """
    try:
        doc = await db.users().find_one({"_id": user.id})
    except (db.DatabaseUnavailable, PyMongoError) as e:
        raise _guard(e)
    if doc is None:
        raise HTTPException(status_code=401, detail="This account no longer exists.",
                            headers={"WWW-Authenticate": "Bearer"})
    return _to_user(doc)
