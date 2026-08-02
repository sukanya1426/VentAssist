"""Password hashing and session tokens for the clinician sign-in.

Two deliberately small pieces, both on the standard library:

  * **Passwords** are stored as ``pbkdf2_sha256$<iterations>$<salt>$<hash>`` —
    PBKDF2-HMAC-SHA256 with a per-user random salt. The plaintext is never
    written anywhere, and verification is a constant-time digest compare.
  * **Sessions** are stateless HMAC-signed tokens (``payload.signature``, both
    base64url). The payload carries the user id, the username and an expiry, so a
    protected request is answered without a database round-trip — which matters
    because the recommendation endpoint is meant to keep working while Mongo is
    unreachable. The cost of statelessness is that a token cannot be revoked
    before it expires; the short lifetime (one shift) is the mitigation.

``AUTH_SECRET`` signs the tokens. If it is unset a random one is generated per
process, which is safe but means every restart invalidates outstanding sessions —
fine for development, so set it in ``backend/.env`` for anything longer-lived.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.pipeline import config  # noqa: F401  — imports load backend/.env

log = logging.getLogger(__name__)

# --- password hashing ------------------------------------------------------- #

_ALGORITHM = "pbkdf2_sha256"
_ITERATIONS = 240_000
_SALT_BYTES = 16

MIN_PASSWORD_LENGTH = 8


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITERATIONS)
    return f"{_ALGORITHM}${_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time check of a plaintext against a stored hash. Never raises."""
    try:
        algorithm, iterations, salt_hex, digest_hex = stored.split("$")
        if algorithm != _ALGORITHM:
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt_hex), int(iterations)
        )
    except (ValueError, AttributeError):
        return False
    return hmac.compare_digest(digest.hex(), digest_hex)


# --- session tokens --------------------------------------------------------- #

TOKEN_TTL_HOURS = float(os.getenv("AUTH_TOKEN_TTL_HOURS", "12"))   # one shift

_SECRET = os.getenv("AUTH_SECRET", "")
if not _SECRET:
    _SECRET = secrets.token_urlsafe(48)
    log.warning("AUTH_SECRET is not set — using a per-process random secret. "
                "Sessions will not survive a backend restart; set AUTH_SECRET in "
                "backend/.env (see .env.example).")


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _b64decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _sign(payload: str) -> str:
    return _b64encode(hmac.new(_SECRET.encode(), payload.encode(), hashlib.sha256).digest())


def create_token(user_id: str, username: str) -> str:
    expires = datetime.now(timezone.utc) + timedelta(hours=TOKEN_TTL_HOURS)
    payload = _b64encode(json.dumps(
        {"sub": user_id, "username": username, "exp": int(expires.timestamp())},
        separators=(",", ":"), sort_keys=True,
    ).encode())
    return f"{payload}.{_sign(payload)}"


def token_expiry(token: str) -> Optional[str]:
    claims = read_token(token)
    if claims is None:
        return None
    return datetime.fromtimestamp(claims["exp"], timezone.utc).isoformat()


def read_token(token: str) -> Optional[dict]:
    """Claims of a valid, unexpired, correctly-signed token; ``None`` otherwise.

    The signature is checked *before* the payload is parsed, so an attacker's
    handcrafted claims are never read, only rejected.
    """
    try:
        payload, signature = token.split(".", 1)
    except (ValueError, AttributeError):
        return None
    if not hmac.compare_digest(signature, _sign(payload)):
        return None
    try:
        claims = json.loads(_b64decode(payload))
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(claims, dict) or "sub" not in claims or "exp" not in claims:
        return None
    if datetime.now(timezone.utc).timestamp() > float(claims["exp"]):
        return None
    return claims


# --- FastAPI dependency ----------------------------------------------------- #

class CurrentUser:
    """The signed-in clinician, as carried by the token."""

    def __init__(self, user_id: str, username: str):
        self.id = user_id
        self.username = username


_bearer = HTTPBearer(auto_error=False, description="Session token from /api/auth/login")

_UNAUTHENTICATED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Sign in to use VentAssist — your session is missing or has expired.",
    headers={"WWW-Authenticate": "Bearer"},
)


def current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> CurrentUser:
    """Guard for every clinical route. 401 means "sign in again", not "server broke"."""
    if credentials is None or not credentials.credentials:
        raise _UNAUTHENTICATED
    claims = read_token(credentials.credentials)
    if claims is None:
        raise _UNAUTHENTICATED
    return CurrentUser(str(claims["sub"]), str(claims.get("username", "")))


# --- credential rules ------------------------------------------------------- #

def normalise_username(username: str) -> str:
    """Usernames are matched case- and space-insensitively, stored lower-case, so
    ``Dr.Chen`` and ``dr.chen`` are the same account rather than two silent ones."""
    return username.strip().lower()
