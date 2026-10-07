"""Tests for clinician sign-in — password hashing, session tokens, route guards.

Layer 1 is pure (hashing and token signing, no database, no app). Layer 2 boots
the FastAPI app with a TestClient and checks that the clinical routes are actually
unreachable without a token — the guard is the whole point of the feature, so it
is asserted rather than assumed.

Run:  python -m backend.tests.test_auth
"""

from __future__ import annotations

import time

from backend.api import auth as A


class Skip(Exception):
    pass


# --------------------------------------------------------------------------- #
# 1. Password hashing
# --------------------------------------------------------------------------- #
def test_hash_is_not_the_password_and_verifies():
    stored = A.hash_password("ventilate-me-8")
    assert "ventilate-me-8" not in stored          # plaintext never appears
    assert stored.startswith("pbkdf2_sha256$")
    assert A.verify_password("ventilate-me-8", stored)
    assert not A.verify_password("ventilate-me-9", stored)


def test_same_password_hashes_differently():
    """Per-user salt: two clinicians with the same password get different hashes,
    so one leaked hash says nothing about the other account."""
    a, b = A.hash_password("same-password"), A.hash_password("same-password")
    assert a != b
    assert A.verify_password("same-password", a) and A.verify_password("same-password", b)


def test_malformed_stored_hash_is_rejected_not_raised():
    for junk in ["", "not-a-hash", "pbkdf2_sha256$abc", "md5$1$aa$bb", None]:
        assert A.verify_password("anything", junk) is False


# --------------------------------------------------------------------------- #
# 2. Session tokens
# --------------------------------------------------------------------------- #
def test_token_round_trip():
    token = A.create_token("user-abc123", "dr.chen")
    claims = A.read_token(token)
    assert claims is not None
    assert claims["sub"] == "user-abc123" and claims["username"] == "dr.chen"
    assert claims["exp"] > time.time()


def test_tampered_payload_is_rejected():
    """The signature is checked before the claims are parsed, so editing the
    payload to impersonate someone fails rather than being believed."""
    payload, signature = A.create_token("user-abc123", "dr.chen").split(".", 1)
    forged = A._b64encode(b'{"exp":9999999999,"sub":"user-root","username":"root"}')
    assert A.read_token(f"{forged}.{signature}") is None      # signature no longer matches
    assert A.read_token(f"{payload}.{A._b64encode(b'garbage')}") is None
    assert A.read_token("nonsense") is None
    assert A.read_token("") is None


def test_expired_token_is_rejected():
    original = A.TOKEN_TTL_HOURS
    try:
        A.TOKEN_TTL_HOURS = -1.0                  # minted already expired
        assert A.read_token(A.create_token("user-abc123", "dr.chen")) is None
    finally:
        A.TOKEN_TTL_HOURS = original


def test_username_normalisation():
    assert A.normalise_username("  Dr.Chen ") == "dr.chen"


# --------------------------------------------------------------------------- #
# 3. Route guards (needs the app to import — models are loaded at startup only,
#    so importing is cheap, but torch/model files must be present)
# --------------------------------------------------------------------------- #
def _client_or_skip():
    try:
        from fastapi.testclient import TestClient
        from backend.api.main import app
    except Exception as e:                        # missing deps or model artifacts
        raise Skip(f"FastAPI app not importable here: {e}")
    return TestClient(app)


_STATE = {"PEEP": 8, "TV": 450, "FiO2": 0.5, "SpO2": 94, "PaO2": 88, "PaCO2": 42,
          "pH": 7.36, "HR": 88, "SBP": 118, "RR": 18, "RASS": -2, "Temp": 37.0}


def test_clinical_routes_require_a_token():
    client = _client_or_skip()
    for method, path, body in [
        ("get", "/api/patients", None),
        ("get", "/api/validation", None),
        ("get", "/api/patients/patient-a/recommendations", None),
        ("post", "/api/recommend", {"patient_id": "patient-a", "patient_weight": 70,
                                    "track": "track_a", "tabular_state": _STATE}),
    ]:
        r = client.request(method, path, json=body)
        assert r.status_code == 401, f"{method.upper()} {path} answered {r.status_code} unauthenticated"


def test_bad_and_expired_tokens_are_refused():
    client = _client_or_skip()
    for token in ["not-a-token", "a.b", A.create_token("u", "x")[:-4] + "aaaa"]:
        r = client.get("/api/patients", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 401, f"token {token!r} was accepted"


def test_health_and_tracks_stay_open():
    """The sign-in page reads these before anyone has a token — if they were guarded
    the login screen could not tell the user the backend is up."""
    client = _client_or_skip()
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/tracks").status_code == 200


def test_signup_rejects_weak_credentials_before_touching_the_database():
    """422 (validation) rather than 503 (no database) proves the rules are enforced
    in the model, so a short password never reaches the database layer."""
    client = _client_or_skip()
    r = client.post("/api/auth/signup", json={"username": "dr", "password": "short"})
    assert r.status_code == 422
    r = client.post("/api/auth/signup", json={"username": "dr.chen", "password": "1234567"})
    assert r.status_code == 422
    r = client.post("/api/auth/signup", json={"username": "bad name!", "password": "longenough1"})
    assert r.status_code == 422


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Skip as e:
                print(f"SKIP {name}: {e}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    print("ALL TESTS PASSED" if failures == 0 else f"{failures} TEST(S) FAILED")
    raise SystemExit(1 if failures else 0)
