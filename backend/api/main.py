"""VentAssist FastAPI app — Dual-Track ventilator decision support."""

from __future__ import annotations

import logging
import os

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import db
from backend.api.auth import current_user
from backend.api.routes.auth import router as auth_router
from backend.api.routes.patients import router as patients_router
from backend.api.routes.recommend import router as recommend_router
from backend.api.routes.validation import router as validation_router
from backend.api.seed import seed_presets
from backend.api.state import get_services

log = logging.getLogger(__name__)

app = FastAPI(title="VentAssist", version="1.0.0",
              description="Dual-Track offline-RL ventilator decision support")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["*"], allow_headers=["*"], allow_credentials=True,
)

# Signing in is the only thing an anonymous caller may do; every clinical route
# below is guarded, so patient data and recommendations are unreachable without a
# valid session token. `/api/health` and `/api/tracks` stay open — they carry no
# patient data and the sign-in page reads them to show whether the system is up.
app.include_router(auth_router, prefix="/api")

_signed_in = [Depends(current_user)]
app.include_router(recommend_router, prefix="/api", dependencies=_signed_in)
app.include_router(validation_router, prefix="/api", dependencies=_signed_in)
app.include_router(patients_router, prefix="/api", dependencies=_signed_in)


@app.on_event("startup")
async def _warm() -> None:
    get_services()   # load models once at startup
    # The roster lives in Mongo, but inference does not depend on it — a database
    # that is down degrades the roster to a 503, it does not stop the app booting.
    if await db.ping():
        await db.ensure_indexes()
        await seed_presets()
    else:
        log.warning("MongoDB unreachable at startup — /api/patients will return 503 "
                    "and recommendations will not be saved.")


@app.on_event("shutdown")
async def _shutdown() -> None:
    await db.close()


@app.get("/api/health")
async def health() -> dict:
    svc = get_services()
    return {
        "status": "ok",
        "track_a": svc.router.track_a is not None,
        "track_b": svc.router.track_b is not None,
        "database": await db.ping(),
    }


@app.get("/api/tracks")
def tracks() -> dict:
    svc = get_services()
    return {
        "tracks": [
            {"id": "track_a", "label": "Clinical Data Policy",
             "available": True, "waveform_required": False},
            {"id": "track_b", "label": "Waveform-Enhanced Policy (proof-of-concept)",
             "available": svc.router.track_b is not None, "waveform_required": True},
        ]
    }
