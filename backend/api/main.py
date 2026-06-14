"""VentAssist FastAPI app — Dual-Track ventilator decision support."""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes.recommend import router as recommend_router
from backend.api.state import get_services

app = FastAPI(title="VentAssist", version="1.0.0",
              description="Dual-Track offline-RL ventilator decision support")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["*"], allow_headers=["*"], allow_credentials=True,
)

app.include_router(recommend_router, prefix="/api")


@app.on_event("startup")
def _warm() -> None:
    get_services()   # load models once at startup


@app.get("/api/health")
def health() -> dict:
    svc = get_services()
    return {
        "status": "ok",
        "track_a": svc.router.track_a is not None,
        "track_b": svc.router.track_b is not None,
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
