# Backend container for VentAssist (FastAPI + the trained policy).
#
# The frontend is NOT in here — it builds on Vercel as a static SPA and reaches
# this service through a rewrite, so the browser sees one origin and no CORS
# configuration is needed. See vercel.json.
#
# Build:  docker build -t ventassist-api .
# Run:    docker run -p 8000:8000 -e DATABASE_URL=... -e AUTH_SECRET=... ventassist-api

FROM python:3.11-slim

# libgomp1 is required by scikit-learn's OpenMP runtime; the slim image omits it
# and the import fails at runtime rather than at build time, which is a
# confusing way to find out.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so a code change does not re-download torch.
# The CPU index is not optional: the default wheel is the CUDA build, ~2 GB of
# GPU runtime that no free-tier host can use.
COPY backend/requirements-serve.txt ./backend/requirements-serve.txt
RUN pip install --no-cache-dir -r backend/requirements-serve.txt \
      --extra-index-url https://download.pytorch.org/whl/cpu

# Application code and the serving artifacts. .dockerignore keeps the 650 MB of
# offline-only models (bc_track_a.pkl, nwe_model*.pkl) and the 42 MB MDP parquet
# out of the image — see that file for what is deliberately excluded.
COPY backend/ ./backend/

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    API_HOST=0.0.0.0 \
    PROCESSED_PATH=backend/data/processed \
    MODEL_PATH=backend/models

# Fail the build if the serving artifacts did not make it in. Without this the
# container builds fine and then crashes on the first request, because
# PolicyRouter loads policy_track_a.pt in its constructor.
RUN python -c "\
from pathlib import Path; import sys;\
req = ['backend/models/policy_track_a.pt', 'backend/models/normaliser_stats.json',\
       'backend/models/action_map.json'];\
missing = [p for p in req if not Path(p).exists()];\
sys.exit('MISSING SERVING ARTIFACTS: %s\\n'\
         'They are gitignored by default — see .gitignore for the allow-list '\
         'that must keep them tracked.' % missing) if missing else \
print('serving artifacts present')"

EXPOSE 8000

# $PORT is injected by Render/Railway/Fly and ignored locally. One worker: each
# one loads its own copy of the policy, and these hosts' small instances do not
# have the RAM for two.
CMD ["sh", "-c", "uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
