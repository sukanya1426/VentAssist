#!/usr/bin/env bash
# Start everything VentAssist needs, in order, and nothing else.
#
# Docker Desktop does NOT start on its own (AutoStart is false, and it is not a
# login item), and nothing in the app can start it — the API just reports
# "database": false and every account/roster route answers 503. So this script is
# the bridge: it launches Docker only when you actually want to work, waits for
# the daemon, brings up the one container this project needs, and leaves every
# other container alone.
#
# Why it starts only ventassist-postgres: this machine also holds a 4-node kind
# (Kubernetes) cluster that burns ~146% CPU and ~2.9 GB of RAM when running,
# which is what makes the laptop hot. Those nodes are set to restart=no so they
# stay down until you ask for them; postgres is restart=unless-stopped, costs
# ~0.01% CPU and 21 MB, and comes back by itself once the daemon is up.
#
#   ./run.sh            backend only (plus Docker + postgres)
#   ./run.sh --all      also start the Vite dev server
#   ./run.sh --stop     stop the backend and the postgres container
#
set -euo pipefail
cd "$(dirname "$0")"

CONTAINER=ventassist-postgres
API_PORT=8000

say() { printf '\033[36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33m!!\033[0m %s\n' "$*"; }

if [[ "${1:-}" == "--stop" ]]; then
  say "stopping the backend"
  pkill -f "uvicorn backend.api.main" 2>/dev/null && echo "   stopped" || echo "   was not running"
  say "stopping $CONTAINER"
  docker stop "$CONTAINER" >/dev/null 2>&1 && echo "   stopped" || echo "   was not running"
  say "done — Docker Desktop is still running; quit it from the menu bar to cool down"
  exit 0
fi

# --- 1. Docker daemon ------------------------------------------------------- #
if ! docker info >/dev/null 2>&1; then
  say "Docker is not running — launching Docker Desktop"
  open -a Docker
  printf '   waiting for the daemon'
  for _ in $(seq 1 60); do
    if docker info >/dev/null 2>&1; then echo " ready"; break; fi
    printf '.'; sleep 2
  done
  docker info >/dev/null 2>&1 || { echo; warn "daemon did not come up — open Docker Desktop manually"; exit 1; }
else
  say "Docker daemon already running"
fi

# --- 2. PostgreSQL ---------------------------------------------------------- #
if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" == "true" ]]; then
  say "$CONTAINER already up"
else
  say "starting $CONTAINER"
  docker start "$CONTAINER" >/dev/null
fi
printf '   waiting for postgres'
for _ in $(seq 1 40); do
  if docker exec "$CONTAINER" pg_isready -U ventassist -d ventassist >/dev/null 2>&1; then
    echo " ready"; break
  fi
  printf '.'; sleep 1
done

# Anything else running is not needed by this project and only adds heat.
others=$(docker ps --format '{{.Names}}' | grep -v "^${CONTAINER}$" || true)
if [[ -n "$others" ]]; then
  warn "other containers are running and will heat the laptop:"
  echo "$others" | sed 's/^/     /'
  echo "     stop them with: docker stop $(echo "$others" | tr '\n' ' ')"
fi

# --- 3. Backend ------------------------------------------------------------- #
if lsof -nP -iTCP:$API_PORT -sTCP:LISTEN >/dev/null 2>&1; then
  warn "port $API_PORT is already in use — a stale backend serves OLD code, which is"
  warn "the usual cause of unexplained 500s. Kill it with:"
  warn "  pkill -f 'uvicorn backend.api.main'"
  exit 1
fi

if [[ "${1:-}" == "--all" ]]; then
  say "starting the frontend (background) → http://localhost:5173"
  (cd frontend && npm run dev >/tmp/ventassist-vite.log 2>&1 &)
fi

say "starting the backend → http://127.0.0.1:$API_PORT"
echo
PYTHONPATH=. exec .venv/bin/uvicorn backend.api.main:app \
  --reload --host 127.0.0.1 --port "$API_PORT"
