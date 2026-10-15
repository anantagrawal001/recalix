#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"

G='\033[0;32m'; Y='\033[1;33m'; C='\033[0;36m'; B='\033[1m'; R='\033[0m'

echo ""
echo -e "${C}${B}"
echo "  ██████╗ ███████╗ ██████╗ █████╗ ██╗     ██╗██╗  ██╗"
echo "  ██╔══██╗██╔════╝██╔════╝██╔══██╗██║     ██║╚██╗██╔╝"
echo "  ██████╔╝█████╗  ██║     ███████║██║     ██║ ╚███╔╝ "
echo "  ██╔══██╗██╔══╝  ██║     ██╔══██║██║     ██║ ██╔██╗ "
echo "  ██║  ██║███████╗╚██████╗██║  ██║███████╗██║██╔╝ ██╗"
echo "  ╚═╝  ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝╚═╝  ╚═╝"
echo -e "${R}${B}                  V9 — Local AI Photo Search${R}"
echo ""

command -v python3 &>/dev/null || { echo "Python 3 required."; exit 1; }

cd "$BACKEND"
mkdir -p data/thumbnails
echo -e "${Y}[Setup]${R} Installing Python dependencies..."
pip3 install -r requirements.txt -q

BACKEND_PID=""; FRONTEND_PID=""
cleanup() {
  echo ""; echo -e "${Y}Stopping...${R}"
  [ -n "$BACKEND_PID"  ] && kill "$BACKEND_PID"  2>/dev/null
  [ -n "$FRONTEND_PID" ] && kill "$FRONTEND_PID" 2>/dev/null
  exit 0
}
trap cleanup SIGINT SIGTERM

echo -e "${G}[Start]${R} Backend  → ${C}http://localhost:8000${R}"
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 &
BACKEND_PID=$!
sleep 2

if command -v node &>/dev/null; then
  [ ! -d "$FRONTEND/node_modules" ] && { echo -e "${Y}[Setup]${R} npm install..."; cd "$FRONTEND" && npm install; cd "$BACKEND"; }
  echo -e "${G}[Start]${R} Frontend → ${C}http://localhost:5173${R}"
  cd "$FRONTEND" && npm run dev &
  FRONTEND_PID=$!
  sleep 1
  echo -e "\n${B}  Open: ${C}http://localhost:5173${R}"
else
  echo -e "\n${B}  Open: ${C}http://localhost:8000${R}"
fi

echo -e "${Y}  Ctrl+C to stop${R}\n"
wait
