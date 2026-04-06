#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

echo ""
echo -e "${CYAN}${BOLD}"
echo "  ██████╗ ███████╗ ██████╗ █████╗ ██╗     ██╗██╗  ██╗"
echo "  ██╔══██╗██╔════╝██╔════╝██╔══██╗██║     ██║╚██╗██╔╝"
echo "  ██████╔╝█████╗  ██║     ███████║██║     ██║ ╚███╔╝ "
echo "  ██╔══██╗██╔══╝  ██║     ██╔══██║██║     ██║ ██╔██╗ "
echo "  ██║  ██║███████╗╚██████╗██║  ██║███████╗██║██╔╝ ██╗"
echo "  ╚═╝  ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝╚═╝  ╚═╝"
echo -e "${RESET}${BOLD}                   V9 — Local AI Photo Search${RESET}"
echo ""

# ── Checks ────────────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo -e "${RED}[ERROR] python3 not found. Install Python 3.10+${RESET}"; exit 1
fi

HAVE_NODE=false
if command -v node &>/dev/null; then HAVE_NODE=true; fi

# ── Backend setup ─────────────────────────────────────────────────────────────
cd "$BACKEND"
mkdir -p data/thumbnails

echo -e "${YELLOW}[Setup]${RESET} Installing Python dependencies..."
pip3 install -r requirements.txt --quiet

# ── Frontend setup ────────────────────────────────────────────────────────────
if $HAVE_NODE; then
    if [ ! -d "$FRONTEND/node_modules" ]; then
        echo -e "${YELLOW}[Setup]${RESET} Installing frontend dependencies..."
        cd "$FRONTEND" && npm install
        cd "$BACKEND"
    fi
fi

# ── Cleanup on exit ───────────────────────────────────────────────────────────
BACKEND_PID=""
FRONTEND_PID=""
cleanup() {
    echo ""
    echo -e "${YELLOW}Stopping servers...${RESET}"
    [ -n "$BACKEND_PID"  ] && kill "$BACKEND_PID"  2>/dev/null
    [ -n "$FRONTEND_PID" ] && kill "$FRONTEND_PID" 2>/dev/null
    echo "Stopped."
    exit 0
}
trap cleanup SIGINT SIGTERM

# ── Start backend ─────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}[Start]${RESET} Backend → ${CYAN}http://localhost:8000${RESET}"
cd "$BACKEND"
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 &
BACKEND_PID=$!
sleep 2

# ── Start frontend ────────────────────────────────────────────────────────────
if $HAVE_NODE && [ -d "$FRONTEND/node_modules" ]; then
    echo -e "${GREEN}[Start]${RESET} Frontend → ${CYAN}http://localhost:5173${RESET}"
    cd "$FRONTEND"
    npm run dev &
    FRONTEND_PID=$!
    sleep 1
    echo ""
    echo -e "${BOLD}  Open: ${CYAN}http://localhost:5173${RESET}"
else
    echo ""
    echo -e "${BOLD}  Open: ${CYAN}http://localhost:8000${RESET}"
fi

echo -e "${YELLOW}  Press Ctrl+C to stop all servers${RESET}"
echo ""

wait
