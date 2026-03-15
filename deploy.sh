#!/bin/bash
# ─────────────────────────────────────────────────────────────
# AutoTOS — LAN Deployment Script
# Run this once on the server PC: bash deploy.sh
# ─────────────────────────────────────────────────────────────

set -e   # stop on any error

echo ""
echo "╔══════════════════════════════════════╗"
echo "║     AutoTOS — Docker LAN Deploy      ║"
echo "╚══════════════════════════════════════╝"
echo ""

# ── 1. Check Docker is installed ────────────────────────────
if ! command -v docker &> /dev/null; then
    echo "⚠  Docker not found. Installing..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker $USER
    echo "✅ Docker installed. Please log out and back in, then re-run this script."
    exit 0
fi
echo "✅ Docker found: $(docker --version)"

# ── 2. Check .env exists ─────────────────────────────────────
if [ ! -f ".env" ]; then
    echo ""
    echo "⚠  .env file not found."
    echo "   Copying .env.example → .env"
    cp .env.example .env
    echo ""
    echo "   ‼  Please edit .env with your passwords, then re-run: bash deploy.sh"
    echo "      nano .env"
    exit 1
fi
echo "✅ .env found"

# ── 3. Create required folders ──────────────────────────────
mkdir -p uploads models
echo "✅ uploads/ and models/ folders ready"

# ── 4. Build and start containers ───────────────────────────
echo ""
echo "🔨 Building containers (first run may take a few minutes)..."
docker compose down --remove-orphans 2>/dev/null || true
docker compose up --build -d

# ── 5. Wait for DB to be ready ──────────────────────────────
echo ""
echo "⏳ Waiting for MySQL to be ready..."
sleep 15

# ── 6. Run DB migrations ─────────────────────────────────────
echo ""
echo "📦 Running database migrations..."
docker compose exec web flask db upgrade || {
    echo ""
    echo "⚠  Migration failed. This might be the first run — trying db init..."
    docker compose exec web flask db init 2>/dev/null || true
    docker compose exec web flask db migrate -m "initial" 2>/dev/null || true
    docker compose exec web flask db upgrade
}

# ── 7. Show status ───────────────────────────────────────────
echo ""
echo "📊 Container status:"
docker compose ps

# ── 8. Show LAN IP ───────────────────────────────────────────
echo ""
LAN_IP=$(hostname -I | awk '{print $1}')
echo "╔══════════════════════════════════════════════╗"
echo "║  ✅ AutoTOS is running!                       ║"
echo "║                                              ║"
echo "║  Open in browser on any device on this WiFi: ║"
echo "║  👉  http://${LAN_IP}:5000                  ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "Useful commands:"
echo "  View logs:    docker compose logs -f web"
echo "  AI logs:      docker compose logs -f ai"
echo "  Stop all:     docker compose down"
echo "  Restart web:  docker compose restart web"