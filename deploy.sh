#!/bin/bash
# ============================================================
# The Bake Story - EC2 Deployment Script
# EC2 IP: 54.162.20.141
# Domain: thebakestory.store
# Run as: ubuntu user on EC2
# ============================================================

set -e  # Exit immediately on error

echo "============================================"
echo "  The Bake Story - EC2 Deployment Script"
echo "============================================"

# ─── 1. Update System Packages ───────────────────────────────
echo "[1/9] Updating system packages..."
sudo apt-get update -y && sudo apt-get upgrade -y
sudo apt-get install -y python3 python3-pip python3-venv nginx git curl

# ─── 2. Clone or Pull Project ────────────────────────────────
echo "[2/9] Setting up project directory..."
APP_DIR="/home/ubuntu/bakery"

if [ -d "$APP_DIR/.git" ]; then
    echo "  → Pulling latest changes..."
    cd "$APP_DIR"
    git pull origin main
else
    echo "  → Cloning repository..."
    git clone https://github.com/MYPROJECTSbandelaajay360-gmail-com/TheBakeStory.git "$APP_DIR"
fi

cd "$APP_DIR"

# ─── 3. Create Virtual Environment ───────────────────────────
echo "[3/9] Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# ─── 4. Install Dependencies ──────────────────────────────────
echo "[4/9] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# ─── 5. Configure Environment ────────────────────────────────
echo "[5/9] Configuring environment..."
cd bakery_project

if [ ! -f ".env" ]; then
    echo "  → Creating .env from .env.production template..."
    cp ../.env.production .env
    echo ""
    echo "  ⚠  IMPORTANT: Edit .env and fill in your real credentials!"
    echo "     nano .env"
    echo ""
fi

# ─── 6. Django Setup ──────────────────────────────────────────
echo "[6/9] Running Django setup..."
python manage.py migrate --noinput
python manage.py collectstatic --noinput
mkdir -p logs media

# Seed initial data if needed (comment out if already seeded)
# python manage.py init_menu
# python manage.py init_admin

# ─── 7. Create Systemd Service ───────────────────────────────
echo "[7/9] Creating systemd service for gunicorn..."
sudo tee /etc/systemd/system/bakery.service > /dev/null <<EOF
[Unit]
Description=The Bake Story - Gunicorn Daemon
After=network.target

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/bakery/bakery_project
Environment="PATH=/home/ubuntu/bakery/venv/bin"
EnvironmentFile=/home/ubuntu/bakery/bakery_project/.env
ExecStart=/home/ubuntu/bakery/venv/bin/gunicorn \\
    bakery_project.wsgi:application \\
    --bind 127.0.0.1:8000 \\
    --workers 3 \\
    --timeout 120 \\
    --log-level warning \\
    --access-logfile /home/ubuntu/bakery/bakery_project/logs/access.log \\
    --error-logfile /home/ubuntu/bakery/bakery_project/logs/error.log
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable bakery
sudo systemctl restart bakery
echo "  ✓ Gunicorn service started"

# ─── 8. Configure Nginx ───────────────────────────────────────
echo "[8/9] Configuring Nginx..."
sudo cp /home/ubuntu/bakery/nginx.conf /etc/nginx/sites-available/bakery
sudo ln -sf /etc/nginx/sites-available/bakery /etc/nginx/sites-enabled/bakery
sudo rm -f /etc/nginx/sites-enabled/default   # Remove default site

# Test nginx config
sudo nginx -t
sudo systemctl restart nginx
sudo systemctl enable nginx
echo "  ✓ Nginx configured and started"

# ─── 9. Firewall ─────────────────────────────────────────────
echo "[9/9] Configuring firewall..."
sudo ufw allow 'Nginx Full'
sudo ufw allow 22   # SSH
sudo ufw --force enable
echo "  ✓ Firewall rules set"

echo ""
echo "============================================"
echo "  Deployment Complete!"
echo "============================================"
echo "  App URL (IP):    http://54.162.20.141"
echo "  App URL (Domain): http://thebakestory.shop"
echo ""
echo "  Useful commands:"
echo "  Check gunicorn: sudo systemctl status bakery"
echo "  Check nginx:    sudo systemctl status nginx"
echo "  View logs:      tail -f /home/ubuntu/bakery/bakery_project/logs/error.log"
echo "  Restart app:    sudo systemctl restart bakery"
echo "============================================"
