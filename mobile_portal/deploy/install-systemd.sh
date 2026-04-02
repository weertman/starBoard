#!/usr/bin/env bash
set -euo pipefail

sudo bash -lc 'cat >/etc/systemd/system/starboard-mobile-portal.service <<"EOF"
[Unit]
Description=starBoard Mobile Portal (FastAPI on port 8091)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=weertman
Group=weertman
WorkingDirectory=/home/weertman/Documents/starBoard
Environment=PYTHONPATH=/home/weertman/Documents/starBoard
Environment=STARBOARD_MOBILE_HOST=127.0.0.1
Environment=STARBOARD_MOBILE_PORT=8091
ExecStart=/home/weertman/miniforge3/envs/starboard-py311/bin/python -m uvicorn mobile_portal.app.main:app --host 127.0.0.1 --port 8091
Restart=on-failure
RestartSec=5s
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now starboard-mobile-portal.service
systemctl status starboard-mobile-portal.service --no-pager'
