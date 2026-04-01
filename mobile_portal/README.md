# starBoard Mobile Portal

Private internal mobile-first field portal for starBoard.

This subproject is intentionally isolated from the desktop PySide6 runtime. It provides:
- authenticated mobile API routes
- a mobile-first web frontend
- archive lookup and media delivery
- query/gallery submissions into the canonical archive

Development:
- backend: `mobile_portal/scripts/run_dev`
- tests: `cd /home/weertman/Documents/starBoard && ./scripts/test -q mobile_portal/tests`
