"""
starBoard Sync Server — entry point.

Usage:
    python -m src.sync                    # Start the sync server on port 8090
    python -m src.sync --port 9000        # Custom port
    python -m src.sync --rebuild-index    # Rebuild the catalog index and exit
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root on path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def main():
    parser = argparse.ArgumentParser(
        description="starBoard Sync Server",
    )
    parser.add_argument(
        "--port", type=int, default=8090,
        help="Port to listen on (default: 8090)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--rebuild-index", action="store_true",
        help="Rebuild the sync index and exit",
    )
    parser.add_argument(
        "--compute-hashes", action="store_true",
        help="Compute SHA-256 hashes during index rebuild (slow)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.rebuild_index:
        from src.sync.server_index import SyncIndex
        from src.sync.server import ARCHIVE_ROOT

        print(f"Rebuilding index for {ARCHIVE_ROOT}...")
        idx = SyncIndex(ARCHIVE_ROOT)
        stats = idx.rebuild_index(compute_hashes=args.compute_hashes)
        print(f"Done: {stats}")
        idx.close()
        return

    # Start the server
    import uvicorn
    from src.sync.config import get_lab_id

    print(f"starBoard Sync Server")
    print(f"  Archive:  {_project_root / 'archive'}")
    print(f"  Lab ID:   {get_lab_id()}")
    print(f"  Binding:  {args.host}:{args.port}")
    print()

    uvicorn.run(
        "src.sync.server:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
