#!/usr/bin/env python
"""
Direct launcher for Sunflower Star Lineup GUI
This uses all the updated files with fixes
"""
import os
import sys

# Set environment variables BEFORE any imports
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['PYTORCH_DISABLE_LIBRARY_INIT'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Patch typing.Self for Python <3.11
import typing

if not hasattr(typing, 'Self'):
    try:
        from typing_extensions import Self

        typing.Self = Self
    except ImportError:
        typing.Self = type('Self', (), {})

# Set up paths
from pathlib import Path

script_dir = Path(__file__).parent
os.chdir(script_dir)
sys.path.insert(0, str(script_dir.parent))

# Now we can import everything else
from PySide6.QtWidgets import QApplication
from lineup_app import SunflowerLineupApp


def main():
    # Create Qt application (or get existing instance)
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("Sunflower Star Lineup")
    app.setStyle('Fusion')

    # File paths
    reid_checkpoint = script_dir / "best_model.pth"
    yolo_checkpoint = script_dir / "starseg_best.pt"
    database_path = script_dir / "lineup_db.npz"

    # Verify files exist
    for file_path, name in [(reid_checkpoint, "ReID model"),
                            (yolo_checkpoint, "YOLO model"),
                            (database_path, "Database")]:
        if not file_path.exists():
            print(f"Error: {name} not found at {file_path}")
            return 1

    try:
        # Create main window
        print("Loading models and database...")
        window = SunflowerLineupApp(
            reid_checkpoint=str(reid_checkpoint),
            yolo_checkpoint=str(yolo_checkpoint),
            database_path=str(database_path)
        )

        # Show window
        window.show()
        window.raise_()  # Bring to front
        window.activateWindow()  # Make active

        print("\n✓ Lineup GUI is now running!")
        print("Close the window to exit.\n")

        # Run event loop - this blocks until window is closed
        return app.exec()

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

        # Show error dialog
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(None, "Startup Error",
                             f"Failed to start lineup application:\n\n{str(e)}")
        return 1


if __name__ == '__main__':
    sys.exit(main())