#!/usr/bin/env python3
"""
starMorphometricTool - Main Entry Point

A tool for measuring the morphology of sunflower sea stars (Pycnopodia helianthoides).

Run this script from the project root directory to start the application.
"""
import sys
import os
import logging

# Add the source directory to the Python path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'starMorphometricTool')
sys.path.insert(0, src_path)

# Configure logging
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'debug_log.txt')
logging.basicConfig(filename=log_path, level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s')
# Empty the log file on start
open(log_path, 'w').close()

# Set working directory to project root for relative paths
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow


def main():
    """Start the Morphometric Tool application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


