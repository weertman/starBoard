# src/ui/main_window.py
from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel, QSizePolicy
)

from .tab_setup import TabSetup
from .tab_first_order import TabFirstOrder
from .tab_second_order import TabSecondOrder
from .tab_gallery_review import TabGalleryReview
from .tab_past_matches import TabPastMatches
from .tab_dl import TabDeepLearning
from src.utils.interaction_logger import get_interaction_logger

# Conditional import for Morphometric tab
try:
    from .tab_morphometric import TabMorphometric
    MORPHOMETRIC_AVAILABLE = True
except ImportError:
    MORPHOMETRIC_AVAILABLE = False

logger = logging.getLogger("starBoard.ui.main_window")


class _PlaceholderTab(QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lbl = QLabel(f"{title} — coming soon")
        lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(lbl)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("starBoard")
        self.setMinimumSize(480, 360)
        self.resize(1280, 860)

        tabs = QTabWidget()
        tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        t_setup = TabSetup()
        t_first = TabFirstOrder()

        # NEW: install Prev/Next gallery navigation under the First‑order gallery pane.
        try:
            t_first.add_gallery_nav_toolbar()
        except Exception:
            # Never let a layout tweak break app startup.
            pass

        t_second = TabSecondOrder()

        # Wire the quick jump from Second‑order back to whatever is selected in First‑order
        try:
            t_second.add_first_order_sync(t_first)
        except Exception:
            pass

        t_past = TabPastMatches()
        t_dl = TabDeepLearning()

        # Wire DL precompute completion to refresh First-order visual controls
        try:
            t_dl.precomputeCompleted.connect(t_first.refresh_visual_state)
        except Exception:
            pass
        
        # Wire DL verification precompute completion to refresh First-order verification controls
        try:
            t_dl.verificationPrecomputeCompleted.connect(t_first.refresh_verification_state)
        except Exception:
            pass
        
        # Wire DL evaluation completion to refresh First-order query sorting
        try:
            t_dl.evaluationCompleted.connect(t_first.refresh_evaluation_state)
        except Exception:
            pass
        
        # Wire match decisions to update First-order evaluation sorting
        try:
            t_second.matchDecisionMade.connect(
                lambda qid, verdict: t_first.on_match_decision_made(qid) if verdict == "yes" else None
            )
        except Exception:
            pass

        tabs.addTab(t_setup, "Data Entry")
        
        # Morphometric tab (inserted between Setup and First-order)
        t_morph = None
        if MORPHOMETRIC_AVAILABLE:
            try:
                t_morph = TabMorphometric()
                tabs.addTab(t_morph, "Morphometric")
                logger.info("Morphometric tab enabled")
            except Exception as e:
                logger.warning("Failed to create Morphometric tab: %s", e)
                t_morph = None
        
        # Wire morphometric data saves to refresh First-order and Second-order
        if t_morph:
            try:
                t_morph.dataSaved.connect(t_first.on_archive_data_changed)
            except Exception:
                pass
            try:
                t_morph.dataSaved.connect(t_second._refresh_ids)
            except Exception:
                pass
        
        tabs.addTab(t_first, "First-order")
        tabs.addTab(t_second, "Second-order")
        
        # Gallery Review tab (after Second-order)
        t_gallery_review = TabGalleryReview()
        tabs.addTab(t_gallery_review, "Gallery Review")
        
        tabs.addTab(t_past, "Analytics & History")
        tabs.addTab(t_dl, "Deep Learning")
        
        # Note: Morphometric tab has its own _notify_first_order_refresh
        # Signal kept for potential external listeners
        self._t_morph = t_morph  # Store reference if needed

        self.setCentralWidget(tabs)
        self.statusBar().showMessage("Ready")
        
        # Track tab changes for interaction logging
        self._tabs = tabs
        self._interaction_logger = get_interaction_logger()
        tabs.currentChanged.connect(self._on_tab_changed)
        # Set initial tab context
        self._on_tab_changed(tabs.currentIndex())

        # Check for first boot prompt (after UI is ready)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(500, lambda: self._check_first_boot(t_dl, tabs))

    def _on_tab_changed(self, index: int) -> None:
        """Handle tab change - update interaction logger context."""
        tab_name = self._tabs.tabText(index) if index >= 0 else ""
        self._interaction_logger.set_current_tab(tab_name)
        self._interaction_logger.log(
            "tab_switch",
            "main_tabs",
            value=tab_name,
            context={"tab_index": index},
        )

    def _check_first_boot(self, t_dl, tabs):
        """Check for first boot and optionally switch to DL tab."""
        try:
            # Only prompt if DL is available but not precomputed
            from src.dl import DL_AVAILABLE
            if DL_AVAILABLE:
                from src.dl.registry import DLRegistry
                registry = DLRegistry.load()
                if not registry.first_boot_completed and not registry.has_precomputed_model():
                    # Switch to DL tab and show prompt
                    tabs.setCurrentWidget(t_dl)
                    t_dl.check_first_boot()
        except Exception:
            pass  # Fail silently - DL is optional
