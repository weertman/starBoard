# src/ui/tab_gallery_review.py
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QCompleter, QMessageBox, QDialog,
)

from src.data.id_registry import list_ids
from src.data.image_index import list_image_files, invalidate_image_cache
from src.data import archive_paths as ap
from src.data.best_photo import reorder_files_with_best, save_best_for_id
from src.ui.annotator_view_second import AnnotatorViewSecond
from src.ui.image_quality_panel import ImageQualityPanel
from src.ui.query_state_delegate import QueryStateDelegate, apply_quality_to_combobox
from src.ui.tab_first_order import _MetadataEditPopup
from src.ui.tab_setup import _RenameIdDialog
from src.data.rename_id import rename_id
from src.data.encounter_info import get_encounter_date_from_path, format_encounter_date
from src.utils.interaction_logger import get_interaction_logger


class TabGalleryReview(QWidget):
    """
    Gallery Review tab: single-panel viewer for efficiently browsing gallery members.
    
    Features:
      - Gallery ID selector with type-to-search
      - Previous/Next navigation between gallery IDs
      - AnnotatorViewSecond for image viewing with annotation tools
      - Image quality panel
      - Set Best photo functionality
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ilog = get_interaction_logger()

        # Session-based trash for undo support
        self._trash_dir = Path(tempfile.gettempdir()) / "starboard_trash_session"
        self._trash_dir.mkdir(exist_ok=True)
        self._deleted_files: List[Tuple[Path, Path, str]] = []  # (original_path, trash_path, gallery_id)

        # Metadata edit popup tracking
        self._meta_edit_popup: _MetadataEditPopup | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        # ---- Row 1: Gallery selector and navigation ----
        row1 = QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)

        row1.addWidget(QLabel("Gallery:"))
        self.cmb_gallery = QComboBox()
        self.cmb_gallery.setMinimumWidth(280)
        # Make combo editable for type-to-search functionality
        self.cmb_gallery.setEditable(True)
        self.cmb_gallery.setInsertPolicy(QComboBox.NoInsert)
        self.cmb_gallery.completer().setFilterMode(Qt.MatchContains)
        self.cmb_gallery.completer().setCompletionMode(QCompleter.PopupCompletion)
        # Apply quality indicator delegate
        self._gallery_delegate = QueryStateDelegate(self.cmb_gallery, show_quality_symbols=True)
        self.cmb_gallery.setItemDelegate(self._gallery_delegate)
        row1.addWidget(self.cmb_gallery)

        # Gallery navigation buttons
        self.btn_prev_gallery = QPushButton("◀")
        self.btn_prev_gallery.setFixedWidth(28)
        self.btn_prev_gallery.setToolTip("Previous gallery member")
        self.btn_prev_gallery.clicked.connect(self._on_prev_gallery_clicked)
        row1.addWidget(self.btn_prev_gallery)

        self.btn_next_gallery = QPushButton("▶")
        self.btn_next_gallery.setFixedWidth(28)
        self.btn_next_gallery.setToolTip("Next gallery member")
        self.btn_next_gallery.clicked.connect(self._on_next_gallery_clicked)
        row1.addWidget(self.btn_next_gallery)

        row1.addSpacing(16)

        # Open folder button
        self.btn_open_folder = QPushButton("Open Folder")
        self.btn_open_folder.setToolTip("Open gallery folder in file explorer")
        self.btn_open_folder.clicked.connect(self._on_open_folder_clicked)
        row1.addWidget(self.btn_open_folder)

        # Refresh button
        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.setToolTip("Refresh gallery list and images")
        self.btn_refresh.clicked.connect(self._on_refresh_clicked)
        row1.addWidget(self.btn_refresh)

        # Edit Metadata button
        self.btn_edit_metadata = QPushButton("Edit Metadata")
        self.btn_edit_metadata.setToolTip("Edit metadata for this gallery member")
        self.btn_edit_metadata.clicked.connect(self._on_edit_metadata_clicked)
        row1.addWidget(self.btn_edit_metadata)

        # Rename button
        self.btn_rename = QPushButton("Rename")
        self.btn_rename.setToolTip("Rename this gallery ID (folder + metadata)")
        self.btn_rename.clicked.connect(self._on_rename_clicked)
        row1.addWidget(self.btn_rename)

        row1.addStretch(1)
        outer.addLayout(row1)

        # ---- Gallery viewer ----
        gallery_container = QWidget()
        gallery_layout = QVBoxLayout(gallery_container)
        gallery_layout.setContentsMargins(0, 0, 0, 0)
        gallery_layout.setSpacing(4)

        self._gallery_encounter_info = QLabel("")
        self._gallery_encounter_info.setStyleSheet(
            "QLabel { color: #e67e22; font-size: 12px; font-weight: bold; padding: 2px 4px; }"
        )
        self._gallery_encounter_info.setToolTip("Encounter date for the current image")
        gallery_layout.addWidget(self._gallery_encounter_info)

        self.view_gallery = AnnotatorViewSecond(target="Gallery", title="Gallery")
        gallery_layout.addWidget(self.view_gallery, 1)

        # Image quality panel
        self.gallery_quality_panel = ImageQualityPanel(
            parent=gallery_container,
            show_save_button=True,
            compact=True,
            title="",
        )
        self.gallery_quality_panel.set_target("Gallery")
        self.gallery_quality_panel.saved.connect(self._on_gallery_quality_saved)
        gallery_layout.addWidget(self.gallery_quality_panel)

        outer.addWidget(gallery_container, 1)

        # ---- Action buttons row ----
        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)

        # Delete Photo button
        self.btn_delete_photo = QPushButton("Delete Photo")
        self.btn_delete_photo.setToolTip("Delete current photo (can be undone this session)")
        self.btn_delete_photo.clicked.connect(self._on_delete_photo_clicked)

        # Undo Delete button
        self.btn_undo_delete = QPushButton("Undo Delete")
        self.btn_undo_delete.setToolTip("No deleted photos to restore")
        self.btn_undo_delete.setEnabled(False)
        self.btn_undo_delete.clicked.connect(self._on_undo_delete_clicked)

        # Set Best button
        self.btn_set_best = QPushButton("Set Best")
        self.btn_set_best.setToolTip("Mark current image as best photo for this gallery member")
        self.btn_set_best.clicked.connect(self._on_set_best_clicked)

        action_row.addStretch(1)
        action_row.addWidget(self.btn_delete_photo)
        action_row.addSpacing(8)
        action_row.addWidget(self.btn_undo_delete)
        action_row.addSpacing(16)
        action_row.addWidget(self.btn_set_best)
        action_row.addStretch(1)
        outer.addLayout(action_row)

        # ---- Signals ----
        self.cmb_gallery.currentIndexChanged.connect(self._on_gallery_changed)
        self.view_gallery.currentImageChanged.connect(self._update_encounter_info)

        # ---- Populate gallery IDs ----
        self._refresh_ids()

    def _refresh_ids(self) -> None:
        """Populate the gallery combo box with all gallery IDs."""
        prev_g = self.cmb_gallery.currentText()

        gs = list_ids("Gallery")

        self.cmb_gallery.blockSignals(True)
        try:
            self.cmb_gallery.clear()
            if gs:
                self.cmb_gallery.addItems(gs)
                # Apply quality indicator symbols
                apply_quality_to_combobox(self.cmb_gallery, gs, "Gallery")

            # Try to restore selection; otherwise default to first item
            if gs:
                j = self.cmb_gallery.findText(prev_g) if prev_g else -1
                self.cmb_gallery.setCurrentIndex(j if j >= 0 else 0)
        finally:
            self.cmb_gallery.blockSignals(False)

        # Drive dependent UI
        self._on_gallery_changed()

    def _on_gallery_changed(self) -> None:
        """Handle gallery ID selection change."""
        gid = self.cmb_gallery.currentText()
        self._ilog.log("combo_change", "cmb_gallery_review", value=gid)
        
        files = list_image_files("Gallery", gid) if gid else []
        files = reorder_files_with_best("Gallery", gid, files) if gid else files
        self.view_gallery.set_files(files)

        # Update image quality panel
        if hasattr(self, 'gallery_quality_panel'):
            self.gallery_quality_panel.load_for_id("Gallery", gid)

        self._update_encounter_info()

    def _on_prev_gallery_clicked(self) -> None:
        """Navigate to the previous gallery in the combo box list."""
        self._ilog.log("button_click", "btn_prev_gallery_review", value="clicked")
        current_idx = self.cmb_gallery.currentIndex()
        if current_idx > 0:
            self.cmb_gallery.setCurrentIndex(current_idx - 1)

    def _on_next_gallery_clicked(self) -> None:
        """Navigate to the next gallery in the combo box list."""
        self._ilog.log("button_click", "btn_next_gallery_review", value="clicked")
        current_idx = self.cmb_gallery.currentIndex()
        max_idx = self.cmb_gallery.count() - 1
        if current_idx < max_idx:
            self.cmb_gallery.setCurrentIndex(current_idx + 1)

    def _on_open_folder_clicked(self) -> None:
        """Open the current gallery folder in the system file explorer."""
        gid = self.cmb_gallery.currentText()
        if not gid:
            return
        self._ilog.log("button_click", "btn_open_folder_gallery_review", value=gid)
        folder = ap.root_for("Gallery") / gid
        try:
            if platform.system() == "Windows":
                os.startfile(str(folder))  # type: ignore[attr-defined]
            elif platform.system() == "Darwin":
                subprocess.call(["open", str(folder)])
            else:
                subprocess.call(["xdg-open", str(folder)])
        except Exception:
            pass

    def _on_refresh_clicked(self) -> None:
        """Refresh the gallery list and current view."""
        self._ilog.log("button_click", "btn_refresh_gallery_review", value="clicked")
        # Invalidate image cache to pick up any external changes
        invalidate_image_cache()
        # Refresh the gallery ID list and reload current view
        self._refresh_ids()

    def _on_set_best_clicked(self) -> None:
        """Mark the current image as the best photo for this gallery member."""
        gid = self.cmb_gallery.currentText()
        if not gid or not self.view_gallery.strip.files:
            return
        idx = max(0, min(self.view_gallery.strip.idx, len(self.view_gallery.strip.files) - 1))
        self._ilog.log("button_click", "btn_set_best_gallery_review", value=gid,
                       context={"image_idx": idx})
        save_best_for_id("Gallery", gid, self.view_gallery.strip.files[idx])
        # Reload with best rolled to front
        files = reorder_files_with_best("Gallery", gid, list(self.view_gallery.strip.files))
        self.view_gallery.set_files(files)

    def _on_gallery_quality_saved(self, target: str, id_value: str) -> None:
        """Handle image quality saved for gallery."""
        # Refresh quality indicators in combo box
        self._refresh_ids()

    def _on_delete_photo_clicked(self) -> None:
        """Delete the current photo with confirmation dialog."""
        gid = self.cmb_gallery.currentText()
        if not gid or not self.view_gallery.strip.files:
            return

        # Get current image
        idx = max(0, min(self.view_gallery.strip.idx, len(self.view_gallery.strip.files) - 1))
        current_file = self.view_gallery.strip.files[idx]

        # Show confirmation dialog
        reply = QMessageBox.question(
            self,
            "Delete Photo",
            f"Delete this photo?\n\n{current_file.name}\n\nThis can be undone during this session.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # Generate unique trash filename to avoid collisions
        import uuid
        trash_name = f"{uuid.uuid4().hex}_{current_file.name}"
        trash_path = self._trash_dir / trash_name

        try:
            # Move file to trash
            shutil.move(str(current_file), str(trash_path))

            # Track for undo
            self._deleted_files.append((current_file, trash_path, gid))

            # Log the action
            self._ilog.log("button_click", "btn_delete_photo_gallery_review", value=gid,
                           context={"file": str(current_file), "image_idx": idx})

            # Update undo button
            self._update_undo_button_state()

            # Invalidate cache and refresh viewer
            invalidate_image_cache("Gallery", gid)
            self._refresh_gallery_view(gid, idx)

        except Exception as e:
            QMessageBox.warning(
                self,
                "Delete Failed",
                f"Could not delete photo:\n{e}"
            )

    def _on_undo_delete_clicked(self) -> None:
        """Undo the most recent photo deletion."""
        if not self._deleted_files:
            return

        # Pop last deleted file
        original_path, trash_path, gid = self._deleted_files.pop()

        try:
            # Ensure parent directory exists (in case it was deleted)
            original_path.parent.mkdir(parents=True, exist_ok=True)

            # Move file back from trash
            shutil.move(str(trash_path), str(original_path))

            # Log the action
            self._ilog.log("button_click", "btn_undo_delete_gallery_review", value=gid,
                           context={"file": str(original_path)})

            # Update undo button
            self._update_undo_button_state()

            # Invalidate cache and refresh viewer
            invalidate_image_cache("Gallery", gid)

            # If we're currently viewing the same gallery, refresh it
            current_gid = self.cmb_gallery.currentText()
            if current_gid == gid:
                self._refresh_gallery_view(gid)

        except Exception as e:
            # Put it back on the stack if restore failed
            self._deleted_files.append((original_path, trash_path, gid))
            QMessageBox.warning(
                self,
                "Undo Failed",
                f"Could not restore photo:\n{e}"
            )

    def _update_undo_button_state(self) -> None:
        """Update the undo button enabled state and tooltip."""
        if self._deleted_files:
            self.btn_undo_delete.setEnabled(True)
            last_file = self._deleted_files[-1][0]
            self.btn_undo_delete.setToolTip(f"Restore: {last_file.name}")
        else:
            self.btn_undo_delete.setEnabled(False)
            self.btn_undo_delete.setToolTip("No deleted photos to restore")

    def _refresh_gallery_view(self, gid: str, deleted_idx: int = None) -> None:
        """Refresh the gallery view after delete/undo."""
        files = list_image_files("Gallery", gid) if gid else []
        files = reorder_files_with_best("Gallery", gid, files) if gid else files

        # Determine which index to show after refresh
        if deleted_idx is not None and files:
            # After delete: stay at same index position, or go to last if we deleted the last
            new_idx = min(deleted_idx, len(files) - 1)
            self.view_gallery.set_files(files)
            if new_idx >= 0 and new_idx < len(files):
                self.view_gallery.strip.idx = new_idx
                self.view_gallery.strip._show_current(reset_view=True)
        else:
            self.view_gallery.set_files(files)

        self._update_encounter_info()

    def _update_encounter_info(self, *_) -> None:
        """Update the encounter date label for the currently displayed image."""
        try:
            current_path = self.view_gallery.current_path()
            gid = self.cmb_gallery.currentText() or ""
            if not current_path or not gid:
                self._gallery_encounter_info.setText("")
                return
            enc_date = get_encounter_date_from_path(current_path)
            date_str = format_encounter_date(enc_date) if enc_date else ""
            self._gallery_encounter_info.setText(date_str)
        except Exception:
            self._gallery_encounter_info.setText("")

    def _on_edit_metadata_clicked(self) -> None:
        """Open metadata edit popup for the current gallery member."""
        gid = self.cmb_gallery.currentText()
        if not gid:
            return
        self._ilog.log("button_click", "btn_edit_metadata_gallery_review", value=gid)

        # Close existing popup if open
        if self._meta_edit_popup is not None:
            self._meta_edit_popup.close()

        # Create and show new popup
        self._meta_edit_popup = _MetadataEditPopup("Gallery", gid, self)
        self._meta_edit_popup.saved.connect(self._on_metadata_saved)
        self._meta_edit_popup.show()

    def _on_metadata_saved(self) -> None:
        """Handle metadata saved - refresh the view."""
        self._on_refresh_clicked()

    def _on_rename_clicked(self) -> None:
        """Open rename dialog for the current gallery ID."""
        old_id = self.cmb_gallery.currentText()
        if not old_id:
            return
        self._ilog.log("button_click", "btn_rename_gallery_review", value=old_id)

        # Show rename dialog
        dlg = _RenameIdDialog("Gallery", old_id, self)
        if dlg.exec() != QDialog.Accepted:
            return

        new_id = dlg.get_new_id()
        if not new_id or new_id == old_id:
            return

        # Perform the rename
        report = rename_id("Gallery", old_id, new_id)

        if report.errors:
            QMessageBox.warning(
                self,
                "Rename Errors",
                f"Rename completed with errors:\n\n" + "\n".join(report.errors)
            )
        else:
            QMessageBox.information(
                self,
                "Renamed",
                f"Renamed gallery '{old_id}' to '{new_id}'."
            )

        # Refresh and select the new ID
        invalidate_image_cache()
        self._refresh_ids()
        idx = self.cmb_gallery.findText(new_id)
        if idx >= 0:
            self.cmb_gallery.setCurrentIndex(idx)
