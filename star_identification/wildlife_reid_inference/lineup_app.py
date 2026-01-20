"""
Sunflower Star Police Lineup Application
A PySide6 application for visual identification of sunflower stars
Updated with thread safety fixes to prevent crashes on multiple image loads
AND outing-based filtering for selective search
"""
# === CRITICAL FIXES - Must be at the very top before any imports ===
import os
import typing

# Disable PyTorch Dynamo to avoid typing.Self issues with Python <3.11
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['PYTORCH_DISABLE_LIBRARY_INIT'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Patch typing.Self for Python <3.11
if not hasattr(typing, 'Self'):
    try:
        from typing_extensions import Self

        typing.Self = Self
    except ImportError:
        # Create a minimal Self implementation
        typing.Self = type('Self', (), {})
# === END CRITICAL FIXES ===

# Now continue with normal imports
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import threading
import time

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QGridLayout, QFileDialog,
    QMessageBox, QProgressBar, QGroupBox, QScrollArea,
    QFrame, QSizePolicy, QToolTip, QCheckBox, QListWidget,
    QListWidgetItem, QAbstractItemView
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QRect, QSize, QMutex, QMutexLocker
from PySide6.QtGui import QPixmap, QImage, QPainter, QFont, QColor, QPalette

from PIL import Image
import torch

# Import our minimal inference module
# Handle both package and direct script imports
try:
    from .inference import WildlifeReIDInference
except ImportError:
    from inference import WildlifeReIDInference


class ThreadSafeFeatureExtractor(QThread):
    """Thread-safe background feature extraction"""
    progress = Signal(int)
    status = Signal(str)
    finished = Signal(np.ndarray, object)  # embedding, preprocessed_image
    error = Signal(str)

    def __init__(self, reid_model, image_path):
        super().__init__()
        self.reid_model = reid_model
        self.image_path = image_path
        self.mutex = QMutex()
        self._is_cancelled = False

    def cancel(self):
        """Cancel the thread operation"""
        self._is_cancelled = True

    def run(self):
        try:
            # Check if cancelled
            if self._is_cancelled:
                return

            # Ensure thread safety
            with QMutexLocker(self.mutex):
                if self._is_cancelled:
                    return

                self.status.emit("Loading image...")
                self.progress.emit(20)

                # Extract embedding with thread-safe model
                self.status.emit("Extracting features...")
                self.progress.emit(50)

                # Use batch size of 1 for thread safety
                embedding = self.reid_model.embed_images(
                    self.image_path,
                    batch_size=1,
                    preprocess=True
                )

                if self._is_cancelled:
                    return

                self.progress.emit(80)

                # Get preprocessed image if available
                preprocessed = None
                if self.reid_model.preprocessor and not self._is_cancelled:
                    try:
                        preprocessed = self.reid_model.preprocessor.process_image(self.image_path)
                    except Exception as e:
                        print(f"Warning: Could not get preprocessed image: {e}")

                self.progress.emit(100)

                if not self._is_cancelled:
                    self.finished.emit(embedding, preprocessed)

        except Exception as e:
            if not self._is_cancelled:
                self.error.emit(f"Feature extraction failed: {str(e)}")


class LineupCard(QFrame):
    """Custom widget for displaying a match in the lineup"""

    clicked = Signal(dict)

    def __init__(self, rank: int, score: float, image_path: str,
                 identity: str, metadata: dict = None):
        super().__init__()
        self.rank = rank
        self.score = score
        self.image_path = image_path
        self.identity = identity
        self.metadata = metadata or {}

        self.setFrameStyle(QFrame.Box)
        self.setLineWidth(2)
        self.setFixedSize(150, 200)
        self.setCursor(Qt.PointingHandCursor)

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        # Rank label
        rank_label = QLabel(f"#{self.rank}")
        rank_label.setAlignment(Qt.AlignCenter)
        rank_font = QFont()
        rank_font.setBold(True)
        rank_font.setPointSize(12)
        rank_label.setFont(rank_font)

        # Image
        self.image_label = QLabel()
        self.image_label.setFixedSize(140, 140)
        self.image_label.setScaledContents(True)
        self.image_label.setStyleSheet("border: 1px solid #ccc;")

        # Load thumbnail with error handling
        try:
            pixmap = QPixmap(self.image_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(140, 140, Qt.KeepAspectRatio,
                                              Qt.SmoothTransformation)
                self.image_label.setPixmap(scaled_pixmap)
            else:
                self.image_label.setText("No Image")
                self.image_label.setAlignment(Qt.AlignCenter)
        except Exception as e:
            print(f"Warning: Could not load image {self.image_path}: {e}")
            self.image_label.setText("Error")
            self.image_label.setAlignment(Qt.AlignCenter)

        # Score label (as percentage)
        score_label = QLabel(f"{self.score * 100:.1f}%")
        score_label.setAlignment(Qt.AlignCenter)
        score_font = QFont()
        score_font.setBold(True)
        score_font.setPointSize(10)
        score_label.setFont(score_font)

        # Color code based on score
        if self.score >= 0.9:
            score_label.setStyleSheet("color: green;")
        elif self.score >= 0.8:
            score_label.setStyleSheet("color: orange;")
        else:
            score_label.setStyleSheet("color: red;")

        # Identity label
        id_label = QLabel(self.identity[:10])  # Truncate long IDs
        id_label.setAlignment(Qt.AlignCenter)
        id_label.setToolTip(self.identity)  # Full ID on hover

        layout.addWidget(rank_label)
        layout.addWidget(self.image_label)
        layout.addWidget(score_label)
        layout.addWidget(id_label)

        self.setLayout(layout)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit({
                'rank': self.rank,
                'score': self.score,
                'image_path': self.image_path,
                'identity': self.identity,
                'metadata': self.metadata
            })

    def enterEvent(self, event):
        self.setStyleSheet("background-color: #f0f0f0;")

        # Show tooltip with metadata
        if self.metadata:
            tooltip_text = f"Identity: {self.identity}\n"
            tooltip_text += f"Score: {self.score * 100:.2f}%\n"
            if 'outing' in self.metadata:
                tooltip_text += f"Outing: {self.metadata['outing']}\n"
            for key, value in self.metadata.items():
                if key != 'outing':  # Already shown above
                    tooltip_text += f"{key}: {value}\n"
            QToolTip.showText(event.globalPos(), tooltip_text)

    def leaveEvent(self, event):
        self.setStyleSheet("")


class SunflowerLineupApp(QMainWindow):
    """Main application window with thread safety fixes and outing filtering"""

    def __init__(self, reid_checkpoint: str, yolo_checkpoint: str,
                 database_path: str):
        super().__init__()

        # Initialize variables
        self.reid_model = None
        self.database = None
        self.current_query_embedding = None
        self.current_results = []
        self.extractor_thread = None
        self._init_lock = threading.Lock()
        self._loading_image = False
        self.available_outings = []  # List of available outings
        self.selected_outings = set()  # Currently selected outings

        # Setup
        self._init_models(reid_checkpoint, yolo_checkpoint)
        self._load_database(database_path)
        self._init_ui()

    def _init_models(self, reid_checkpoint: str, yolo_checkpoint: str):
        """Initialize ReID and YOLO models with thread safety"""
        try:
            with self._init_lock:
                # Initialize ReID model
                self.reid_model = WildlifeReIDInference(reid_checkpoint, device='cuda')

                # Set YOLO preprocessor
                self.reid_model.set_preprocessor(yolo_checkpoint, confidence=0.7)

                print("Models initialized successfully")

        except Exception as e:
            error_msg = f"Failed to initialize models: {str(e)}"
            print(error_msg)
            QMessageBox.critical(self, "Error", error_msg)
            sys.exit(1)

    def _load_database(self, database_path: str):
        """Load embedding database and extract outing information"""
        try:
            self.database = np.load(database_path, allow_pickle=True)

            # Validate database structure
            required_keys = ['embeddings', 'image_paths', 'identities']
            for key in required_keys:
                if key not in self.database:
                    raise ValueError(f"Database missing required key: {key}")

            print(f"Loaded database with {len(self.database['embeddings'])} embeddings")

            # Extract available outings
            if 'outings' in self.database:
                self.available_outings = sorted(list(set(self.database['outings'])))
                print(f"Found {len(self.available_outings)} unique outings")
            else:
                print("Warning: No outing information found in database")
                self.available_outings = []

        except Exception as e:
            error_msg = f"Failed to load database: {str(e)}"
            print(error_msg)
            QMessageBox.critical(self, "Error", error_msg)
            sys.exit(1)

    def _init_ui(self):
        """Initialize user interface with outing selection"""
        self.setWindowTitle("🌟 Sunflower Star Police Lineup")
        self.setGeometry(100, 100, 1400, 900)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Left panel - Outing selection
        left_panel = QWidget()
        left_panel.setMaximumWidth(250)
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)

        # Outing selection group
        outing_group = QGroupBox("Select Outings")
        outing_layout = QVBoxLayout()

        # Buttons for select all/none
        button_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.select_all_outings)
        self.select_none_btn = QPushButton("Select None")
        self.select_none_btn.clicked.connect(self.select_no_outings)
        button_layout.addWidget(self.select_all_btn)
        button_layout.addWidget(self.select_none_btn)
        outing_layout.addLayout(button_layout)

        # Outing list widget
        self.outing_list = QListWidget()
        self.outing_list.setSelectionMode(QAbstractItemView.MultiSelection)

        # Populate outing list
        for outing in self.available_outings:
            item = QListWidgetItem(outing)
            self.outing_list.addItem(item)
            # Select all by default
            item.setSelected(True)
            self.selected_outings.add(outing)

        # Connect selection change
        self.outing_list.itemSelectionChanged.connect(self.on_outing_selection_changed)

        outing_layout.addWidget(self.outing_list)

        # Selected count label
        self.selected_count_label = QLabel()
        self.update_selected_count()
        outing_layout.addWidget(self.selected_count_label)

        outing_group.setLayout(outing_layout)
        left_layout.addWidget(outing_group)

        # Right panel - Original UI
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)

        # Query section
        query_group = QGroupBox("Query Image")
        query_layout = QHBoxLayout()

        # Original image
        original_frame = QFrame()
        original_frame.setFrameStyle(QFrame.Box)
        original_layout = QVBoxLayout()
        original_layout.addWidget(QLabel("Original"))
        self.original_label = QLabel()
        self.original_label.setFixedSize(300, 300)
        self.original_label.setScaledContents(True)
        self.original_label.setStyleSheet("border: 1px solid #ccc;")
        self.original_label.setText("No image loaded")
        self.original_label.setAlignment(Qt.AlignCenter)
        original_layout.addWidget(self.original_label)
        original_frame.setLayout(original_layout)

        # Processed image
        processed_frame = QFrame()
        processed_frame.setFrameStyle(QFrame.Box)
        processed_layout = QVBoxLayout()
        processed_layout.addWidget(QLabel("YOLO Detection"))
        self.processed_label = QLabel()
        self.processed_label.setFixedSize(300, 300)
        self.processed_label.setScaledContents(True)
        self.processed_label.setStyleSheet("border: 1px solid #ccc;")
        self.processed_label.setText("No detection")
        self.processed_label.setAlignment(Qt.AlignCenter)
        processed_layout.addWidget(self.processed_label)
        processed_frame.setLayout(processed_layout)

        # Buttons
        button_layout = QVBoxLayout()
        button_layout.addStretch()

        self.load_button = QPushButton("📁 Load Image")
        self.load_button.setFixedSize(150, 40)
        self.load_button.clicked.connect(self.load_image)

        self.search_button = QPushButton("🔍 Search")
        self.search_button.setFixedSize(150, 40)
        self.search_button.clicked.connect(self.perform_search)
        self.search_button.setEnabled(False)
        self.search_button.setStyleSheet("""
            QPushButton:enabled {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
            }
        """)

        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.search_button)
        button_layout.addStretch()

        query_layout.addWidget(original_frame)
        query_layout.addWidget(processed_frame)
        query_layout.addLayout(button_layout)
        query_group.setLayout(query_layout)

        # Controls section
        controls_layout = QHBoxLayout()

        # Threshold slider
        threshold_label = QLabel("Similarity Threshold:")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(70)
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.setTickInterval(10)
        self.threshold_slider.valueChanged.connect(self.update_threshold)

        self.threshold_value_label = QLabel("70%")
        self.threshold_value_label.setMinimumWidth(40)

        # Exclude same folder checkbox
        self.exclude_same_folder_checkbox = QCheckBox("Exclude same folder")
        self.exclude_same_folder_checkbox.setChecked(True)
        self.exclude_same_folder_checkbox.setToolTip(
            "Exclude images from the same folder as the query image"
        )
        # Re-run search when checkbox changes (if we have results)
        self.exclude_same_folder_checkbox.stateChanged.connect(self.on_exclude_folder_changed)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        controls_layout.addWidget(threshold_label)
        controls_layout.addWidget(self.threshold_slider)
        controls_layout.addWidget(self.threshold_value_label)
        controls_layout.addSpacing(20)
        controls_layout.addWidget(self.exclude_same_folder_checkbox)
        controls_layout.addStretch()
        controls_layout.addWidget(self.progress_bar)

        # Results section
        results_group = QGroupBox("Lineup Results")
        results_layout = QVBoxLayout()

        # Scroll area for lineup
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(250)

        self.lineup_widget = QWidget()
        self.lineup_layout = QHBoxLayout()
        self.lineup_layout.setAlignment(Qt.AlignLeft)
        self.lineup_widget.setLayout(self.lineup_layout)

        scroll_area.setWidget(self.lineup_widget)
        results_layout.addWidget(scroll_area)
        results_group.setLayout(results_layout)

        # Status bar
        self.status_label = QLabel("Ready to load query image")
        self.status_label.setStyleSheet("padding: 5px;")

        # Add all to right layout
        right_layout.addWidget(query_group)
        right_layout.addLayout(controls_layout)
        right_layout.addWidget(results_group)
        right_layout.addWidget(self.status_label)

        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

        # Style
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                margin-top: 10px;
            }
            QGroupBox::title {
                padding: 0 5px;
            }
        """)

    def select_all_outings(self):
        """Select all outings"""
        for i in range(self.outing_list.count()):
            self.outing_list.item(i).setSelected(True)

    def select_no_outings(self):
        """Deselect all outings"""
        for i in range(self.outing_list.count()):
            self.outing_list.item(i).setSelected(False)

    def on_outing_selection_changed(self):
        """Handle outing selection change"""
        # Update selected outings set
        self.selected_outings.clear()
        for item in self.outing_list.selectedItems():
            self.selected_outings.add(item.text())

        self.update_selected_count()

        # Re-run search if we have results
        if self.current_query_embedding is not None:
            self.perform_search()

    def update_selected_count(self):
        """Update the selected count label"""
        count = len(self.selected_outings)
        total = len(self.available_outings)
        self.selected_count_label.setText(f"Selected: {count}/{total} outings")

    def _cleanup_thread(self):
        """Properly clean up the extractor thread"""
        if self.extractor_thread:
            # Disconnect all signals first
            try:
                self.extractor_thread.progress.disconnect()
                self.extractor_thread.status.disconnect()
                self.extractor_thread.finished.disconnect()
                self.extractor_thread.error.disconnect()
            except:
                pass  # Signals might already be disconnected

            # Cancel and wait for thread
            if self.extractor_thread.isRunning():
                self.extractor_thread.cancel()
                self.extractor_thread.quit()
                # Wait up to 2 seconds for thread to finish
                if not self.extractor_thread.wait(2000):
                    # Force terminate if still running
                    self.extractor_thread.terminate()
                    self.extractor_thread.wait()

            # Delete the thread object
            self.extractor_thread.deleteLater()
            self.extractor_thread = None

            # Small delay to ensure cleanup
            QApplication.processEvents()

    def load_image(self):
        """Load query image with improved thread safety"""
        # Prevent multiple simultaneous loads
        if self._loading_image:
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Query Image",
            "",
            "Images (*.png *.jpg *.jpeg *.JPG *.JPEG)"
        )

        if file_path:
            self._loading_image = True
            self.load_button.setEnabled(False)
            self.current_image_path = file_path

            # Display original image
            try:
                pixmap = QPixmap(file_path)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio,
                                                  Qt.SmoothTransformation)
                    self.original_label.setPixmap(scaled_pixmap)
                else:
                    QMessageBox.warning(self, "Error", "Failed to load image")
                    self._loading_image = False
                    self.load_button.setEnabled(True)
                    return
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load image: {e}")
                self._loading_image = False
                self.load_button.setEnabled(True)
                return

            # Clear previous results
            self.clear_lineup()
            self.processed_label.setText("Processing...")
            self.search_button.setEnabled(False)

            # Clean up previous thread completely
            self._cleanup_thread()

            # Start feature extraction
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            # Create new thread
            self.extractor_thread = ThreadSafeFeatureExtractor(self.reid_model, file_path)

            # Connect signals
            self.extractor_thread.progress.connect(self.progress_bar.setValue)
            self.extractor_thread.status.connect(self.status_label.setText)
            self.extractor_thread.finished.connect(self.on_extraction_finished)
            self.extractor_thread.error.connect(self.on_extraction_error)

            # Start thread
            self.extractor_thread.start()

            self.status_label.setText(f"Processing: {Path(file_path).name}")

    def on_extraction_finished(self, embedding: np.ndarray,
                               preprocessed_image: Optional[Image.Image]):
        """Handle completed feature extraction"""
        self.current_query_embedding = embedding

        # Display preprocessed image
        if preprocessed_image:
            try:
                # Convert PIL to QPixmap
                preprocessed_image = preprocessed_image.convert("RGB")
                data = preprocessed_image.tobytes("raw", "RGB")
                qimage = QImage(data, preprocessed_image.width,
                                preprocessed_image.height, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimage)
                scaled_pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio,
                                              Qt.SmoothTransformation)
                self.processed_label.setPixmap(scaled_pixmap)
            except Exception as e:
                print(f"Warning: Could not display preprocessed image: {e}")
                self.processed_label.setText("Detection failed")
        else:
            self.processed_label.setText("No detection\n(using full image)")

        self.progress_bar.setVisible(False)
        self.search_button.setEnabled(True)
        self.status_label.setText("Ready to search")
        self._loading_image = False
        self.load_button.setEnabled(True)

    def on_extraction_error(self, error_msg: str):
        """Handle extraction error"""
        self.progress_bar.setVisible(False)
        QMessageBox.warning(self, "Extraction Error", f"Failed to process image: {error_msg}")
        self.status_label.setText("Error processing image")
        self.search_button.setEnabled(False)
        self._loading_image = False
        self.load_button.setEnabled(True)

    def perform_search(self):
        """Search for similar images in database with outing filtering"""
        if self.current_query_embedding is None:
            return

        # Check if any outings are selected
        if not self.selected_outings:
            QMessageBox.warning(self, "No Outings Selected",
                              "Please select at least one outing to search in.")
            return

        try:
            # Get the folder of the query image
            query_folder = str(Path(self.current_image_path).parent)
            exclude_same_folder = self.exclude_same_folder_checkbox.isChecked()

            if exclude_same_folder:
                print(f"Query image folder: {query_folder}")

            # Get indices of images from selected outings
            outing_indices = []
            if 'outings' in self.database:
                for i, outing in enumerate(self.database['outings']):
                    if outing in self.selected_outings:
                        outing_indices.append(i)
            else:
                # If no outing info, use all indices
                outing_indices = list(range(len(self.database['embeddings'])))

            if not outing_indices:
                QMessageBox.warning(self, "No Images Found",
                                  "No images found in the selected outings.")
                return

            print(f"Searching in {len(outing_indices)} images from {len(self.selected_outings)} selected outings")

            # Filter embeddings to selected outings
            filtered_embeddings = self.database['embeddings'][outing_indices]

            # Compute similarities only for filtered embeddings
            similarities = self.reid_model.compute_similarity(
                self.current_query_embedding,
                filtered_embeddings
            )

            # Get all results sorted by similarity
            similarities = similarities[0]  # First (only) query
            sorted_indices_filtered = np.argsort(similarities)[::-1]

            # Map back to original indices
            sorted_indices = [outing_indices[i] for i in sorted_indices_filtered]

            # Extract metadata properly
            metadata_dict = {}
            if 'metadata' in self.database:
                # When saved with np.savez, dicts are stored as 0-d arrays
                metadata_obj = self.database['metadata']
                if hasattr(metadata_obj, 'item'):
                    # Extract the dictionary from the numpy array
                    metadata_dict = metadata_obj.item()
                elif isinstance(metadata_obj, dict):
                    metadata_dict = metadata_obj

            # Store results, optionally filtering out same folder
            self.current_results = []
            excluded_count = 0
            excluded_outing_count = 0

            for idx, filtered_idx in zip(sorted_indices, sorted_indices_filtered):
                db_image_path = str(self.database['image_paths'][idx])

                # Skip if from same folder (when option is enabled)
                if exclude_same_folder:
                    db_folder = str(Path(db_image_path).parent)
                    if db_folder == query_folder:
                        excluded_count += 1
                        continue

                self.current_results.append({
                    'score': similarities[filtered_idx],
                    'image_path': db_image_path,
                    'identity': str(self.database['identities'][idx]),
                    'metadata': metadata_dict.get(int(idx), {})
                })

            # Display results
            self.update_lineup()

            # Update status with exclusion info
            status_msg = f"Found {len(self.current_results)} matches"
            status_msg += f" from {len(self.selected_outings)} selected outings"
            if exclude_same_folder and excluded_count > 0:
                status_msg += f" (excluded {excluded_count} from same folder)"
            self.status_label.setText(status_msg)

        except Exception as e:
            QMessageBox.critical(self, "Search Error", f"Search failed: {str(e)}")
            self.status_label.setText("Search failed")

    def update_lineup(self):
        """Update the lineup display based on current threshold"""
        self.clear_lineup()

        if not self.current_results:
            return

        # Get threshold
        threshold = self.threshold_slider.value() / 100.0

        # Filter and display results
        rank = 1
        for result in self.current_results[:20]:  # Max 20 in lineup
            if result['score'] >= threshold:
                card = LineupCard(
                    rank=rank,
                    score=result['score'],
                    image_path=result['image_path'],
                    identity=result['identity'],
                    metadata=result['metadata']
                )
                card.clicked.connect(self.show_match_details)
                self.lineup_layout.addWidget(card)
                rank += 1

        # Add stretch at end
        self.lineup_layout.addStretch()

        # Update status
        matches_shown = rank - 1
        self.status_label.setText(f"Showing {matches_shown} matches above {threshold:.0%} threshold")

    def clear_lineup(self):
        """Clear the lineup display"""
        while self.lineup_layout.count():
            child = self.lineup_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def update_threshold(self, value):
        """Handle threshold slider change"""
        self.threshold_value_label.setText(f"{value}%")
        if self.current_results:
            self.update_lineup()

    def on_exclude_folder_changed(self, state):
        """Handle exclude folder checkbox change"""
        # Re-run search if we have a query embedding
        if self.current_query_embedding is not None:
            self.perform_search()

    def show_match_details(self, match_info: dict):
        """Show detailed information about a match"""
        details = f"Match Details\n"
        details += f"{'=' * 30}\n"
        details += f"Rank: #{match_info['rank']}\n"
        details += f"Identity: {match_info['identity']}\n"
        details += f"Similarity: {match_info['score'] * 100:.2f}%\n"
        details += f"Image: {Path(match_info['image_path']).name}\n"

        if match_info['metadata']:
            details += f"\nMetadata:\n"
            for key, value in match_info['metadata'].items():
                details += f"  {key}: {value}\n"

        QMessageBox.information(self, "Match Details", details)

    def closeEvent(self, event):
        """Clean up when closing"""
        # Clean up thread properly
        self._cleanup_thread()
        event.accept()


def main():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Sunflower Star Police Lineup')
    parser.add_argument('--reid-checkpoint', type=str, required=True,
                        help='Path to ReID model checkpoint')
    parser.add_argument('--yolo-checkpoint', type=str, required=True,
                        help='Path to YOLO model checkpoint')
    parser.add_argument('--database', type=str, required=True,
                        help='Path to embedding database (.npz file)')

    args = parser.parse_args()

    # Create application
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    # Create and show main window
    window = SunflowerLineupApp(
        reid_checkpoint=args.reid_checkpoint,
        yolo_checkpoint=args.yolo_checkpoint,
        database_path=args.database
    )
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()