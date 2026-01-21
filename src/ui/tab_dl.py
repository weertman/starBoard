# src/ui/tab_dl.py
"""
Deep Learning tab for starBoard.

Provides:
- Model status and management
- Precomputation controls
- Fine-tuning interface (advanced)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton,
    QComboBox, QSpinBox, QCheckBox, QProgressBar, QFileDialog,
    QMessageBox, QScrollArea, QSizePolicy, QListWidget, QListWidgetItem,
    QFrame, QDialog, QDialogButtonBox, QFormLayout
)

from src.ui.collapsible import CollapsibleSection
from src.utils.interaction_logger import get_interaction_logger

log = logging.getLogger("starBoard.ui.tab_dl")


# Default t-SNE configuration
DEFAULT_TSNE_CONFIG = {
    "perplexity": 30,
    "max_iter": 1000,
    "learning_rate": "auto",
    "init": "pca",
    "random_state": 42,
}


class TSNEConfigDialog(QDialog):
    """Dialog for configuring t-SNE visualization parameters."""
    
    def __init__(self, current_config: dict = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("t-SNE Configuration")
        self.setMinimumWidth(320)
        
        # Use current config or defaults
        self._config = dict(current_config) if current_config else dict(DEFAULT_TSNE_CONFIG)
        
        self._build_ui()
    
    def _build_ui(self):
        layout = QVBoxLayout(self)
        
        # Form layout for parameters
        form = QFormLayout()
        
        # Perplexity
        self.spin_perplexity = QSpinBox()
        self.spin_perplexity.setRange(5, 200)
        self.spin_perplexity.setValue(self._config["perplexity"])
        self.spin_perplexity.setToolTip(
            "Controls local vs global structure balance.\n"
            "Lower (5-30): emphasizes local clusters\n"
            "Higher (50-100): reveals global relationships\n"
            "Rule of thumb: ~sqrt(n_samples)"
        )
        form.addRow("Perplexity:", self.spin_perplexity)
        
        # Max iterations
        self.spin_max_iter = QSpinBox()
        self.spin_max_iter.setRange(250, 5000)
        self.spin_max_iter.setSingleStep(250)
        self.spin_max_iter.setValue(self._config["max_iter"])
        self.spin_max_iter.setToolTip(
            "Number of optimization iterations.\n"
            "Higher values may improve convergence but take longer."
        )
        form.addRow("Max iterations:", self.spin_max_iter)
        
        # Learning rate
        self.cmb_learning_rate = QComboBox()
        self.cmb_learning_rate.addItem("auto (recommended)", "auto")
        self.cmb_learning_rate.addItem("50", 50.0)
        self.cmb_learning_rate.addItem("100", 100.0)
        self.cmb_learning_rate.addItem("200", 200.0)
        self.cmb_learning_rate.addItem("500", 500.0)
        self.cmb_learning_rate.addItem("1000", 1000.0)
        # Set current value
        lr = self._config["learning_rate"]
        if lr == "auto":
            self.cmb_learning_rate.setCurrentIndex(0)
        else:
            idx = self.cmb_learning_rate.findData(float(lr))
            if idx >= 0:
                self.cmb_learning_rate.setCurrentIndex(idx)
        self.cmb_learning_rate.setToolTip(
            "Step size for gradient descent.\n"
            "'auto' lets sklearn choose based on data size."
        )
        form.addRow("Learning rate:", self.cmb_learning_rate)
        
        # Initialization
        self.cmb_init = QComboBox()
        self.cmb_init.addItem("PCA (recommended)", "pca")
        self.cmb_init.addItem("Random", "random")
        idx = self.cmb_init.findData(self._config["init"])
        if idx >= 0:
            self.cmb_init.setCurrentIndex(idx)
        self.cmb_init.setToolTip(
            "Initialization method.\n"
            "PCA: faster convergence, more reproducible\n"
            "Random: may find different local optima"
        )
        form.addRow("Initialization:", self.cmb_init)
        
        # Random state
        self.spin_random_state = QSpinBox()
        self.spin_random_state.setRange(0, 99999)
        self.spin_random_state.setValue(self._config["random_state"])
        self.spin_random_state.setToolTip(
            "Random seed for reproducibility.\n"
            "Change to get different layouts."
        )
        form.addRow("Random seed:", self.spin_random_state)
        
        layout.addLayout(form)
        
        # Reset to defaults button
        self.btn_reset = QPushButton("Reset to Defaults")
        self.btn_reset.clicked.connect(self._reset_to_defaults)
        layout.addWidget(self.btn_reset)
        
        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def _reset_to_defaults(self):
        """Reset all fields to default values."""
        self.spin_perplexity.setValue(DEFAULT_TSNE_CONFIG["perplexity"])
        self.spin_max_iter.setValue(DEFAULT_TSNE_CONFIG["max_iter"])
        self.cmb_learning_rate.setCurrentIndex(0)  # auto
        self.cmb_init.setCurrentIndex(0)  # pca
        self.spin_random_state.setValue(DEFAULT_TSNE_CONFIG["random_state"])
    
    def get_config(self) -> dict:
        """Return the current configuration."""
        return {
            "perplexity": self.spin_perplexity.value(),
            "max_iter": self.spin_max_iter.value(),
            "learning_rate": self.cmb_learning_rate.currentData(),
            "init": self.cmb_init.currentData(),
            "random_state": self.spin_random_state.value(),
        }


class TabDeepLearning(QWidget):
    """
    Deep Learning management tab.
    
    Provides interface for:
    - Viewing DL system status
    - Managing registered models
    - Running precomputation
    - Fine-tuning models (advanced)
    """
    
    # Signal emitted when precomputation completes (for other tabs to refresh)
    precomputeCompleted = Signal()
    
    # Signal emitted when verification precomputation completes
    verificationPrecomputeCompleted = Signal()
    
    # Signal emitted when evaluation completes (for first-order tab to enable sorting)
    evaluationCompleted = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("TabDeepLearning")
        
        self._worker = None
        self._tsne_config = dict(DEFAULT_TSNE_CONFIG)  # Store t-SNE configuration
        self._ilog = get_interaction_logger()
        
        # Check if DL is available
        try:
            from src.dl import (
                DL_AVAILABLE, DEVICE, DEVICE_NAME, TORCH_VERSION, 
                get_status_message, VERIFICATION_AVAILABLE
            )
            self._dl_available = DL_AVAILABLE
            self._device = DEVICE
            self._device_name = DEVICE_NAME
            self._torch_version = TORCH_VERSION
            self._status_message = get_status_message()
            self._verification_available = VERIFICATION_AVAILABLE
        except ImportError:
            self._dl_available = False
            self._device = None
            self._device_name = None
            self._torch_version = None
            self._status_message = "DL module not found"
            self._verification_available = False
        
        self._build_ui()
        self._refresh_all()
    
    def _build_ui(self):
        """Build the tab UI."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(12)
        
        # Scroll area for the content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(12)
        
        # Status section
        status_group = self._build_status_group()
        content_layout.addWidget(status_group)
        
        # Model management section
        model_group = self._build_model_group()
        content_layout.addWidget(model_group)
        
        # Precomputation section
        precompute_group = self._build_precompute_group()
        content_layout.addWidget(precompute_group)
        
        # Visualization section
        viz_group = self._build_visualization_group()
        content_layout.addWidget(viz_group)
        
        # Evaluation section (collapsed by default)
        eval_section = CollapsibleSection("Embedding Evaluation", start_collapsed=True)
        eval_group = self._build_evaluation_group()
        eval_section.setContent(eval_group)
        content_layout.addWidget(eval_section)
        
        # Verification Evaluation section (collapsed by default)
        verif_eval_section = CollapsibleSection("Verification Evaluation", start_collapsed=True)
        verif_eval_group = self._build_verification_evaluation_group()
        verif_eval_section.setContent(verif_eval_group)
        content_layout.addWidget(verif_eval_section)
        
        # Fine-tuning section (collapsed by default)
        finetune_section = CollapsibleSection("Fine-Tuning (Advanced)", start_collapsed=True)
        finetune_group = self._build_finetune_group()
        finetune_section.setContent(finetune_group)
        content_layout.addWidget(finetune_section)
        
        content_layout.addStretch(1)
        
        scroll.setWidget(content)
        outer.addWidget(scroll)
    
    def _build_status_group(self) -> QGroupBox:
        """Build the status display group."""
        group = QGroupBox("Status")
        layout = QVBoxLayout(group)
        
        # Device info
        device_row = QHBoxLayout()
        device_row.addWidget(QLabel("<b>Device:</b>"))
        self.lbl_device = QLabel("—")
        device_row.addWidget(self.lbl_device)
        device_row.addStretch(1)
        layout.addLayout(device_row)
        
        # Active model
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("<b>Active Model:</b>"))
        self.lbl_active_model = QLabel("—")
        model_row.addWidget(self.lbl_active_model)
        model_row.addStretch(1)
        layout.addLayout(model_row)
        
        # Precomputation status
        precompute_row = QHBoxLayout()
        precompute_row.addWidget(QLabel("<b>Precomputation:</b>"))
        self.lbl_precompute_status = QLabel("—")
        precompute_row.addWidget(self.lbl_precompute_status)
        precompute_row.addStretch(1)
        layout.addLayout(precompute_row)
        
        # Verification status
        verif_row = QHBoxLayout()
        verif_row.addWidget(QLabel("<b>Verification:</b>"))
        self.lbl_verification_status = QLabel("—")
        verif_row.addWidget(self.lbl_verification_status)
        verif_row.addStretch(1)
        layout.addLayout(verif_row)
        
        # Pending updates
        pending_row = QHBoxLayout()
        pending_row.addWidget(QLabel("<b>Pending updates:</b>"))
        self.lbl_pending = QLabel("0 identities")
        pending_row.addWidget(self.lbl_pending)
        pending_row.addStretch(1)
        layout.addLayout(pending_row)
        
        return group
    
    def _build_model_group(self) -> QGroupBox:
        """Build the model management group."""
        group = QGroupBox("Model Management")
        layout = QVBoxLayout(group)
        
        layout.addWidget(QLabel("Registered Models:"))
        
        # Model list
        self.list_models = QListWidget()
        self.list_models.setMinimumHeight(100)
        self.list_models.setMaximumHeight(150)
        self.list_models.itemSelectionChanged.connect(self._on_model_selection_changed)
        layout.addWidget(self.list_models)
        
        # Buttons row
        btn_row = QHBoxLayout()
        
        self.btn_set_active = QPushButton("Set Active")
        self.btn_set_active.setEnabled(False)
        self.btn_set_active.clicked.connect(self._on_set_active)
        btn_row.addWidget(self.btn_set_active)
        
        self.btn_precompute_model = QPushButton("Precompute")
        self.btn_precompute_model.setEnabled(False)
        self.btn_precompute_model.clicked.connect(self._on_precompute_selected)
        btn_row.addWidget(self.btn_precompute_model)
        
        btn_row.addStretch(1)
        
        self.btn_set_default = QPushButton("Set as Default")
        self.btn_set_default.setToolTip("Set the selected model as the default model")
        self.btn_set_default.setEnabled(False)
        self.btn_set_default.clicked.connect(self._on_set_default_model)
        btn_row.addWidget(self.btn_set_default)
        
        self.btn_import_model = QPushButton("Import Model...")
        self.btn_import_model.clicked.connect(self._on_import_model)
        btn_row.addWidget(self.btn_import_model)
        
        self.btn_remove_model = QPushButton("Remove")
        self.btn_remove_model.setEnabled(False)
        self.btn_remove_model.clicked.connect(self._on_remove_model)
        btn_row.addWidget(self.btn_remove_model)
        
        layout.addLayout(btn_row)
        
        return group
    
    def _build_precompute_group(self) -> QGroupBox:
        """Build the precomputation controls group."""
        group = QGroupBox("Precomputation")
        layout = QVBoxLayout(group)
        
        # Scope selection
        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("Scope:"))
        self.chk_gallery = QCheckBox("Gallery")
        self.chk_gallery.setChecked(True)
        self.chk_gallery.toggled.connect(
            lambda checked: self._ilog.log("checkbox_toggle", "chk_gallery", value=str(checked)))
        scope_row.addWidget(self.chk_gallery)
        self.chk_queries = QCheckBox("Queries")
        self.chk_queries.setChecked(True)
        self.chk_queries.toggled.connect(
            lambda checked: self._ilog.log("checkbox_toggle", "chk_queries", value=str(checked)))
        scope_row.addWidget(self.chk_queries)
        scope_row.addStretch(1)
        layout.addLayout(scope_row)
        
        # Speed mode selector
        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Speed mode:"))
        self.cmb_speed_mode = QComboBox()
        self.cmb_speed_mode.addItem("🔄 Auto (adapts to hardware)", "auto")
        self.cmb_speed_mode.addItem("⚡ Fast (CPU-optimized, less TTA)", "fast")
        self.cmb_speed_mode.addItem("🎯 Quality (full TTA, slower)", "quality")
        self.cmb_speed_mode.setToolTip(
            "Auto: Adapts to GPU/CPU automatically\n"
            "Fast: Skip vertical flip TTA on CPU (~2x faster)\n"
            "Quality: Full TTA (horizontal + vertical flip)"
        )
        self.cmb_speed_mode.currentIndexChanged.connect(self._on_speed_mode_changed)
        speed_row.addWidget(self.cmb_speed_mode)
        speed_row.addStretch(1)
        layout.addLayout(speed_row)
        
        # Options row 1
        options_row = QHBoxLayout()
        self.chk_tta = QCheckBox("TTA (flip augmentation)")
        self.chk_tta.setChecked(True)
        self.chk_tta.setToolTip("Test-Time Augmentation: average embeddings from original and flipped images")
        self.chk_tta.toggled.connect(
            lambda checked: self._ilog.log("checkbox_toggle", "chk_tta", value=str(checked)))
        options_row.addWidget(self.chk_tta)
        
        self.chk_reranking = QCheckBox("k-Reciprocal Re-ranking")
        self.chk_reranking.setChecked(True)
        self.chk_reranking.setToolTip("Apply k-reciprocal re-ranking for improved accuracy")
        self.chk_reranking.toggled.connect(
            lambda checked: self._ilog.log("checkbox_toggle", "chk_reranking", value=str(checked)))
        options_row.addWidget(self.chk_reranking)
        options_row.addStretch(1)
        layout.addLayout(options_row)
        
        # Options row 2 (verification)
        options_row2 = QHBoxLayout()
        self.chk_verification = QCheckBox("Include Verification (best-photo pairwise)")
        self.chk_verification.setChecked(self._verification_available)
        self.chk_verification.setEnabled(self._verification_available)
        self.chk_verification.setToolTip(
            "Compute pairwise verification scores using the cross-attention model.\n"
            "Compares 'best' photo from each query to each gallery identity.\n"
            "Provides P(same individual) scores shown alongside similarity scores."
        )
        self.chk_verification.toggled.connect(
            lambda checked: self._ilog.log("checkbox_toggle", "chk_verification", value=str(checked)))
        options_row2.addWidget(self.chk_verification)
        
        if not self._verification_available:
            lbl_verif_unavail = QLabel("<i>(verification module not available)</i>")
            lbl_verif_unavail.setStyleSheet("color: gray;")
            options_row2.addWidget(lbl_verif_unavail)
        
        options_row2.addStretch(1)
        layout.addLayout(options_row2)
        
        # Batch size
        batch_row = QHBoxLayout()
        batch_row.addWidget(QLabel("Batch size:"))
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 64)
        self.spin_batch.setValue(8)
        self.spin_batch.setToolTip("Number of images to process at once (lower for less memory)")
        self.spin_batch.valueChanged.connect(
            lambda v: self._ilog.log("spin_change", "spin_batch", value=str(v)))
        batch_row.addWidget(self.spin_batch)
        batch_row.addStretch(1)
        layout.addLayout(batch_row)
        
        # Action buttons
        action_row = QHBoxLayout()
        
        self.btn_run_full = QPushButton("▶ Run Full Precomputation")
        self.btn_run_full.clicked.connect(self._on_run_full_precompute)
        action_row.addWidget(self.btn_run_full)
        
        self.btn_update_pending = QPushButton("Update Pending Only")
        self.btn_update_pending.clicked.connect(self._on_update_pending)
        action_row.addWidget(self.btn_update_pending)
        
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel)
        action_row.addWidget(self.btn_cancel)
        
        action_row.addStretch(1)
        layout.addLayout(action_row)
        
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.lbl_progress = QLabel("")
        self.lbl_progress.setVisible(False)
        layout.addWidget(self.lbl_progress)
        
        return group
    
    def _build_visualization_group(self) -> QGroupBox:
        """Build the embedding visualization group."""
        group = QGroupBox("Embedding Visualizations")
        layout = QVBoxLayout(group)
        
        # Description
        desc = QLabel(
            "Visualize the embedding space using t-SNE dimensionality reduction. "
            "Requires precomputation to be completed first."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Buttons row
        btn_row = QHBoxLayout()
        
        self.btn_viz_identity = QPushButton("Centroid-Level t-SNE")
        self.btn_viz_identity.setToolTip("Visualize identity centroids (one point per identity)")
        self.btn_viz_identity.clicked.connect(self._on_viz_identity)
        btn_row.addWidget(self.btn_viz_identity)
        
        self.btn_viz_image = QPushButton("Image-Level t-SNE")
        self.btn_viz_image.setToolTip("Visualize all image embeddings (one point per image)")
        self.btn_viz_image.clicked.connect(self._on_viz_image)
        btn_row.addWidget(self.btn_viz_image)
        
        btn_row.addSpacing(12)
        
        self.btn_tsne_config = QPushButton("Configure t-SNE...")
        self.btn_tsne_config.setToolTip("Adjust t-SNE parameters (perplexity, iterations, etc.)")
        self.btn_tsne_config.clicked.connect(self._on_configure_tsne)
        btn_row.addWidget(self.btn_tsne_config)
        
        btn_row.addStretch(1)
        layout.addLayout(btn_row)
        
        # Status label (shows current config summary)
        self.lbl_viz_status = QLabel(self._get_tsne_config_summary())
        layout.addWidget(self.lbl_viz_status)
        
        return group
    
    def _get_tsne_config_summary(self) -> str:
        """Return a short summary of current t-SNE config."""
        c = self._tsne_config
        return f"t-SNE config: perplexity={c['perplexity']}, iter={c['max_iter']}, lr={c['learning_rate']}, init={c['init']}"
    
    def _on_configure_tsne(self):
        """Open the t-SNE configuration dialog."""
        self._ilog.log("button_click", "btn_configure_tsne", value="clicked")
        dialog = TSNEConfigDialog(self._tsne_config, self)
        if dialog.exec() == QDialog.Accepted:
            self._tsne_config = dialog.get_config()
            self.lbl_viz_status.setText(self._get_tsne_config_summary())
    
    def _on_viz_identity(self):
        """Launch identity-level t-SNE visualization."""
        self._ilog.log("button_click", "btn_viz_identity", value="clicked")
        try:
            self.lbl_viz_status.setText("Loading visualization...")
            self.btn_viz_identity.setEnabled(False)
            self.btn_viz_image.setEnabled(False)
            
            # Import and run
            from src.dl.tmp_tsne_viz_demo import load_embeddings, create_visualization
            
            data = load_embeddings()
            fig = create_visualization(data, tsne_config=self._tsne_config)
            
            self.lbl_viz_status.setText("Opening in browser...")
            fig.show()
            
            self.lbl_viz_status.setText("Visualization opened in browser")
            
        except FileNotFoundError as e:
            QMessageBox.warning(self, "Not Available", 
                               "Embeddings not found. Run precomputation first.")
            self.lbl_viz_status.setText(self._get_tsne_config_summary())
        except Exception as e:
            log.exception("Visualization error")
            QMessageBox.critical(self, "Error", f"Visualization failed: {e}")
            self.lbl_viz_status.setText(self._get_tsne_config_summary())
        finally:
            self.btn_viz_identity.setEnabled(True)
            self.btn_viz_image.setEnabled(True)
    
    def _on_viz_image(self):
        """Launch image-level t-SNE visualization."""
        self._ilog.log("button_click", "btn_viz_image", value="clicked")
        try:
            self.lbl_viz_status.setText("Loading embeddings (this may take a moment)...")
            self.btn_viz_identity.setEnabled(False)
            self.btn_viz_image.setEnabled(False)
            
            # Force UI update
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents()
            
            # Import and run
            from src.dl.tmp_tsne_image_viz_demo import load_image_embeddings, create_visualization
            
            data = load_image_embeddings()
            
            self.lbl_viz_status.setText("Running t-SNE (may take 1-2 minutes)...")
            QApplication.processEvents()
            
            fig = create_visualization(data, tsne_config=self._tsne_config)
            
            self.lbl_viz_status.setText("Opening in browser...")
            fig.show()
            
            self.lbl_viz_status.setText("Visualization opened in browser")
            
        except FileNotFoundError as e:
            QMessageBox.warning(self, "Not Available", 
                               "Per-image embeddings not found. Run precomputation first.")
            self.lbl_viz_status.setText(self._get_tsne_config_summary())
        except Exception as e:
            log.exception("Visualization error")
            QMessageBox.critical(self, "Error", f"Visualization failed: {e}")
            self.lbl_viz_status.setText(self._get_tsne_config_summary())
        finally:
            self.btn_viz_identity.setEnabled(True)
            self.btn_viz_image.setEnabled(True)
    
    def _build_evaluation_group(self) -> QGroupBox:
        """Build the evaluation group for model performance metrics."""
        group = QGroupBox("")
        layout = QVBoxLayout(group)
        
        # Description
        desc = QLabel(
            "Evaluate model performance using your past match annotations. "
            "Requires precomputation to be completed."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Model selector row
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.cmb_eval_model = QComboBox()
        self.cmb_eval_model.setToolTip("Select model to evaluate")
        model_row.addWidget(self.cmb_eval_model, 1)
        model_row.addStretch(1)
        layout.addLayout(model_row)
        
        # Run button and status
        run_row = QHBoxLayout()
        self.btn_run_eval = QPushButton("▶ Run Evaluation")
        self.btn_run_eval.setToolTip("Compute metrics comparing model rankings to your verdicts")
        self.btn_run_eval.clicked.connect(self._on_run_evaluation)
        run_row.addWidget(self.btn_run_eval)
        run_row.addStretch(1)
        layout.addLayout(run_row)
        
        # Results summary
        self.lbl_eval_summary = QLabel("")
        self.lbl_eval_summary.setWordWrap(True)
        layout.addWidget(self.lbl_eval_summary)
        
        # Visualization buttons (initially disabled)
        viz_row = QHBoxLayout()
        
        self.btn_eval_rank_dist = QPushButton("Rank Distribution")
        self.btn_eval_rank_dist.setToolTip("View histogram of where correct matches ranked")
        self.btn_eval_rank_dist.setEnabled(False)
        self.btn_eval_rank_dist.clicked.connect(self._on_show_rank_distribution)
        viz_row.addWidget(self.btn_eval_rank_dist)
        
        self.btn_eval_per_query = QPushButton("Per-Query Results")
        self.btn_eval_per_query.setToolTip("View detailed results for each query")
        self.btn_eval_per_query.setEnabled(False)
        self.btn_eval_per_query.clicked.connect(self._on_show_per_query)
        viz_row.addWidget(self.btn_eval_per_query)
        
        self.btn_eval_gallery = QPushButton("Gallery Stats")
        self.btn_eval_gallery.setToolTip("View performance by gallery individual")
        self.btn_eval_gallery.setEnabled(False)
        self.btn_eval_gallery.clicked.connect(self._on_show_gallery_stats)
        viz_row.addWidget(self.btn_eval_gallery)
        
        self.btn_eval_suggestions = QPushButton("Match Suggestions")
        self.btn_eval_suggestions.setToolTip("View suggested matches for unmatched queries")
        self.btn_eval_suggestions.setEnabled(False)
        self.btn_eval_suggestions.clicked.connect(self._on_show_suggestions)
        viz_row.addWidget(self.btn_eval_suggestions)
        
        viz_row.addStretch(1)
        layout.addLayout(viz_row)
        
        # Store evaluation results for visualization buttons
        self._eval_results = None
        
        # Populate model combo box
        self._refresh_eval_model_combo()
        
        return group
    
    def _refresh_eval_model_combo(self):
        """Refresh the evaluation model combo box."""
        self.cmb_eval_model.clear()
        
        if not self._dl_available:
            self.btn_run_eval.setEnabled(False)
            return
        
        try:
            from src.dl.registry import DLRegistry, DEFAULT_MODEL_KEY
            registry = DLRegistry.load()
            
            default_idx = 0
            for i, (key, entry) in enumerate(registry.models.items()):
                suffix = ""
                if not entry.precomputed:
                    suffix = " (not precomputed)"
                self.cmb_eval_model.addItem(f"{entry.display_name}{suffix}", key)
                if key == DEFAULT_MODEL_KEY:
                    default_idx = i
            
            if self.cmb_eval_model.count() > 0:
                self.cmb_eval_model.setCurrentIndex(default_idx)
                
        except Exception as e:
            log.error("Failed to refresh eval model combo: %s", e)
    
    def _on_run_evaluation(self):
        """Run model evaluation."""
        self._ilog.log("button_click", "btn_run_eval", value="clicked")
        
        model_key = self.cmb_eval_model.currentData()
        if not model_key:
            QMessageBox.warning(self, "No Model", "Please select a model to evaluate.")
            return
        
        try:
            from src.dl.registry import DLRegistry
            registry = DLRegistry.load()
            
            if model_key not in registry.models:
                QMessageBox.warning(self, "Model Not Found", "Selected model not found.")
                return
            
            if not registry.models[model_key].precomputed:
                QMessageBox.warning(
                    self, "Not Precomputed",
                    "This model needs precomputation before evaluation.\n"
                    "Run precomputation first."
                )
                return
            
            # Disable buttons during evaluation
            self.btn_run_eval.setEnabled(False)
            self.lbl_eval_summary.setText("Running evaluation...")
            
            # Force UI update
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents()
            
            # Run evaluation
            from src.dl.evaluation import run_evaluation, save_evaluation
            self._eval_results = run_evaluation(model_key)
            
            # Auto-save evaluation results for use by first-order tab
            if save_evaluation(self._eval_results):
                log.info("Evaluation results saved for model %s", model_key)
            
            # Update summary
            r = self._eval_results
            summary = (
                f"<b>Results for {r.model_name}:</b><br>"
                f"Evaluated: {r.total_evaluated} of {r.total_yes_verdicts} confirmed matches<br>"
                f"Rank@1: <b>{r.rank_at_1:.1%}</b> | "
                f"Rank@5: <b>{r.rank_at_5:.1%}</b> | "
                f"Rank@10: <b>{r.rank_at_10:.1%}</b><br>"
                f"Mean Reciprocal Rank: <b>{r.mrr:.3f}</b><br>"
                f"Match suggestions: {len(r.suggestions)}"
            )
            
            if r.missing_queries:
                summary += f"<br><span style='color: orange;'>⚠ {len(r.missing_queries)} queries not in precomputed data</span>"
            
            self.lbl_eval_summary.setText(summary)
            
            # Enable visualization buttons
            self.btn_eval_rank_dist.setEnabled(True)
            self.btn_eval_per_query.setEnabled(True)
            self.btn_eval_gallery.setEnabled(True)
            self.btn_eval_suggestions.setEnabled(len(r.suggestions) > 0)
            
            # Emit signal so first-order tab can refresh
            self.evaluationCompleted.emit()
            
        except Exception as e:
            log.exception("Evaluation failed")
            QMessageBox.critical(self, "Evaluation Failed", f"Error: {e}")
            self.lbl_eval_summary.setText(f"<span style='color: red;'>Evaluation failed: {e}</span>")
        finally:
            self.btn_run_eval.setEnabled(True)
    
    def _on_show_rank_distribution(self):
        """Show rank distribution visualization."""
        self._ilog.log("button_click", "btn_eval_rank_dist", value="clicked")
        if not self._eval_results:
            return
        
        try:
            from src.dl.evaluation import create_rank_distribution_figure
            fig = create_rank_distribution_figure(self._eval_results)
            fig.show()
        except Exception as e:
            log.exception("Failed to show rank distribution")
            QMessageBox.critical(self, "Error", f"Failed to create visualization: {e}")
    
    def _on_show_per_query(self):
        """Show per-query results visualization."""
        self._ilog.log("button_click", "btn_eval_per_query", value="clicked")
        if not self._eval_results:
            return
        
        try:
            from src.dl.evaluation import create_per_query_figure
            fig = create_per_query_figure(self._eval_results)
            fig.show()
        except Exception as e:
            log.exception("Failed to show per-query results")
            QMessageBox.critical(self, "Error", f"Failed to create visualization: {e}")
    
    def _on_show_gallery_stats(self):
        """Show gallery statistics visualization."""
        self._ilog.log("button_click", "btn_eval_gallery_stats", value="clicked")
        if not self._eval_results:
            return
        
        try:
            from src.dl.evaluation import create_gallery_stats_figure
            fig = create_gallery_stats_figure(self._eval_results)
            fig.show()
        except Exception as e:
            log.exception("Failed to show gallery stats")
            QMessageBox.critical(self, "Error", f"Failed to create visualization: {e}")
    
    def _on_show_suggestions(self):
        """Show match suggestions visualization."""
        self._ilog.log("button_click", "btn_eval_suggestions", value="clicked")
        if not self._eval_results:
            return
        
        try:
            from src.dl.evaluation import create_suggestions_figure
            fig = create_suggestions_figure(self._eval_results)
            fig.show()
        except Exception as e:
            log.exception("Failed to show suggestions")
            QMessageBox.critical(self, "Error", f"Failed to create visualization: {e}")
    
    # ==================== Verification Evaluation ====================
    
    def _build_verification_evaluation_group(self) -> QGroupBox:
        """Build the verification evaluation group."""
        group = QGroupBox("")
        layout = QVBoxLayout(group)
        
        # Description
        desc = QLabel(
            "Evaluate verification model performance using your past match annotations. "
            "Computes both ranking metrics (Rank@K) and classification metrics (AUC, precision/recall)."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Model selector row
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.cmb_verif_eval_model = QComboBox()
        self.cmb_verif_eval_model.setToolTip("Select verification model to evaluate")
        model_row.addWidget(self.cmb_verif_eval_model, 1)
        model_row.addStretch(1)
        layout.addLayout(model_row)
        
        # Run button and status
        run_row = QHBoxLayout()
        self.btn_run_verif_eval = QPushButton("▶ Run Verification Evaluation")
        self.btn_run_verif_eval.setToolTip("Compute metrics comparing verification scores to your verdicts")
        self.btn_run_verif_eval.clicked.connect(self._on_run_verification_evaluation)
        run_row.addWidget(self.btn_run_verif_eval)
        run_row.addStretch(1)
        layout.addLayout(run_row)
        
        # Results summary
        self.lbl_verif_eval_summary = QLabel("")
        self.lbl_verif_eval_summary.setWordWrap(True)
        layout.addWidget(self.lbl_verif_eval_summary)
        
        # Visualization buttons (initially disabled)
        viz_row = QHBoxLayout()
        
        self.btn_verif_confidence_dist = QPushButton("Confidence Distribution")
        self.btn_verif_confidence_dist.setToolTip("View P(same) distribution for matches vs non-matches")
        self.btn_verif_confidence_dist.setEnabled(False)
        self.btn_verif_confidence_dist.clicked.connect(self._on_show_verif_confidence_dist)
        viz_row.addWidget(self.btn_verif_confidence_dist)
        
        self.btn_verif_rank_dist = QPushButton("Rank Distribution")
        self.btn_verif_rank_dist.setToolTip("View where matches ranked by verification confidence")
        self.btn_verif_rank_dist.setEnabled(False)
        self.btn_verif_rank_dist.clicked.connect(self._on_show_verif_rank_dist)
        viz_row.addWidget(self.btn_verif_rank_dist)
        
        self.btn_verif_roc = QPushButton("ROC Curve")
        self.btn_verif_roc.setToolTip("View ROC curve and AUC")
        self.btn_verif_roc.setEnabled(False)
        self.btn_verif_roc.clicked.connect(self._on_show_verif_roc)
        viz_row.addWidget(self.btn_verif_roc)
        
        viz_row.addStretch(1)
        layout.addLayout(viz_row)
        
        # Second row of visualization buttons
        viz_row2 = QHBoxLayout()
        
        self.btn_verif_per_pair = QPushButton("Per-Pair Results")
        self.btn_verif_per_pair.setToolTip("View detailed results for each query-gallery pair")
        self.btn_verif_per_pair.setEnabled(False)
        self.btn_verif_per_pair.clicked.connect(self._on_show_verif_per_pair)
        viz_row2.addWidget(self.btn_verif_per_pair)
        
        self.btn_verif_suggestions = QPushButton("High-Confidence Suggestions")
        self.btn_verif_suggestions.setToolTip("View suggested matches based on verification confidence")
        self.btn_verif_suggestions.setEnabled(False)
        self.btn_verif_suggestions.clicked.connect(self._on_show_verif_suggestions)
        viz_row2.addWidget(self.btn_verif_suggestions)
        
        viz_row2.addStretch(1)
        layout.addLayout(viz_row2)
        
        # Store verification evaluation results
        self._verif_eval_results = None
        
        # Populate model combo box
        self._refresh_verif_eval_model_combo()
        
        return group
    
    def _refresh_verif_eval_model_combo(self):
        """Refresh the verification evaluation model combo box."""
        self.cmb_verif_eval_model.clear()
        
        if not self._dl_available or not self._verification_available:
            self.btn_run_verif_eval.setEnabled(False)
            return
        
        try:
            from src.dl.registry import DLRegistry, DEFAULT_VERIFICATION_KEY
            registry = DLRegistry.load()
            
            default_idx = 0
            for i, (key, entry) in enumerate(registry.verification_models.items()):
                suffix = ""
                if not entry.precomputed:
                    suffix = " (not precomputed)"
                self.cmb_verif_eval_model.addItem(f"{entry.display_name}{suffix}", key)
                if key == DEFAULT_VERIFICATION_KEY:
                    default_idx = i
            
            if self.cmb_verif_eval_model.count() > 0:
                self.cmb_verif_eval_model.setCurrentIndex(default_idx)
                self.btn_run_verif_eval.setEnabled(True)
            else:
                self.btn_run_verif_eval.setEnabled(False)
                
        except Exception as e:
            log.error("Failed to refresh verif eval model combo: %s", e)
    
    def _on_run_verification_evaluation(self):
        """Run verification model evaluation."""
        self._ilog.log("button_click", "btn_run_verif_eval", value="clicked")
        
        model_key = self.cmb_verif_eval_model.currentData()
        if not model_key:
            QMessageBox.warning(self, "No Model", "Please select a verification model to evaluate.")
            return
        
        try:
            from src.dl.registry import DLRegistry
            registry = DLRegistry.load()
            
            if model_key not in registry.verification_models:
                QMessageBox.warning(self, "Model Not Found", "Selected verification model not found.")
                return
            
            if not registry.verification_models[model_key].precomputed:
                QMessageBox.warning(
                    self, "Not Precomputed",
                    "This verification model needs precomputation before evaluation.\n"
                    "Run precomputation first."
                )
                return
            
            # Disable buttons during evaluation
            self.btn_run_verif_eval.setEnabled(False)
            self.lbl_verif_eval_summary.setText("Running verification evaluation...")
            
            # Force UI update
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents()
            
            # Run evaluation
            from src.dl.verification_evaluation import (
                run_verification_evaluation, save_verification_evaluation
            )
            self._verif_eval_results = run_verification_evaluation(model_key)
            
            # Auto-save evaluation results
            if save_verification_evaluation(self._verif_eval_results):
                log.info("Verification evaluation results saved for model %s", model_key)
            
            # Update summary
            r = self._verif_eval_results
            summary = (
                f"<b>Results for {r.model_name}:</b><br>"
                f"Evaluated: {r.total_evaluated} pairs ({r.total_yes_verdicts} matches, {r.total_no_verdicts} non-matches)<br>"
                f"<b>Ranking:</b> Rank@1: <b>{r.rank_at_1:.1%}</b> | "
                f"Rank@5: <b>{r.rank_at_5:.1%}</b> | "
                f"MRR: <b>{r.mrr:.3f}</b><br>"
                f"<b>Classification:</b> AUC: <b>{r.auc_roc:.3f}</b> | "
                f"Precision: <b>{r.precision:.1%}</b> | "
                f"Recall: <b>{r.recall:.1%}</b><br>"
                f"Optimal threshold: {r.optimal_threshold:.2f} | "
                f"Suggestions: {len(r.suggestions)}"
            )
            
            if r.missing_queries:
                summary += f"<br><span style='color: orange;'>⚠ {len(r.missing_queries)} queries not in precomputed data</span>"
            
            self.lbl_verif_eval_summary.setText(summary)
            
            # Enable visualization buttons
            self.btn_verif_confidence_dist.setEnabled(True)
            self.btn_verif_rank_dist.setEnabled(True)
            self.btn_verif_roc.setEnabled(True)
            self.btn_verif_per_pair.setEnabled(True)
            self.btn_verif_suggestions.setEnabled(len(r.suggestions) > 0)
            
        except Exception as e:
            log.exception("Verification evaluation failed")
            QMessageBox.critical(self, "Evaluation Failed", f"Error: {e}")
            self.lbl_verif_eval_summary.setText(f"<span style='color: red;'>Evaluation failed: {e}</span>")
        finally:
            self.btn_run_verif_eval.setEnabled(True)
    
    def _on_show_verif_confidence_dist(self):
        """Show verification confidence distribution visualization."""
        self._ilog.log("button_click", "btn_verif_confidence_dist", value="clicked")
        if not self._verif_eval_results:
            return
        
        try:
            from src.dl.verification_evaluation import create_confidence_distribution_figure
            fig = create_confidence_distribution_figure(self._verif_eval_results)
            fig.show()
        except Exception as e:
            log.exception("Failed to show confidence distribution")
            QMessageBox.critical(self, "Error", f"Failed to create visualization: {e}")
    
    def _on_show_verif_rank_dist(self):
        """Show verification rank distribution visualization."""
        self._ilog.log("button_click", "btn_verif_rank_dist", value="clicked")
        if not self._verif_eval_results:
            return
        
        try:
            from src.dl.verification_evaluation import create_verification_rank_figure
            fig = create_verification_rank_figure(self._verif_eval_results)
            fig.show()
        except Exception as e:
            log.exception("Failed to show rank distribution")
            QMessageBox.critical(self, "Error", f"Failed to create visualization: {e}")
    
    def _on_show_verif_roc(self):
        """Show ROC curve visualization."""
        self._ilog.log("button_click", "btn_verif_roc", value="clicked")
        if not self._verif_eval_results:
            return
        
        try:
            from src.dl.verification_evaluation import create_roc_curve_figure
            fig = create_roc_curve_figure(self._verif_eval_results)
            fig.show()
        except Exception as e:
            log.exception("Failed to show ROC curve")
            QMessageBox.critical(self, "Error", f"Failed to create visualization: {e}")
    
    def _on_show_verif_per_pair(self):
        """Show per-pair results visualization."""
        self._ilog.log("button_click", "btn_verif_per_pair", value="clicked")
        if not self._verif_eval_results:
            return
        
        try:
            from src.dl.verification_evaluation import create_verification_per_pair_figure
            fig = create_verification_per_pair_figure(self._verif_eval_results)
            fig.show()
        except Exception as e:
            log.exception("Failed to show per-pair results")
            QMessageBox.critical(self, "Error", f"Failed to create visualization: {e}")
    
    def _on_show_verif_suggestions(self):
        """Show verification-based match suggestions."""
        self._ilog.log("button_click", "btn_verif_suggestions", value="clicked")
        if not self._verif_eval_results:
            return
        
        try:
            from src.dl.verification_evaluation import create_verification_suggestions_figure
            fig = create_verification_suggestions_figure(self._verif_eval_results)
            fig.show()
        except Exception as e:
            log.exception("Failed to show suggestions")
            QMessageBox.critical(self, "Error", f"Failed to create visualization: {e}")
    
    def _build_finetune_group(self) -> QGroupBox:
        """Build the fine-tuning group (advanced)."""
        from PySide6.QtWidgets import QLineEdit, QDoubleSpinBox
        
        group = QGroupBox("")
        layout = QVBoxLayout(group)
        
        # Note
        note = QLabel(
            "<i>Fine-tuning allows you to train the model on your specific data. "
            "This is an advanced feature that requires significant compute time.</i>"
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        
        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep)
        
        # Model type selector (Embedding vs Verification)
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Model type:"))
        self.cmb_finetune_type = QComboBox()
        self.cmb_finetune_type.addItem("Embedding (Circle Loss)", "embedding")
        self.cmb_finetune_type.addItem("Verification (Pairwise)", "verification")
        self.cmb_finetune_type.currentIndexChanged.connect(self._on_finetune_type_changed)
        type_row.addWidget(self.cmb_finetune_type, 1)
        layout.addLayout(type_row)
        
        # Base model
        base_row = QHBoxLayout()
        base_row.addWidget(QLabel("Base model:"))
        self.cmb_base_model = QComboBox()
        base_row.addWidget(self.cmb_base_model, 1)
        layout.addLayout(base_row)
        
        # Data source selector
        data_row = QHBoxLayout()
        data_row.addWidget(QLabel("Data source:"))
        self.cmb_data_source = QComboBox()
        self.cmb_data_source.addItem("Archive data only", "archive_only")
        self.cmb_data_source.addItem("Archive + star_dataset", "archive_plus_star")
        self.cmb_data_source.currentIndexChanged.connect(self._on_data_source_changed)
        data_row.addWidget(self.cmb_data_source, 1)
        layout.addLayout(data_row)
        
        # Star dataset path (hidden by default)
        self.star_dataset_row = QHBoxLayout()
        self.star_dataset_row.addWidget(QLabel("star_dataset path:"))
        self.edit_star_dataset_path = QLineEdit()
        self.edit_star_dataset_path.setPlaceholderText("star_identification/star_dataset_resized")
        self.star_dataset_row.addWidget(self.edit_star_dataset_path, 1)
        self.btn_browse_star_dataset = QPushButton("Browse...")
        self.btn_browse_star_dataset.clicked.connect(self._on_browse_star_dataset)
        self.star_dataset_row.addWidget(self.btn_browse_star_dataset)
        layout.addLayout(self.star_dataset_row)
        
        # Initially hide star_dataset path widgets
        self._set_star_dataset_visible(False)
        
        # Output name
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Output name:"))
        self.edit_output_name = QLineEdit()
        self.edit_output_name.setPlaceholderText("e.g., finetuned_jan_2026")
        name_row.addWidget(self.edit_output_name, 1)
        layout.addLayout(name_row)
        
        # Hyperparameters
        hyper_row = QHBoxLayout()
        
        hyper_row.addWidget(QLabel("Epochs:"))
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 500)
        self.spin_epochs.setValue(25)
        hyper_row.addWidget(self.spin_epochs)
        
        hyper_row.addWidget(QLabel("LR:"))
        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setRange(1e-6, 1e-2)
        self.spin_lr.setDecimals(6)
        self.spin_lr.setSingleStep(1e-5)
        self.spin_lr.setValue(5e-5)
        hyper_row.addWidget(self.spin_lr)
        
        hyper_row.addWidget(QLabel("Batch:"))
        self.spin_train_batch = QSpinBox()
        self.spin_train_batch.setRange(1, 64)
        self.spin_train_batch.setValue(8)
        hyper_row.addWidget(self.spin_train_batch)
        
        hyper_row.addStretch(1)
        layout.addLayout(hyper_row)
        
        # Training progress bar
        self.progress_training = QProgressBar()
        self.progress_training.setVisible(False)
        layout.addWidget(self.progress_training)
        
        # Start/Cancel buttons
        train_row = QHBoxLayout()
        self.btn_start_training = QPushButton("Start Training")
        self.btn_start_training.clicked.connect(self._on_start_training)
        train_row.addWidget(self.btn_start_training)
        
        self.btn_cancel_training = QPushButton("Cancel")
        self.btn_cancel_training.clicked.connect(self._on_cancel_training)
        self.btn_cancel_training.setEnabled(False)
        train_row.addWidget(self.btn_cancel_training)
        
        train_row.addStretch(1)
        layout.addLayout(train_row)
        
        # Training progress label
        self.lbl_train_progress = QLabel("")
        layout.addWidget(self.lbl_train_progress)
        
        # Data summary label
        self.lbl_data_summary = QLabel("")
        self.lbl_data_summary.setWordWrap(True)
        layout.addWidget(self.lbl_data_summary)
        
        # Initialize finetune worker reference
        self._finetune_worker = None
        
        # Update data summary
        self._update_finetune_data_summary()
        
        return group
    
    def _set_star_dataset_visible(self, visible: bool):
        """Show or hide star_dataset path widgets."""
        self.edit_star_dataset_path.setVisible(visible)
        self.btn_browse_star_dataset.setVisible(visible)
        # Also hide/show the label
        for i in range(self.star_dataset_row.count()):
            item = self.star_dataset_row.itemAt(i)
            if item.widget():
                item.widget().setVisible(visible)
    
    def _on_finetune_type_changed(self, index: int):
        """Handle model type selection change."""
        self._refresh_finetune_base_models()
    
    def _on_data_source_changed(self, index: int):
        """Handle data source selection change."""
        data_source = self.cmb_data_source.currentData()
        self._set_star_dataset_visible(data_source == "archive_plus_star")
    
    def _on_browse_star_dataset(self):
        """Browse for star_dataset directory."""
        path = QFileDialog.getExistingDirectory(
            self,
            "Select star_dataset Directory",
            str(Path.cwd())
        )
        if path:
            self.edit_star_dataset_path.setText(path)
    
    def _refresh_finetune_base_models(self):
        """Refresh the base model combo based on selected model type."""
        self.cmb_base_model.clear()
        
        if not self._dl_available:
            return
        
        try:
            from src.dl.registry import DLRegistry
            registry = DLRegistry.load()
            
            model_type = self.cmb_finetune_type.currentData()
            
            if model_type == "verification":
                # Show verification models first, then embedding models (can be used as backbone)
                for key, entry in registry.verification_models.items():
                    self.cmb_base_model.addItem(f"[Verification] {entry.display_name}", key)
                for key, entry in registry.models.items():
                    self.cmb_base_model.addItem(f"[Embedding → Verification] {entry.display_name}", key)
            else:
                # Show embedding models only
                for key, entry in registry.models.items():
                    self.cmb_base_model.addItem(entry.display_name, key)
        except Exception as e:
            log.error("Failed to refresh finetune base models: %s", e)
    
    def _update_finetune_data_summary(self):
        """Update the data summary label."""
        try:
            from src.dl.finetune import get_data_summary
            summary = get_data_summary()
            
            if not summary["cache_exists"]:
                self.lbl_data_summary.setText(
                    "<span style='color: orange;'>⚠ Image cache not found. "
                    "Run precomputation first to cache images for training.</span>"
                )
            else:
                text = (
                    f"<b>Available data:</b> {summary['total_images']} cached images "
                    f"({summary['gallery_with_cached_images']} gallery, "
                    f"{summary['query_with_cached_images']} queries)"
                )
                if summary['confirmed_matches'] > 0:
                    text += f", {summary['confirmed_matches']} confirmed matches"
                self.lbl_data_summary.setText(text)
        except Exception as e:
            log.warning("Failed to get data summary: %s", e)
            self.lbl_data_summary.setText("")
    
    def _on_cancel_training(self):
        """Cancel ongoing training."""
        if self._finetune_worker is not None:
            self._finetune_worker.cancel()
            self.lbl_train_progress.setText("Cancelling...")
    
    def _refresh_all(self):
        """Refresh all displayed information."""
        self._refresh_status()
        self._refresh_model_list()
        self._refresh_eval_model_combo()
        self._refresh_verif_eval_model_combo()
        self._refresh_finetune_base_models()
        self._update_finetune_data_summary()
        self._update_button_states()
    
    def _refresh_status(self):
        """Refresh the status display."""
        if not self._dl_available:
            self.lbl_device.setText(f"<span style='color: red;'>Not available</span> — {self._status_message}")
            self.lbl_active_model.setText("—")
            self.lbl_precompute_status.setText("—")
            self.lbl_pending.setText("—")
            return
        
        # Device
        device_text = f"{self._device_name}"
        if self._torch_version:
            device_text += f" (PyTorch {self._torch_version})"
        self.lbl_device.setText(device_text)
        
        # Load registry
        try:
            from src.dl.registry import DLRegistry
            registry = DLRegistry.load()
            
            # Active model
            active = registry.get_active_model()
            if active:
                self.lbl_active_model.setText(active.display_name)
            else:
                self.lbl_active_model.setText("<span style='color: orange;'>None selected</span>")
            
            # Precomputation status
            if registry.has_precomputed_model():
                precomputed = registry.get_precomputed_models()
                model_names = [m.display_name for m in precomputed.values()]
                active_entry = registry.get_active_model()
                if active_entry and active_entry.precomputed:
                    self.lbl_precompute_status.setText(
                        f"<span style='color: green;'>✓ Ready</span> "
                        f"({active_entry.gallery_count} gallery, {active_entry.query_count} queries)"
                    )
                else:
                    self.lbl_precompute_status.setText(
                        f"<span style='color: orange;'>Available but not active</span>"
                    )
            else:
                self.lbl_precompute_status.setText(
                    "<span style='color: red;'>✗ Not computed</span> — Run precomputation to enable visual ranking"
                )
            
            # Verification status
            if self._verification_available:
                if registry.has_precomputed_verification_model():
                    active_verif = registry.get_active_verification_model()
                    if active_verif:
                        self.lbl_verification_status.setText(
                            f"<span style='color: green;'>✓ Ready</span> "
                            f"({active_verif.n_pairs} pairs)"
                        )
                    else:
                        self.lbl_verification_status.setText(
                            "<span style='color: orange;'>Available but not active</span>"
                        )
                else:
                    # Check if checkpoint exists
                    from src.dl.registry import DEFAULT_VERIFICATION_KEY
                    if DEFAULT_VERIFICATION_KEY in registry.verification_models:
                        self.lbl_verification_status.setText(
                            "<span style='color: orange;'>✗ Not computed</span> — will run with precomputation"
                        )
                    else:
                        self.lbl_verification_status.setText(
                            "<span style='color: gray;'>No checkpoint found</span>"
                        )
            else:
                self.lbl_verification_status.setText(
                    "<span style='color: gray;'>Module not available</span>"
                )
            
            # Pending
            pending_count = registry.get_pending_count()
            if pending_count > 0:
                self.lbl_pending.setText(f"<span style='color: orange;'>{pending_count} identities</span>")
            else:
                self.lbl_pending.setText("0 identities")
                
        except Exception as e:
            log.error("Failed to refresh status: %s", e)
            self.lbl_active_model.setText(f"<span style='color: red;'>Error: {e}</span>")
    
    def _refresh_model_list(self):
        """Refresh the model list widget."""
        self.list_models.clear()
        
        if not self._dl_available:
            return
        
        try:
            from src.dl.registry import DLRegistry
            registry = DLRegistry.load()
            
            # Add embedding models
            for key, entry in registry.models.items():
                # Status indicator
                if entry.precomputed:
                    status = "✓ Ready"
                else:
                    status = "✗ Pending"
                
                # Active indicator
                active_marker = "●" if key == registry.active_model else " "
                
                text = f" {active_marker} [Embedding] {entry.display_name}  [{status}]"
                
                item = QListWidgetItem(text)
                item.setData(Qt.UserRole, key)
                item.setData(Qt.UserRole + 1, "embedding")  # Store model type
                
                self.list_models.addItem(item)
            
            # Add verification models
            for key, entry in registry.verification_models.items():
                # Status indicator
                if entry.precomputed:
                    status = f"✓ Ready ({entry.n_pairs} pairs)"
                else:
                    status = "✗ Pending"
                
                # Active indicator
                active_marker = "●" if key == registry.active_verification_model else " "
                
                text = f" {active_marker} [Verification] {entry.display_name}  [{status}]"
                
                item = QListWidgetItem(text)
                item.setData(Qt.UserRole, key)
                item.setData(Qt.UserRole + 1, "verification")  # Store model type
                
                self.list_models.addItem(item)
                
        except Exception as e:
            log.error("Failed to refresh model list: %s", e)
    
    def _update_button_states(self):
        """Update button enabled states based on current selection."""
        if not self._dl_available:
            self.btn_run_full.setEnabled(False)
            self.btn_update_pending.setEnabled(False)
            self.btn_import_model.setEnabled(False)
            self.btn_start_training.setEnabled(False)
            return
        
        # Check if worker is running
        is_running = self._worker is not None and self._worker.isRunning()
        
        self.btn_run_full.setEnabled(not is_running)
        self.btn_update_pending.setEnabled(not is_running)
        self.btn_cancel.setEnabled(is_running)
        self.btn_import_model.setEnabled(not is_running)
        self.btn_start_training.setEnabled(not is_running)
        
        # Model list buttons
        selected_items = self.list_models.selectedItems()
        has_selection = len(selected_items) > 0
        
        if has_selection:
            key = selected_items[0].data(Qt.UserRole)
            model_type = selected_items[0].data(Qt.UserRole + 1)
            try:
                from src.dl.registry import DLRegistry, DEFAULT_MODEL_KEY, DEFAULT_VERIFICATION_KEY
                registry = DLRegistry.load()
                
                if model_type == "verification":
                    entry = registry.verification_models.get(key)
                    is_default = (key == DEFAULT_VERIFICATION_KEY)
                    self.btn_set_active.setEnabled(entry and entry.precomputed and not is_running)
                    self.btn_precompute_model.setEnabled(not is_running)
                    self.btn_remove_model.setEnabled(not is_default and not is_running)
                    self.btn_set_default.setEnabled(not is_default and not is_running)
                else:
                    entry = registry.models.get(key)
                    is_default = (key == DEFAULT_MODEL_KEY)
                    self.btn_set_active.setEnabled(entry and entry.precomputed and not is_running)
                    self.btn_precompute_model.setEnabled(not is_running)
                    self.btn_remove_model.setEnabled(not is_default and not is_running)
                    self.btn_set_default.setEnabled(not is_default and not is_running)
            except Exception:
                self.btn_set_active.setEnabled(False)
                self.btn_precompute_model.setEnabled(False)
                self.btn_remove_model.setEnabled(False)
                self.btn_set_default.setEnabled(False)
        else:
            self.btn_set_active.setEnabled(False)
            self.btn_precompute_model.setEnabled(False)
            self.btn_remove_model.setEnabled(False)
            self.btn_set_default.setEnabled(False)
    
    def _on_model_selection_changed(self):
        """Handle model list selection change."""
        selected = self.list_models.selectedItems()
        model_key = selected[0].data(Qt.UserRole) if selected else ""
        self._ilog.log("list_selection", "list_models", value=model_key)
        self._update_button_states()
    
    def _on_speed_mode_changed(self):
        """Handle speed mode selection change."""
        mode = self.cmb_speed_mode.currentData()
        self._ilog.log("combo_change", "cmb_speed_mode", value=str(mode))
        
        # Update TTA checkbox based on mode
        if mode == "fast":
            self.chk_tta.setChecked(True)  # TTA is still used, but optimized
            self.chk_tta.setToolTip("Fast mode: horizontal flip only (no vertical)")
        elif mode == "quality":
            self.chk_tta.setChecked(True)
            self.chk_tta.setToolTip("Quality mode: full TTA (horizontal + vertical flip)")
        else:  # auto
            self.chk_tta.setChecked(True)
            self.chk_tta.setToolTip("Auto mode: adapts TTA based on GPU/CPU")
    
    def _on_set_active(self):
        """Set the selected model as active."""
        selected = self.list_models.selectedItems()
        if not selected:
            return
        
        key = selected[0].data(Qt.UserRole)
        model_type = selected[0].data(Qt.UserRole + 1)
        self._ilog.log("button_click", "btn_set_active", value=key, 
                      context={"model_type": model_type})
        
        try:
            from src.dl.registry import DLRegistry
            registry = DLRegistry.load()
            
            if model_type == "verification":
                if registry.set_active_verification_model(key):
                    self._refresh_all()
                    QMessageBox.information(self, "Verification Model Activated", 
                        f"Verification model is now active.")
                else:
                    QMessageBox.warning(self, "Cannot Activate", 
                        "Verification model must be precomputed before it can be activated.")
            else:
                if registry.set_active_model(key):
                    self._refresh_all()
                    QMessageBox.information(self, "Model Activated", 
                        f"Model is now active. Visual ranking is enabled in First-order tab.")
                else:
                    QMessageBox.warning(self, "Cannot Activate", 
                        "Model must be precomputed before it can be activated.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to set active model: {e}")
    
    def _on_precompute_selected(self):
        """Precompute for the selected model."""
        selected = self.list_models.selectedItems()
        if not selected:
            return
        
        key = selected[0].data(Qt.UserRole)
        model_type = selected[0].data(Qt.UserRole + 1)
        self._ilog.log("button_click", "btn_precompute_selected", value=key,
                      context={"model_type": model_type})
        
        if model_type == "verification":
            self._start_verification_precomputation(key)
        else:
            self._start_precomputation(key)
    
    def _on_set_default_model(self):
        """Set the selected model as the new default."""
        selected = self.list_models.selectedItems()
        if not selected:
            return
        
        key = selected[0].data(Qt.UserRole)
        model_type = selected[0].data(Qt.UserRole + 1)
        self._ilog.log("button_click", "btn_set_default_model", value=key,
                      context={"model_type": model_type})
        
        from src.dl.registry import DLRegistry, DEFAULT_MODEL_KEY, DEFAULT_VERIFICATION_KEY
        registry = DLRegistry.load()
        
        if model_type == "verification":
            # Handle verification model
            if key == DEFAULT_VERIFICATION_KEY:
                QMessageBox.information(self, "Already Default", 
                    "This is already the default verification model.")
                return
            
            if key not in registry.verification_models:
                return
            
            model_entry = registry.verification_models[key]
            
            reply = QMessageBox.question(
                self, "Set as Default Verification Model?",
                f"Set '{model_entry.display_name}' as the new default verification model?\n\n"
                f"Checkpoint: {model_entry.checkpoint_path}\n\n"
                "Note: The default verification model slot will be updated.",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return
            
            try:
                # Update the default verification model entry
                checkpoint_hash = registry._compute_file_hash(Path(model_entry.checkpoint_path))
                registry.verification_models[DEFAULT_VERIFICATION_KEY] = type(model_entry)(
                    checkpoint_path=model_entry.checkpoint_path,
                    checkpoint_hash=checkpoint_hash,
                    display_name=model_entry.display_name,
                    precomputed=False  # Needs recomputation
                )
                if registry.active_verification_model == DEFAULT_VERIFICATION_KEY:
                    registry.active_verification_model = None
                registry.save()
                
                self._refresh_all()
                QMessageBox.information(
                    self, "Default Verification Model Updated",
                    f"'{model_entry.display_name}' is now the default verification model.\n\n"
                    "Please run precomputation if not already computed."
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to set default model: {e}")
        else:
            # Handle embedding model (original logic)
            if key == DEFAULT_MODEL_KEY:
                QMessageBox.information(self, "Already Default", "This model is already the default.")
                return
            
            if key not in registry.models:
                return
            
            model_entry = registry.models[key]
            
            reply = QMessageBox.question(
                self, "Set as Default Model?",
                f"Set '{model_entry.display_name}' as the new default embedding model?\n\n"
                f"Checkpoint: {model_entry.checkpoint_path}\n\n"
                "Note: The default model slot will be updated to use this checkpoint.",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return
            
            try:
                if registry.set_default_model(model_entry.checkpoint_path, model_entry.display_name):
                    self._refresh_all()
                    QMessageBox.information(
                        self, "Default Model Updated",
                        f"'{model_entry.display_name}' is now the default embedding model.\n\n"
                        "Please run precomputation if not already computed."
                    )
                else:
                    QMessageBox.warning(
                        self, "Failed",
                        "Failed to set the default model."
                    )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to set default model: {e}")
    
    def _on_import_model(self):
        """Import a new model checkpoint."""
        self._ilog.log("button_click", "btn_import_model", value="clicked")
        
        # Ask what type of model
        from PySide6.QtWidgets import QInputDialog
        model_types = ["Embedding Model (visual similarity)", "Verification Model (pairwise comparison)"]
        model_type, ok = QInputDialog.getItem(
            self, "Model Type",
            "What type of model are you importing?",
            model_types, 0, False
        )
        
        if not ok:
            return
        
        is_verification = "Verification" in model_type
        
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model Checkpoint", "",
            "PyTorch Checkpoints (*.pth *.pt);;All Files (*)"
        )
        
        if not path:
            return
        
        # Ask for display name
        name, ok = QInputDialog.getText(
            self, "Model Name", 
            "Enter a display name for this model:",
            text=Path(path).stem
        )
        
        if not ok or not name.strip():
            return
        
        try:
            from src.dl.registry import DLRegistry
            registry = DLRegistry.load()
            
            if is_verification:
                model_key = registry.register_verification_model(path, name.strip())
                self._refresh_all()
                
                reply = QMessageBox.question(
                    self, "Precompute Now?",
                    "Verification model registered. Would you like to run precomputation now?\n\n"
                    "Precomputation computes pairwise verification scores for all gallery-query pairs.",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                
                if reply == QMessageBox.Yes:
                    self._start_verification_precomputation(model_key)
            else:
                registry.register_model(path, name.strip())
                self._refresh_all()
                
                reply = QMessageBox.question(
                    self, "Precompute Now?",
                    "Model registered. Would you like to run precomputation now?\n\n"
                    "Precomputation is required before the model can be used for visual ranking.",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                
                if reply == QMessageBox.Yes:
                    # Find the key for the new model
                    registry = DLRegistry.load()
                    for key, entry in registry.models.items():
                        if entry.checkpoint_path == path:
                            self._start_precomputation(key)
                            break
                        
        except Exception as e:
            QMessageBox.critical(self, "Import Failed", f"Failed to import model: {e}")
    
    def _on_remove_model(self):
        """Remove the selected model."""
        selected = self.list_models.selectedItems()
        if not selected:
            return
        
        key = selected[0].data(Qt.UserRole)
        model_type = selected[0].data(Qt.UserRole + 1)
        self._ilog.log("button_click", "btn_remove_model", value=key,
                      context={"model_type": model_type})
        
        type_name = "verification model" if model_type == "verification" else "model"
        reply = QMessageBox.question(
            self, f"Remove {type_name.title()}?",
            f"Are you sure you want to remove this {type_name}?\n\n"
            "This will delete all precomputed data for this model.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        try:
            import shutil
            from src.dl.registry import DLRegistry
            registry = DLRegistry.load()
            
            if model_type == "verification":
                # Remove verification model
                if key in registry.verification_models:
                    del registry.verification_models[key]
                    if registry.active_verification_model == key:
                        registry.active_verification_model = None
                    registry.save()
                    
                    # Also remove precomputed data
                    model_dir = DLRegistry.get_verification_model_data_dir(key)
                    if model_dir.exists():
                        shutil.rmtree(model_dir)
                    
                    self._refresh_all()
                else:
                    QMessageBox.warning(self, "Cannot Remove", "Verification model not found.")
            else:
                if registry.remove_model(key):
                    # Also remove precomputed data
                    model_dir = DLRegistry.get_model_data_dir(key)
                    if model_dir.exists():
                        shutil.rmtree(model_dir)
                    
                    self._refresh_all()
                else:
                    QMessageBox.warning(self, "Cannot Remove", "This model cannot be removed.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to remove model: {e}")
    
    def _on_run_full_precompute(self):
        """Run full precomputation for active or default model."""
        self._ilog.log("button_click", "btn_run_full_precompute", value="clicked")
        try:
            from src.dl.registry import DLRegistry, DEFAULT_MODEL_KEY
            registry = DLRegistry.load()
            
            # Use active model if set, otherwise default
            model_key = registry.active_model or DEFAULT_MODEL_KEY
            
            if model_key not in registry.models:
                QMessageBox.warning(self, "No Model", 
                    "No model is available. Please import a model first.")
                return
            
            self._start_precomputation(model_key)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start precomputation: {e}")
    
    def _on_update_pending(self):
        """Update only pending IDs."""
        self._ilog.log("button_click", "btn_update_pending", value="clicked")
        try:
            from src.dl.registry import DLRegistry
            registry = DLRegistry.load()
            
            if registry.get_pending_count() == 0:
                QMessageBox.information(self, "No Pending", "There are no pending IDs to update.")
                return
            
            model_key = registry.active_model
            if not model_key:
                QMessageBox.warning(self, "No Active Model", 
                    "Please set an active model first.")
                return
            
            self._start_precomputation(model_key, only_pending=True)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to update pending: {e}")
    
    def _start_precomputation(self, model_key: str, only_pending: bool = False):
        """Start the precomputation worker."""
        if self._worker is not None and self._worker.isRunning():
            return
        
        include_verification = (
            self._verification_available and 
            self.chk_verification.isChecked()
        )
        
        self._ilog.log("button_click", "btn_precompute", value="started",
                      context={"model_key": model_key, "only_pending": only_pending,
                               "include_verification": include_verification})
        
        try:
            from src.dl.precompute import PrecomputeWorker
            
            self._worker = PrecomputeWorker(
                model_key=model_key,
                use_tta=self.chk_tta.isChecked(),
                use_reranking=self.chk_reranking.isChecked(),
                batch_size=self.spin_batch.value(),
                speed_mode=self.cmb_speed_mode.currentData(),
                include_gallery=self.chk_gallery.isChecked(),
                include_queries=self.chk_queries.isChecked(),
                only_pending=only_pending,
                include_verification=include_verification,
                parent=self
            )
            
            self._worker.progress.connect(self._on_progress)
            self._worker.finished.connect(self._on_finished)
            
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            self.lbl_progress.setVisible(True)
            
            self._update_button_states()
            self._worker.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start worker: {e}")
    
    def _on_cancel(self):
        """Cancel the running precomputation."""
        self._ilog.log("button_click", "btn_cancel", value="cancelled")
        if self._worker is not None:
            self._worker.cancel()
    
    def _start_verification_precomputation(self, verification_model_key: str):
        """Start verification-only precomputation for a specific verification model."""
        if self._worker is not None and self._worker.isRunning():
            return
        
        self._ilog.log("button_click", "btn_precompute_verification", value="started",
                      context={"verification_model_key": verification_model_key})
        
        try:
            from src.dl.registry import DLRegistry
            from src.dl.verification_precompute import run_verification_precompute
            
            registry = DLRegistry.load()
            
            if verification_model_key not in registry.verification_models:
                QMessageBox.warning(self, "Model Not Found", 
                    "Verification model not found in registry.")
                return
            
            verif_entry = registry.verification_models[verification_model_key]
            checkpoint_path = verif_entry.checkpoint_path
            
            if not Path(checkpoint_path).exists():
                QMessageBox.warning(self, "Checkpoint Not Found", 
                    f"Verification checkpoint not found:\n{checkpoint_path}")
                return
            
            # Show progress
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            self.lbl_progress.setVisible(True)
            self.lbl_progress.setText("Running verification precomputation...")
            self._update_button_states()
            
            # Force UI update
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents()
            
            # Get output directory
            output_dir = DLRegistry.get_verification_model_data_dir(verification_model_key) / "verification"
            
            # Run verification (synchronous for now - could make async later)
            def progress_cb(msg, cur, tot):
                self.progress_bar.setValue(int(100 * cur / max(tot, 1)))
                self.lbl_progress.setText(msg)
                QApplication.processEvents()
            
            success, message = run_verification_precompute(
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                device=self._device or "cpu",
                batch_size=self.spin_batch.value(),
                progress_callback=progress_cb,
            )
            
            if success:
                # Update registry
                from src.data.id_registry import list_ids
                n_gallery = len(list_ids("Gallery"))
                n_queries = len(list_ids("Queries"))
                registry.mark_verification_precomputed(verification_model_key, n_gallery * n_queries)
                
                self._refresh_all()
                QMessageBox.information(self, "Verification Complete", message)
                
                # Emit signal to notify other tabs
                self.verificationPrecomputeCompleted.emit()
            else:
                QMessageBox.warning(self, "Verification Failed", message)
            
        except Exception as e:
            log.exception("Verification precomputation failed")
            QMessageBox.critical(self, "Error", f"Verification failed: {e}")
        finally:
            self.progress_bar.setVisible(False)
            self.lbl_progress.setVisible(False)
            self._update_button_states()
    
    def _on_progress(self, message: str, current: int, total: int):
        """Handle progress updates from worker."""
        if total > 0:
            self.progress_bar.setValue(int(100 * current / total))
        self.lbl_progress.setText(message)
    
    def _on_finished(self, success: bool, message: str):
        """Handle worker completion."""
        self.progress_bar.setVisible(False)
        self.lbl_progress.setVisible(False)
        self._worker = None
        
        self._refresh_all()
        
        if success:
            QMessageBox.information(self, "Precomputation Complete", message)
            self.precomputeCompleted.emit()
        else:
            if message != "Cancelled":
                QMessageBox.warning(self, "Precomputation Failed", message)
    
    def _on_start_training(self):
        """Start fine-tuning."""
        self._ilog.log("button_click", "btn_start_training", value="clicked")
        
        if not self._dl_available:
            QMessageBox.warning(self, "Not Available", "Deep learning is not available.")
            return
        
        # Get selected base model
        base_model_key = self.cmb_base_model.currentData()
        if not base_model_key:
            QMessageBox.warning(self, "No Model", "Please select a base model.")
            return
        
        # Build configuration
        try:
            from src.dl.finetune import FinetuneUIConfig, FinetuneMode, DataSource, FinetuneWorker
            
            model_type = self.cmb_finetune_type.currentData()
            mode = FinetuneMode.EMBEDDING if model_type == "embedding" else FinetuneMode.VERIFICATION
            
            data_source_str = self.cmb_data_source.currentData()
            data_source = DataSource.ARCHIVE_PLUS_STAR if data_source_str == "archive_plus_star" else DataSource.ARCHIVE_ONLY
            
            star_path = None
            if data_source == DataSource.ARCHIVE_PLUS_STAR:
                star_path = self.edit_star_dataset_path.text().strip() or None
            
            config = FinetuneUIConfig(
                mode=mode,
                base_model_key=base_model_key,
                output_name=self.edit_output_name.text().strip(),
                data_source=data_source,
                star_dataset_path=star_path,
                epochs=self.spin_epochs.value(),
                learning_rate=self.spin_lr.value(),
                batch_size=self.spin_train_batch.value(),
            )
            
            # Validate
            valid, error = config.validate()
            if not valid:
                QMessageBox.warning(self, "Invalid Configuration", error)
                return
            
        except ImportError as e:
            QMessageBox.critical(
                self, "Import Error",
                f"Failed to import fine-tuning module: {e}\n\n"
                "Make sure PyTorch is installed."
            )
            return
        except Exception as e:
            QMessageBox.critical(self, "Configuration Error", str(e))
            return
        
        # Confirm start
        confirm = QMessageBox.question(
            self, "Start Fine-Tuning",
            f"Start fine-tuning with the following settings?\n\n"
            f"Mode: {config.mode.value}\n"
            f"Base model: {base_model_key}\n"
            f"Epochs: {config.epochs}\n"
            f"Learning rate: {config.learning_rate}\n"
            f"Batch size: {config.batch_size}\n\n"
            f"This may take a while (hours for full training).",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if confirm != QMessageBox.Yes:
            return
        
        # Update UI state
        self.btn_start_training.setEnabled(False)
        self.btn_cancel_training.setEnabled(True)
        self.progress_training.setVisible(True)
        self.progress_training.setRange(0, config.epochs)
        self.progress_training.setValue(0)
        self.lbl_train_progress.setText("Starting training...")
        
        # Start worker
        self._finetune_worker = FinetuneWorker(config, parent=self)
        self._finetune_worker.progress.connect(self._on_training_progress)
        self._finetune_worker.finished.connect(self._on_training_finished)
        self._finetune_worker.start()
        
        log.info("Started fine-tuning: mode=%s, base=%s, epochs=%d",
                 config.mode.value, base_model_key, config.epochs)
    
    def _on_training_progress(self, message: str, current: int, total: int):
        """Handle training progress update."""
        self.progress_training.setRange(0, total)
        self.progress_training.setValue(current)
        self.lbl_train_progress.setText(message)
    
    def _on_training_finished(self, success: bool, message: str, output_path: str):
        """Handle training completion."""
        # Reset UI state
        self.btn_start_training.setEnabled(True)
        self.btn_cancel_training.setEnabled(False)
        self.progress_training.setVisible(False)
        self._finetune_worker = None
        
        if success:
            self.lbl_train_progress.setText(f"✓ {message}")
            
            # Offer to register the model
            result = QMessageBox.question(
                self, "Training Complete",
                f"{message}\n\n"
                f"Output: {output_path}\n\n"
                "Would you like to register this model and run precomputation?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if result == QMessageBox.Yes:
                self._register_finetuned_model(output_path)
        else:
            self.lbl_train_progress.setText(f"✗ {message}")
            if message != "Cancelled":
                QMessageBox.warning(self, "Training Failed", message)
    
    def _register_finetuned_model(self, checkpoint_path: str):
        """Register a fine-tuned model in the registry."""
        try:
            from src.dl.registry import DLRegistry
            
            registry = DLRegistry.load()
            
            # Get display name
            output_name = self.edit_output_name.text().strip()
            if not output_name:
                output_name = Path(checkpoint_path).parent.name
            
            model_type = self.cmb_finetune_type.currentData()
            display_name = f"{output_name} (Fine-tuned)"
            
            # Register
            key = registry.register_finetuned_model(
                checkpoint_path=checkpoint_path,
                display_name=display_name,
                model_type=model_type,
            )
            
            log.info("Registered fine-tuned model: %s -> %s", key, checkpoint_path)
            
            # Refresh model list
            self._refresh_model_list()
            self._refresh_finetune_base_models()
            
            QMessageBox.information(
                self, "Model Registered",
                f"Model registered as: {display_name}\n\n"
                "Run precomputation to use this model for ranking."
            )
            
        except Exception as e:
            log.error("Failed to register model: %s", e)
            QMessageBox.warning(
                self, "Registration Failed",
                f"Failed to register model: {e}\n\n"
                f"You can manually import the checkpoint from:\n{checkpoint_path}"
            )
    
    def check_first_boot(self) -> bool:
        """
        Check if this is first boot and show prompt if needed.
        
        Returns True if precomputation is available, False otherwise.
        """
        if not self._dl_available:
            return False
        
        try:
            from src.dl.registry import DLRegistry
            registry = DLRegistry.load()
            
            # If first boot not completed and no precomputed models
            if not registry.first_boot_completed and not registry.has_precomputed_model():
                self._show_first_boot_prompt(registry)
                return registry.has_precomputed_model()
            
            return registry.has_precomputed_model()
            
        except Exception as e:
            log.error("First boot check failed: %s", e)
            return False
    
    def _show_first_boot_prompt(self, registry):
        """Show the first-boot precomputation prompt."""
        from src.data.id_registry import list_ids
        from src.dl.precompute import estimate_time
        
        gallery_count = len(list_ids("Gallery"))
        query_count = len(list_ids("Queries"))
        
        time_estimate = estimate_time(gallery_count, query_count)
        
        msg = QMessageBox(self)
        msg.setWindowTitle("Deep Learning Features Available")
        msg.setText(
            "starBoard can use visual similarity (deep learning) to help rank potential matches.\n\n"
            "This requires a one-time computation of your gallery and query images."
        )
        msg.setInformativeText(
            f"Gallery: {gallery_count} identities\n"
            f"Queries: {query_count} identities\n"
            f"Estimated time: {time_estimate}\n\n"
            "Would you like to run precomputation now?"
        )
        
        btn_run = msg.addButton("Run Precomputation Now", QMessageBox.AcceptRole)
        btn_skip = msg.addButton("Skip for Now", QMessageBox.RejectRole)
        chk_dont_ask = QCheckBox("Don't ask again")
        msg.setCheckBox(chk_dont_ask)
        
        msg.exec()
        
        if chk_dont_ask.isChecked():
            registry.first_boot_completed = True
            registry.save()
        
        if msg.clickedButton() == btn_run:
            # Switch to DL tab and start precomputation
            from src.dl.registry import DEFAULT_MODEL_KEY
            self._start_precomputation(DEFAULT_MODEL_KEY)
    
    def prompt_for_new_data(self, target: str, new_ids: list) -> bool:
        """
        Prompt the user about updating precomputation for new data.
        
        Args:
            target: "Gallery" or "Queries"
            new_ids: List of new ID strings
            
        Returns:
            True if user chose to update, False otherwise
        """
        if not self._dl_available:
            return False
        
        try:
            from src.dl.registry import DLRegistry
            registry = DLRegistry.load()
            
            if not registry.has_precomputed_model():
                # No precomputation yet, just track pending
                for id_ in new_ids:
                    registry.add_pending_id(target, id_)
                return False
            
            # Build model list for selection
            precomputed = registry.get_precomputed_models()
            if not precomputed:
                for id_ in new_ids:
                    registry.add_pending_id(target, id_)
                return False
            
            msg = QMessageBox(self.window() if self.window() else self)
            msg.setWindowTitle("Update Visual Similarity Index?")
            msg.setText(
                f"You added {len(new_ids)} new identities to {target}.\n\n"
                "The deep learning similarity index needs to be updated for "
                "these to appear in visual ranking results."
            )
            
            btn_update = msg.addButton("Update Now", QMessageBox.AcceptRole)
            btn_skip = msg.addButton("Skip", QMessageBox.RejectRole)
            
            msg.exec()
            
            if msg.clickedButton() == btn_update:
                # Add to pending and run update
                for id_ in new_ids:
                    registry.add_pending_id(target, id_)
                
                # Run pending update for active model
                if registry.active_model:
                    self._start_precomputation(registry.active_model, only_pending=True)
                    return True
            else:
                # Just track as pending
                for id_ in new_ids:
                    registry.add_pending_id(target, id_)
            
            return False
            
        except Exception as e:
            log.error("Failed to handle new data prompt: %s", e)
            return False

