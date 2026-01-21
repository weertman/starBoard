"""
Registry for managing DL models and precomputation state.

The registry tracks:
- Registered models and their precomputation status
- Active model selection
- Pending IDs that need precomputation
- First-boot state
"""

from __future__ import annotations

import json
import hashlib
import logging
from dataclasses import dataclass, field, asdict, fields
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from src.data import archive_paths as ap

log = logging.getLogger("starBoard.dl.registry")

# Singleton instance for DLRegistry
_REGISTRY_INSTANCE: Optional['DLRegistry'] = None

# Default checkpoint path (relative to project root)
# Drop your best.pth into this folder to set it as the default model
DEFAULT_CHECKPOINT_PATH = "star_identification/checkpoints/default/best.pth"
# Fallback to legacy location if default folder is empty
FALLBACK_CHECKPOINT_PATH = "star_identification/checkpoints/megastarid/finetune/best.pth"
DEFAULT_MODEL_KEY = "default_megastarid_v1"
DEFAULT_MODEL_NAME = "ConvNeXt-Tiny Circle Loss (Default)"

# Default verification model checkpoint
DEFAULT_VERIFICATION_CHECKPOINT = "checkpoints/verification/extended_training/circleloss/nofreeze_inat1_neg0_20260109_050432/best.pth"
DEFAULT_VERIFICATION_KEY = "default_verification_v1"
DEFAULT_VERIFICATION_NAME = "CircleLoss Verification (Default)"


@dataclass
class ModelEntry:
    """Entry for a registered model."""
    checkpoint_path: str
    checkpoint_hash: str
    display_name: str
    precomputed: bool = False
    last_computed: Optional[str] = None
    gallery_count: int = 0
    query_count: int = 0
    image_count: int = 0
    # Optional fields for fine-tuned models
    backbone: Optional[str] = None
    embedding_dim: Optional[int] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> 'ModelEntry':
        # Handle unknown fields gracefully
        known_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)


@dataclass
class VerificationModelEntry:
    """Entry for a registered verification model."""
    checkpoint_path: str
    checkpoint_hash: str
    display_name: str
    precomputed: bool = False
    last_computed: Optional[str] = None
    n_pairs: int = 0
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> 'VerificationModelEntry':
        return cls(**d)


@dataclass
class PendingIds:
    """IDs that have been added since last precomputation."""
    gallery: List[str] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {"gallery": self.gallery, "queries": self.queries}
    
    @classmethod
    def from_dict(cls, d: dict) -> 'PendingIds':
        return cls(
            gallery=d.get("gallery", []),
            queries=d.get("queries", [])
        )
    
    def is_empty(self) -> bool:
        return len(self.gallery) == 0 and len(self.queries) == 0
    
    def add_gallery(self, id_: str):
        if id_ not in self.gallery:
            self.gallery.append(id_)
    
    def add_query(self, id_: str):
        if id_ not in self.queries:
            self.queries.append(id_)
    
    def clear(self):
        self.gallery.clear()
        self.queries.clear()


class DLRegistry:
    """
    Registry for DL models and precomputation state.
    
    Stored in archive/_dl_precompute/_dl_registry.json
    """
    
    VERSION = 2  # Bumped for verification model support
    
    def __init__(self):
        self.models: Dict[str, ModelEntry] = {}
        self.active_model: Optional[str] = None
        self.pending_ids: PendingIds = PendingIds()
        self.first_boot_completed: bool = False
        self._path: Optional[Path] = None
        
        # Verification models
        self.verification_models: Dict[str, VerificationModelEntry] = {}
        self.active_verification_model: Optional[str] = None
    
    @classmethod
    def get_registry_path(cls) -> Path:
        """Get the path to the registry file."""
        return cls.get_precompute_root() / "_dl_registry.json"
    
    @classmethod
    def get_precompute_root(cls) -> Path:
        """Get the root directory for precomputed data."""
        return ap.archive_root() / "_dl_precompute"
    
    @classmethod
    def get_model_data_dir(cls, model_key: str) -> Path:
        """Get the directory for a model's precomputed data."""
        return cls.get_precompute_root() / model_key
    
    @classmethod
    def get_verification_model_data_dir(cls, model_key: str) -> Path:
        """Get the directory for a verification model's precomputed data."""
        return cls.get_precompute_root() / f"verification_{model_key}"
    
    @classmethod
    def load(cls) -> 'DLRegistry':
        """Load registry from disk, or return cached singleton instance."""
        global _REGISTRY_INSTANCE
        
        # Return cached instance if available
        if _REGISTRY_INSTANCE is not None:
            return _REGISTRY_INSTANCE
        
        registry = cls()
        path = cls.get_registry_path()
        registry._path = path
        
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                registry.first_boot_completed = data.get("first_boot_completed", False)
                registry.active_model = data.get("active_model")
                registry.pending_ids = PendingIds.from_dict(data.get("pending_ids", {}))
                
                for key, model_data in data.get("models", {}).items():
                    registry.models[key] = ModelEntry.from_dict(model_data)
                
                # Load verification models
                registry.active_verification_model = data.get("active_verification_model")
                for key, model_data in data.get("verification_models", {}).items():
                    registry.verification_models[key] = VerificationModelEntry.from_dict(model_data)
                
                log.info("Loaded DL registry with %d models, %d verification models",
                         len(registry.models), len(registry.verification_models))
                
            except Exception as e:
                log.warning("Failed to load DL registry: %s", e)
                registry = cls()
                registry._path = path
        
        # Ensure default models are registered
        registry._ensure_default_model()
        registry._ensure_default_verification_model()
        
        # Cache the instance
        _REGISTRY_INSTANCE = registry
        
        return registry
    
    @classmethod
    def reload(cls) -> 'DLRegistry':
        """Force reload registry from disk, clearing the cached singleton."""
        global _REGISTRY_INSTANCE
        _REGISTRY_INSTANCE = None
        return cls.load()
    
    def _ensure_default_model(self):
        """Ensure the default model is registered if checkpoint exists."""
        if DEFAULT_MODEL_KEY in self.models:
            return
        
        # Try multiple paths for the checkpoint
        paths_to_try = [
            Path(DEFAULT_CHECKPOINT_PATH),
            Path(FALLBACK_CHECKPOINT_PATH),
        ]
        
        # Also try relative to archive root's parent (project root)
        project_root = ap.archive_root().parent
        paths_to_try.extend([
            project_root / DEFAULT_CHECKPOINT_PATH,
            project_root / FALLBACK_CHECKPOINT_PATH,
        ])
        
        checkpoint_path = None
        for p in paths_to_try:
            if p.exists():
                checkpoint_path = p
                break
        
        if checkpoint_path is not None:
            checkpoint_hash = self._compute_file_hash(checkpoint_path)
            self.models[DEFAULT_MODEL_KEY] = ModelEntry(
                checkpoint_path=str(checkpoint_path),
                checkpoint_hash=checkpoint_hash,
                display_name=DEFAULT_MODEL_NAME,
                precomputed=False
            )
            log.info("Registered default model: %s", checkpoint_path)
        else:
            log.warning("Default checkpoint not found. Tried: %s", DEFAULT_CHECKPOINT_PATH)
    
    def _ensure_default_verification_model(self):
        """Ensure the default verification model is registered if checkpoint exists."""
        if DEFAULT_VERIFICATION_KEY in self.verification_models:
            return
        
        # Try multiple paths for the checkpoint
        paths_to_try = [
            Path(DEFAULT_VERIFICATION_CHECKPOINT),
        ]
        
        # Also try relative to archive root's parent (project root)
        project_root = ap.archive_root().parent
        paths_to_try.append(project_root / DEFAULT_VERIFICATION_CHECKPOINT)
        
        checkpoint_path = None
        for p in paths_to_try:
            if p.exists():
                checkpoint_path = p
                break
        
        if checkpoint_path is not None:
            checkpoint_hash = self._compute_file_hash(checkpoint_path)
            self.verification_models[DEFAULT_VERIFICATION_KEY] = VerificationModelEntry(
                checkpoint_path=str(checkpoint_path),
                checkpoint_hash=checkpoint_hash,
                display_name=DEFAULT_VERIFICATION_NAME,
                precomputed=False
            )
            log.info("Registered default verification model: %s", checkpoint_path)
        else:
            log.debug("Default verification checkpoint not found: %s", DEFAULT_VERIFICATION_CHECKPOINT)
    
    def save(self):
        """Save registry to disk."""
        path = self._path or self.get_registry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "version": self.VERSION,
            "models": {k: v.to_dict() for k, v in self.models.items()},
            "active_model": self.active_model,
            "pending_ids": self.pending_ids.to_dict(),
            "first_boot_completed": self.first_boot_completed,
            # Verification models
            "verification_models": {k: v.to_dict() for k, v in self.verification_models.items()},
            "active_verification_model": self.active_verification_model,
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        log.debug("Saved DL registry to %s", path)
    
    @staticmethod
    def _compute_file_hash(path: Path) -> str:
        """Compute SHA256 hash of a file (first 1MB only for speed)."""
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            chunk = f.read(1024 * 1024)  # First 1MB
            h.update(chunk)
        return h.hexdigest()[:16]
    
    def has_precomputed_model(self) -> bool:
        """Check if any model has been precomputed."""
        return any(m.precomputed for m in self.models.values())
    
    def get_active_model(self) -> Optional[ModelEntry]:
        """Get the currently active model entry."""
        if self.active_model and self.active_model in self.models:
            return self.models[self.active_model]
        return None
    
    def get_precomputed_models(self) -> Dict[str, ModelEntry]:
        """Get all models that have been precomputed."""
        return {k: v for k, v in self.models.items() if v.precomputed}
    
    def set_active_model(self, model_key: str) -> bool:
        """Set the active model. Returns False if model not precomputed."""
        if model_key not in self.models:
            return False
        if not self.models[model_key].precomputed:
            return False
        self.active_model = model_key
        self.save()
        return True
    
    def register_model(self, 
                       checkpoint_path: str, 
                       display_name: str,
                       model_key: Optional[str] = None) -> str:
        """
        Register a new model.
        
        Returns the model key.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint_hash = self._compute_file_hash(path)
        
        if model_key is None:
            # Generate a key from the display name
            model_key = display_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            # Ensure uniqueness
            base_key = model_key
            counter = 1
            while model_key in self.models:
                model_key = f"{base_key}_{counter}"
                counter += 1
        
        self.models[model_key] = ModelEntry(
            checkpoint_path=str(path),
            checkpoint_hash=checkpoint_hash,
            display_name=display_name,
            precomputed=False
        )
        
        self.save()
        log.info("Registered model %s: %s", model_key, checkpoint_path)
        return model_key
    
    def mark_precomputed(self, 
                         model_key: str, 
                         gallery_count: int, 
                         query_count: int,
                         image_count: int):
        """Mark a model as precomputed with statistics."""
        if model_key not in self.models:
            return
        
        entry = self.models[model_key]
        entry.precomputed = True
        entry.last_computed = datetime.now().isoformat()
        entry.gallery_count = gallery_count
        entry.query_count = query_count
        entry.image_count = image_count
        
        # If no active model, set this one
        if self.active_model is None:
            self.active_model = model_key
        
        # Clear pending IDs
        self.pending_ids.clear()
        
        self.save()
        log.info("Marked model %s as precomputed", model_key)
    
    def reset_precomputed(self, model_key: str, delete_files: bool = True) -> bool:
        """
        Reset a model's precomputed state, optionally deleting embeddings/similarity.
        
        This keeps the YOLO image cache intact (expensive to rebuild).
        Use this when you want to rerun embedding extraction with new settings
        (e.g., different outlier detection, different TTA settings).
        
        Args:
            model_key: The model to reset
            delete_files: If True, delete embeddings and similarity files
            
        Returns:
            True if reset successful
        """
        import shutil
        
        if model_key not in self.models:
            log.warning("Model not found: %s", model_key)
            return False
        
        entry = self.models[model_key]
        
        # Clear precomputed state
        entry.precomputed = False
        entry.last_computed = None
        entry.gallery_count = 0
        entry.query_count = 0
        entry.image_count = 0
        
        # Clear active model if this was it
        if self.active_model == model_key:
            self.active_model = None
        
        if delete_files:
            model_dir = self.get_model_data_dir(model_key)
            
            # Delete embeddings directory
            embeddings_dir = model_dir / "embeddings"
            if embeddings_dir.exists():
                shutil.rmtree(embeddings_dir)
                log.info("Deleted embeddings: %s", embeddings_dir)
            
            # Delete similarity directory
            similarity_dir = model_dir / "similarity"
            if similarity_dir.exists():
                shutil.rmtree(similarity_dir)
                log.info("Deleted similarity: %s", similarity_dir)
        
        # Clear similarity lookup cache
        try:
            from .similarity_lookup import clear_cache
            clear_cache()
        except ImportError:
            pass
        
        self.save()
        log.info("Reset precomputed state for model %s", model_key)
        return True
    
    def remove_model(self, model_key: str) -> bool:
        """Remove a model from the registry."""
        if model_key not in self.models:
            return False
        if model_key == DEFAULT_MODEL_KEY:
            log.warning("Cannot remove default model")
            return False
        
        del self.models[model_key]
        
        if self.active_model == model_key:
            # Switch to another precomputed model or None
            precomputed = self.get_precomputed_models()
            self.active_model = next(iter(precomputed.keys()), None) if precomputed else None
        
        self.save()
        return True
    
    def set_default_model(self, checkpoint_path: str, display_name: Optional[str] = None) -> bool:
        """
        Set a new default model checkpoint.
        
        This updates the DEFAULT_MODEL_KEY entry with the new checkpoint.
        
        Args:
            checkpoint_path: Path to the new default checkpoint
            display_name: Optional display name (defaults to "MegaStarID (Default)")
            
        Returns:
            True if successful, False otherwise
        """
        path = Path(checkpoint_path)
        if not path.exists():
            log.error("Cannot set default model: checkpoint not found: %s", checkpoint_path)
            return False
        
        try:
            checkpoint_hash = self._compute_file_hash(path)
            
            if display_name is None:
                display_name = f"MegaStarID (Default)"
            
            # Update the default model entry
            self.models[DEFAULT_MODEL_KEY] = ModelEntry(
                checkpoint_path=str(path),
                checkpoint_hash=checkpoint_hash,
                display_name=display_name,
                precomputed=False  # New model needs precomputation
            )
            
            # Clear active model since the default changed
            if self.active_model == DEFAULT_MODEL_KEY:
                self.active_model = None
            
            self.save()
            log.info("Set new default model: %s", checkpoint_path)
            return True
            
        except Exception as e:
            log.error("Failed to set default model: %s", e)
            return False
    
    def get_default_model_path(self) -> Optional[str]:
        """Get the current default model checkpoint path."""
        if DEFAULT_MODEL_KEY in self.models:
            return self.models[DEFAULT_MODEL_KEY].checkpoint_path
        return None
    
    def add_pending_id(self, target: str, id_: str):
        """Add an ID to the pending list."""
        if target.lower() == "gallery":
            self.pending_ids.add_gallery(id_)
        else:
            self.pending_ids.add_query(id_)
        self.save()
    
    def get_pending_count(self) -> int:
        """Get the total count of pending IDs."""
        return len(self.pending_ids.gallery) + len(self.pending_ids.queries)
    
    # ============================================================
    # Verification Model Management
    # ============================================================
    
    def get_active_verification_model(self) -> Optional[VerificationModelEntry]:
        """Get the currently active verification model entry."""
        if self.active_verification_model and self.active_verification_model in self.verification_models:
            return self.verification_models[self.active_verification_model]
        return None
    
    def get_precomputed_verification_models(self) -> Dict[str, VerificationModelEntry]:
        """Get all verification models that have been precomputed."""
        return {k: v for k, v in self.verification_models.items() if v.precomputed}
    
    def set_active_verification_model(self, model_key: str) -> bool:
        """Set the active verification model. Returns False if model not precomputed."""
        if model_key not in self.verification_models:
            return False
        if not self.verification_models[model_key].precomputed:
            return False
        self.active_verification_model = model_key
        self.save()
        return True
    
    def register_verification_model(
        self,
        checkpoint_path: str,
        display_name: str,
        model_key: Optional[str] = None,
    ) -> str:
        """
        Register a new verification model.
        
        Returns the model key.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Verification checkpoint not found: {checkpoint_path}")
        
        checkpoint_hash = self._compute_file_hash(path)
        
        if model_key is None:
            # Generate a key from the display name
            model_key = "verif_" + display_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            # Ensure uniqueness
            base_key = model_key
            counter = 1
            while model_key in self.verification_models:
                model_key = f"{base_key}_{counter}"
                counter += 1
        
        self.verification_models[model_key] = VerificationModelEntry(
            checkpoint_path=str(path),
            checkpoint_hash=checkpoint_hash,
            display_name=display_name,
            precomputed=False
        )
        
        self.save()
        log.info("Registered verification model %s: %s", model_key, checkpoint_path)
        return model_key
    
    def mark_verification_precomputed(self, model_key: str, n_pairs: int):
        """Mark a verification model as precomputed with statistics."""
        if model_key not in self.verification_models:
            return
        
        entry = self.verification_models[model_key]
        entry.precomputed = True
        entry.last_computed = datetime.now().isoformat()
        entry.n_pairs = n_pairs
        
        # If no active verification model, set this one
        if self.active_verification_model is None:
            self.active_verification_model = model_key
        
        self.save()
        log.info("Marked verification model %s as precomputed (%d pairs)", model_key, n_pairs)
    
    def reset_verification_precomputed(self, model_key: str, delete_files: bool = True) -> bool:
        """
        Reset a verification model's precomputed state.
        
        Args:
            model_key: The verification model to reset
            delete_files: If True, delete verification files
            
        Returns:
            True if reset successful
        """
        import shutil
        
        if model_key not in self.verification_models:
            log.warning("Verification model not found: %s", model_key)
            return False
        
        entry = self.verification_models[model_key]
        
        # Clear precomputed state
        entry.precomputed = False
        entry.last_computed = None
        entry.n_pairs = 0
        
        # Clear active model if this was it
        if self.active_verification_model == model_key:
            self.active_verification_model = None
        
        if delete_files:
            model_dir = self.get_verification_model_data_dir(model_key)
            verification_dir = model_dir / "verification"
            if verification_dir.exists():
                shutil.rmtree(verification_dir)
                log.info("Deleted verification data: %s", verification_dir)
        
        # Clear verification lookup cache
        try:
            from .verification_lookup import clear_verification_cache
            clear_verification_cache()
        except ImportError:
            pass
        
        self.save()
        log.info("Reset verification precomputed state for model %s", model_key)
        return True
    
    def has_precomputed_verification_model(self) -> bool:
        """Check if any verification model has been precomputed."""
        return any(m.precomputed for m in self.verification_models.values())
    
    # ============================================================
    # Fine-tuned Model Registration
    # ============================================================
    
    def register_finetuned_model(
        self,
        checkpoint_path: str,
        display_name: str,
        model_type: str = "embedding",
        model_key: Optional[str] = None,
        backbone: Optional[str] = None,
        embedding_dim: Optional[int] = None,
    ) -> str:
        """
        Register a fine-tuned model from the UI.
        
        This is a convenience method that handles both embedding and verification
        models with sensible defaults for fine-tuned checkpoints.
        
        Args:
            checkpoint_path: Path to the fine-tuned checkpoint
            display_name: Human-readable name for the model
            model_type: "embedding" or "verification"
            model_key: Optional custom key (auto-generated if not provided)
            backbone: Backbone architecture (for embedding models)
            embedding_dim: Embedding dimension (for embedding models)
            
        Returns:
            The registered model key
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        if model_type == "verification":
            return self.register_verification_model(
                checkpoint_path=checkpoint_path,
                display_name=display_name,
                model_key=model_key,
            )
        else:
            # For embedding models, we store additional metadata
            key = self.register_model(
                checkpoint_path=checkpoint_path,
                display_name=display_name,
                model_key=model_key,
            )
            
            # Store backbone and embedding_dim if provided
            # These are used when loading the model for inference
            entry = self.models[key]
            if backbone:
                # Store as additional attribute (will be saved via to_dict)
                entry.backbone = backbone  # type: ignore
            if embedding_dim:
                entry.embedding_dim = embedding_dim  # type: ignore
            
            self.save()
            return key

