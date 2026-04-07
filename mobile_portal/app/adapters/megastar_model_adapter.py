from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.dl import DL_AVAILABLE
from src.dl.reid_adapter import ReIDAdapter

from .megastar_artifact_loader import MegaStarArtifactAvailability


class MegaStarModelUnavailable(RuntimeError):
    pass


@dataclass
class MegaStarModelAdapter:
    availability: MegaStarArtifactAvailability
    backend_factory: type[ReIDAdapter] = ReIDAdapter

    def __post_init__(self):
        self._backend: ReIDAdapter | None = None
        self._gpu_device = None

    def _ensure_loaded(self) -> ReIDAdapter:
        if not self.availability.enabled:
            raise MegaStarModelUnavailable(self.availability.reason or 'megastar_unavailable')
        if not DL_AVAILABLE:
            raise MegaStarModelUnavailable('dl_unavailable')
        if self.availability.checkpoint_path is None:
            raise MegaStarModelUnavailable('checkpoint_missing')

        if self._backend is None:
            self._backend = self.backend_factory()

        loaded_path = self._backend.get_loaded_model_path()
        checkpoint_path = str(self.availability.checkpoint_path)
        if not self._backend.is_model_loaded() or loaded_path != checkpoint_path:
            if isinstance(self._backend, ReIDAdapter):
                if not self._load_reid_backend_cpu(self._backend, checkpoint_path):
                    raise MegaStarModelUnavailable('model_load_failed')
            else:
                if not self._backend.load_model(checkpoint_path):
                    raise MegaStarModelUnavailable('model_load_failed')
            self._park_backend_on_cpu(self._backend)
        return self._backend

    def extract_embedding(self, image_tensor: 'torch.Tensor') -> np.ndarray:
        backend = self._ensure_loaded()
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as exc:
            raise MegaStarModelUnavailable('dl_unavailable') from exc

        expected_size = self.availability.image_size or backend.get_image_size()
        if tuple(image_tensor.shape) != (3, expected_size, expected_size):
            raise ValueError(f'Expected tensor shape (3, {expected_size}, {expected_size}), got {tuple(image_tensor.shape)}')

        model = backend._model
        device = backend._device
        if model is None or device is None:
            raise MegaStarModelUnavailable('model_not_loaded')

        compute_device = self._gpu_device if self._gpu_device is not None else device
        if compute_device.type == 'cuda':
            model = model.to(compute_device)
            backend._model = model
            backend._device = compute_device
        else:
            compute_device = device

        model.eval()
        batch = image_tensor.unsqueeze(0).to(compute_device)
        if compute_device.type == 'cuda':
            from torch.amp import autocast
            ctx = autocast('cuda')
        else:
            ctx = nullcontext()

        try:
            with torch.no_grad():
                with ctx:
                    embedding = model(batch, return_normalized=True)
                    if self.availability.use_tta:
                        embedding_hflip = model(torch.flip(batch, dims=[3]), return_normalized=True)
                        embedding_vflip = model(torch.flip(batch, dims=[2]), return_normalized=True)
                        embedding = F.normalize((embedding + embedding_hflip + embedding_vflip) / 3.0, p=2, dim=1)
        finally:
            if compute_device.type == 'cuda':
                self._park_backend_on_cpu(backend)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        vector = embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise MegaStarModelUnavailable('embedding_normalization_failed')
        return vector / norm

    def _park_backend_on_cpu(self, backend: ReIDAdapter) -> None:
        try:
            import torch
        except ImportError:
            return
        model = getattr(backend, '_model', None)
        if model is None:
            return
        try:
            model = model.to(torch.device('cpu'))
            backend._model = model
            backend._device = torch.device('cpu')
            if torch.cuda.is_available() and self._gpu_device is None:
                self._gpu_device = torch.device('cuda')
        except Exception:
            return

    def _load_reid_backend_cpu(self, backend: ReIDAdapter, checkpoint_path: str) -> bool:
        try:
            import torch
        except ImportError:
            return False
        path = Path(checkpoint_path)
        if not path.exists():
            return False
        try:
            cpu_device = torch.device('cpu')
            checkpoint = torch.load(checkpoint_path, map_location=cpu_device, weights_only=False)
            config = checkpoint.get('config', None)
            if isinstance(config, dict):
                # Dict config from training scripts — extract image_size if present,
                # then discard so _create_model_from_checkpoint uses state_dict inference
                model_cfg = config.get('model', {})
                if isinstance(model_cfg, dict) and 'image_size' in model_cfg:
                    backend._image_size = model_cfg['image_size']
                else:
                    backend._image_size = 384
                config = None
            elif config and hasattr(config, 'model') and hasattr(config.model, 'image_size'):
                backend._image_size = config.model.image_size
            else:
                backend._image_size = 384
            model = backend._create_model_from_checkpoint(checkpoint, config)
            if model is None:
                return False
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            model = model.to(cpu_device)
            model.eval()
            backend._model = model
            backend._model_path = str(path)
            backend._device = cpu_device
            if torch.cuda.is_available():
                self._gpu_device = torch.device('cuda')
            return True
        except Exception:
            return False
