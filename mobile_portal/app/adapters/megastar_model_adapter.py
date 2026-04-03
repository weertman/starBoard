from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

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
            if not self._backend.load_model(checkpoint_path):
                raise MegaStarModelUnavailable('model_load_failed')
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

        model.eval()
        batch = image_tensor.unsqueeze(0).to(device)
        if device.type == 'cuda':
            from torch.amp import autocast
            ctx = autocast('cuda')
        else:
            ctx = nullcontext()

        with torch.no_grad():
            with ctx:
                embedding = model(batch, return_normalized=True)
                if self.availability.use_tta:
                    embedding_hflip = model(torch.flip(batch, dims=[3]), return_normalized=True)
                    embedding_vflip = model(torch.flip(batch, dims=[2]), return_normalized=True)
                    embedding = F.normalize((embedding + embedding_hflip + embedding_vflip) / 3.0, p=2, dim=1)

        vector = embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise MegaStarModelUnavailable('embedding_normalization_failed')
        return vector / norm
