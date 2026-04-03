from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

from src.dl.image_cache import CACHE_SIZE

log = logging.getLogger('starboard.mobile_portal.megastar_query_preprocess')

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class MegaStarQueryPreprocessError(ValueError):
    pass


@dataclass(frozen=True)
class MegaStarPreprocessedQuery:
    image_tensor: 'torch.Tensor'
    cache_image_size: tuple[int, int]
    original_image_size: tuple[int, int]


class MegaStarQueryPreprocessor:
    def __init__(self, *, image_size: int, yolo_preprocessor=None, cache_size: int = CACHE_SIZE):
        self.image_size = image_size
        self.cache_size = cache_size
        self.yolo_preprocessor = yolo_preprocessor

    def preprocess_upload_bytes(self, payload: bytes) -> MegaStarPreprocessedQuery:
        image = self._decode_image(payload)
        return self.preprocess_pil_image(image)

    def preprocess_pil_image(self, image: Image.Image) -> MegaStarPreprocessedQuery:
        try:
            import torch
        except ImportError as exc:
            raise MegaStarQueryPreprocessError('torch_unavailable') from exc

        normalized = ImageOps.exif_transpose(image).convert('RGB')
        original_size = normalized.size
        cached_image = self._apply_cache_preprocessing(normalized)
        tensor = self._to_tensor(cached_image, torch)
        return MegaStarPreprocessedQuery(
            image_tensor=tensor,
            cache_image_size=cached_image.size,
            original_image_size=original_size,
        )

    def _decode_image(self, payload: bytes) -> Image.Image:
        if not payload:
            raise MegaStarQueryPreprocessError('empty_upload')
        try:
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
                return image.copy()
        except Exception as exc:
            raise MegaStarQueryPreprocessError('image_decode_failed') from exc

    def _apply_cache_preprocessing(self, image: Image.Image) -> Image.Image:
        processed = image
        if self.yolo_preprocessor is not None:
            try:
                yolo_image = self.yolo_preprocessor.process_image(image)
            except Exception as exc:
                log.warning('YOLO preprocessing raised for MegaStar query; falling back to RGB image: %s', exc)
                yolo_image = None
            if yolo_image is not None:
                processed = yolo_image.convert('RGB')
        return self._downscale_only(processed)

    def _downscale_only(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width <= 0 or height <= 0:
            raise MegaStarQueryPreprocessError('invalid_image_dimensions')
        scale = self.cache_size / max(width, height)
        if scale < 1.0:
            new_width = max(1, int(width * scale))
            new_height = max(1, int(height * scale))
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return image.copy()

    def _to_tensor(self, image: Image.Image, torch_module) -> 'torch.Tensor':
        resized = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        if array.ndim != 3 or array.shape[2] != 3:
            raise MegaStarQueryPreprocessError('invalid_rgb_channels')
        normalized = (array - IMAGENET_MEAN) / IMAGENET_STD
        chw = np.transpose(normalized, (2, 0, 1)).copy()
        return torch_module.from_numpy(chw)
