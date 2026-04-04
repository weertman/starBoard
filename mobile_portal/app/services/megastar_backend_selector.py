from __future__ import annotations

from dataclasses import dataclass

from ..adapters.megastar_artifact_loader import load_megastar_artifact_availability
from ..config import MegaStarCapabilityStatus, Settings, get_settings
from ..models.megastar_api import MegaStarLookupResponse
from .megastar_lookup_service import MegaStarLookupService
from .megastar_worker_client import MegaStarWorkerClient


class MegaStarLookupBackend:
    def capability_status(self) -> MegaStarCapabilityStatus:
        raise NotImplementedError

    def lookup_upload(self, *, filename: str, content: bytes, content_type: str | None = None) -> MegaStarLookupResponse:
        raise NotImplementedError


@dataclass(frozen=True)
class LocalMegaStarLookupBackend(MegaStarLookupBackend):
    settings: Settings

    def capability_status(self) -> MegaStarCapabilityStatus:
        availability = load_megastar_artifact_availability(self.settings)
        return MegaStarCapabilityStatus(
            enabled=availability.enabled,
            state=availability.state,
            backend='local',
            reason=availability.reason,
            model_key=availability.model_key,
            artifact_dir=availability.artifact_dir,
        )

    def lookup_upload(self, *, filename: str, content: bytes, content_type: str | None = None) -> MegaStarLookupResponse:
        service = get_local_megastar_lookup_service(self.settings)
        return service.lookup_upload(filename=filename, content=content, content_type=content_type)


@dataclass(frozen=True)
class WorkerMegaStarLookupBackend(MegaStarLookupBackend):
    client: MegaStarWorkerClient

    def capability_status(self) -> MegaStarCapabilityStatus:
        return self.client.capability_status()

    def lookup_upload(self, *, filename: str, content: bytes, content_type: str | None = None) -> MegaStarLookupResponse:
        return self.client.lookup_upload(filename=filename, content=content, content_type=content_type)


def get_local_megastar_lookup_service(settings: Settings) -> MegaStarLookupService:
    return MegaStarLookupService(settings=settings)


def get_megastar_lookup_backend(settings: Settings | None = None) -> MegaStarLookupBackend:
    settings = settings or get_settings()
    if settings.megastar_backend == 'worker':
        return WorkerMegaStarLookupBackend(client=MegaStarWorkerClient(settings=settings))
    return LocalMegaStarLookupBackend(settings=settings)


def get_megastar_capability_status(settings: Settings | None = None) -> MegaStarCapabilityStatus:
    return get_megastar_lookup_backend(settings).capability_status()
