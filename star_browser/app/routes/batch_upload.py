from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from ..auth import require_authenticated_email
from ..models.batch_upload_api import (
    BatchUploadDiscoverRequest,
    BatchUploadDiscoverResponse,
    BatchUploadExecuteRequest,
    BatchUploadExecuteResponse,
    BatchUploadUploadResponse,
)
from ..services.batch_upload_discover_service import build_discover_preview
from ..services.batch_upload_execute_service import BatchUploadPlanNotFoundError, execute_batch_upload
from ..services.batch_upload_upload_service import stage_uploaded_bundle, stage_uploaded_folder

router = APIRouter()


def _validate_required_batch_location(request: BatchUploadDiscoverRequest) -> None:
    if not request.batch_location.location.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Location is required before upload')


@router.post('/batch-upload/uploads', response_model=BatchUploadUploadResponse)
def batch_upload_uploads(
    file: UploadFile = File(...),
    _user_email: str = Depends(require_authenticated_email),
):
    return stage_uploaded_bundle(file)


@router.post('/batch-upload/folder-uploads', response_model=BatchUploadUploadResponse)
def batch_upload_folder_uploads(
    files: list[UploadFile] = File(...),
    _user_email: str = Depends(require_authenticated_email),
):
    return stage_uploaded_folder(files)


@router.post('/batch-upload/discover', response_model=BatchUploadDiscoverResponse)
def batch_upload_discover(
    request: BatchUploadDiscoverRequest,
    _user_email: str = Depends(require_authenticated_email),
):
    _validate_required_batch_location(request)
    return build_discover_preview(request)


@router.post('/batch-upload/execute', response_model=BatchUploadExecuteResponse)
def batch_upload_execute(
    request: BatchUploadExecuteRequest,
    _user_email: str = Depends(require_authenticated_email),
):
    try:
        return execute_batch_upload(request)
    except BatchUploadPlanNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='batch_plan_not_found')
