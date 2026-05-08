from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from ..auth import require_authenticated_email
from ..models.batch_upload_api import (
    BatchUploadChunkedUploadStartResponse,
    BatchUploadDiscoverRequest,
    BatchUploadDiscoverResponse,
    BatchUploadExecuteRequest,
    BatchUploadExecuteResponse,
    BatchUploadUploadResponse,
)
from ..services.batch_upload_discover_service import build_discover_preview
from ..services.batch_upload_execute_service import BatchUploadPlanNotFoundError, execute_batch_upload
from ..services.batch_upload_upload_service import (
    append_uploaded_zip_chunk,
    finalize_chunked_uploaded_bundle,
    start_chunked_uploaded_bundle,
    stage_uploaded_bundle,
    stage_uploaded_folder,
)

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


@router.post('/batch-upload/uploads/chunked', response_model=BatchUploadChunkedUploadStartResponse)
def batch_upload_start_chunked_upload(
    _user_email: str = Depends(require_authenticated_email),
):
    return start_chunked_uploaded_bundle()


@router.post('/batch-upload/uploads/chunked/{upload_token}/chunks')
def batch_upload_append_chunk(
    upload_token: str,
    offset: int = Form(...),
    total_size: int = Form(...),
    filename: str = Form(...),
    chunk: UploadFile = File(...),
    _user_email: str = Depends(require_authenticated_email),
):
    append_uploaded_zip_chunk(upload_token, offset, total_size, filename, chunk)
    return {'status': 'ok'}


@router.post('/batch-upload/uploads/chunked/{upload_token}/finalize', response_model=BatchUploadUploadResponse)
def batch_upload_finalize_chunked_upload(
    upload_token: str,
    _user_email: str = Depends(require_authenticated_email),
):
    return finalize_chunked_uploaded_bundle(upload_token)


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
