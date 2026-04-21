export type ImageDescriptor = {
  image_id: string
  label: string
  encounter?: string | null
  preview_url: string
  fullres_url: string
}

export type EncounterOption = {
  encounter: string
  date: string
  label: string
}

export type GalleryEntityResponse = {
  entity_id: string
  metadata_summary: Record<string, string>
  encounters: EncounterOption[]
  images: ImageDescriptor[]
}

export type FirstOrderSearchRequest = {
  query_id: string
  top_k?: number
  preset?: 'all' | 'colors' | 'text' | 'arms_patterns'
}

export type FirstOrderSearchResponse = {
  query_id: string
  preset: 'all' | 'colors' | 'text' | 'arms_patterns'
  candidates: Array<{
    entity_id: string
    score: number
    k_contrib: number
    field_breakdown: Record<string, number>
  }>
}

export type BatchUploadImportSource =
  | { type: 'server_path'; path: string }
  | { type: 'uploaded_bundle'; upload_token: string }

export type BatchUploadLocationDraft = {
  location: string
  latitude: string
  longitude: string
}

export type BatchUploadDiscoverRequest = {
  target_archive: 'gallery' | 'query'
  discovery_mode: 'auto' | 'flat' | 'encounters' | 'grouped'
  id_prefix: string
  id_suffix: string
  flat_encounter_date?: string
  flat_encounter_suffix?: string
  batch_location?: BatchUploadLocationDraft
  import_source: BatchUploadImportSource
}

export type BatchUploadWarning = {
  code: string
  message: string
  row_id?: string | null
}

export type BatchUploadDiscoverRow = {
  row_id: string
  original_detected_id: string
  transformed_target_id: string
  action: 'create_new' | 'append_existing' | 'skip'
  target_exists: boolean
  group_name?: string | null
  encounter_folder_name?: string | null
  encounter_date?: string | null
  encounter_suffix?: string | null
  image_count: number
  sample_labels: string[]
  source_ref: string
  warnings: BatchUploadWarning[]
}

export type BatchUploadDiscoverResponse = {
  plan_id: string
  target_archive: 'gallery' | 'query'
  requested_discovery_mode: 'auto' | 'flat' | 'encounters' | 'grouped'
  resolved_discovery_mode: 'flat' | 'encounters' | 'grouped' | 'single_id' | 'empty'
  summary: {
    detected_rows: number
    detected_ids: number
    total_images: number
    new_ids: number
    existing_ids: number
    warnings: number
    errors: number
  }
  rows: BatchUploadDiscoverRow[]
  warnings: BatchUploadWarning[]
  errors: BatchUploadWarning[]
}

export type BatchUploadExecuteRequest = {
  plan_id: string
  accepted_row_ids: string[]
}

export type BatchUploadExecuteResponse = {
  status: 'ok' | 'partial' | 'error'
  plan_id: string
  batch_id: string
  target_archive: 'gallery' | 'query'
  summary: {
    executed_rows: number
    created_ids: number
    appended_ids: number
    accepted_images: number
    skipped_images: number
    rows_with_errors: number
  }
  rows: Array<{
    row_id: string
    target_id: string
    action: 'create_new' | 'append_existing' | 'skip'
    accepted_images: number
    skipped_images: number
    encounter_folder: string
    archive_paths_written: string[]
    warnings: BatchUploadWarning[]
    errors: BatchUploadWarning[]
  }>
  message: string
}

export type BatchUploadUploadResponse = {
  upload_token: string
  file_count: number
  root_entries: string[]
}

async function parseJsonOrThrow<T>(res: Response): Promise<T> {
  const text = await res.text()
  if (!text) {
    if (!res.ok) throw new Error(`Request failed: ${res.status}`)
    return {} as T
  }
  const payload = JSON.parse(text) as T
  if (!res.ok) {
    throw new Error(text)
  }
  return payload
}

export async function uploadBatchZip(file: File): Promise<BatchUploadUploadResponse> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch('/api/batch-upload/uploads', { method: 'POST', body: form })
  return parseJsonOrThrow<BatchUploadUploadResponse>(res)
}

export async function discoverBatchUpload(req: BatchUploadDiscoverRequest): Promise<BatchUploadDiscoverResponse> {
  const res = await fetch('/api/batch-upload/discover', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  return parseJsonOrThrow<BatchUploadDiscoverResponse>(res)
}

export async function executeBatchUpload(req: BatchUploadExecuteRequest): Promise<BatchUploadExecuteResponse> {
  const res = await fetch('/api/batch-upload/execute', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  return parseJsonOrThrow<BatchUploadExecuteResponse>(res)
}

export async function getGalleryEntity(entityId: string): Promise<GalleryEntityResponse> {
  const res = await fetch(`/api/gallery/entities/${encodeURIComponent(entityId)}`)
  return parseJsonOrThrow<GalleryEntityResponse>(res)
}

export async function runFirstOrderSearch(req: FirstOrderSearchRequest): Promise<FirstOrderSearchResponse> {
  const res = await fetch('/api/first-order/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  return parseJsonOrThrow<FirstOrderSearchResponse>(res)
}
