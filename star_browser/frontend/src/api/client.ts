export type LocationSite = {
  name: string
  latitude: number
  longitude: number
}

export type LocationSitesResponse = {
  sites: LocationSite[]
}

export type SchemaFieldOption = {
  label: string
  value: string | number
}

export type SchemaField = {
  name: string
  display_name: string
  field_type: string
  group: string
  group_display_name: string
  required: boolean
  tooltip: string
  min_value?: number | null
  max_value?: number | null
  options: SchemaFieldOption[]
  vocabulary: string[]
  mobile_widget: string
}

export type MetadataSchemaResponse = {
  fields: SchemaField[]
}

export type SubmissionRequest = {
  target_type: 'gallery' | 'query'
  target_mode: 'create' | 'append'
  target_id: string
  encounter_date: string
  encounter_suffix?: string
  metadata: Record<string, string>
  files: File[]
}

export type SubmissionResponse = {
  status: string
  entity_type: 'gallery' | 'query'
  entity_id: string
  encounter_folder: string
  accepted_images: number
  skipped_images: number
  archive_paths_written: string[]
  message: string
}

export type ImageDescriptor = {
  image_id: string
  label: string
  encounter?: string | null
  preview_url: string
  fullres_url: string
}

export type FirstOrderMediaImage = ImageDescriptor & {
  is_best: boolean
}

export type FirstOrderMediaResponse = {
  target_type: 'query' | 'gallery'
  entity_id: string
  images: FirstOrderMediaImage[]
}

export type EncounterOption = {
  encounter: string
  date: string
  label: string
}

export type MetadataRow = {
  row_index: number
  source: string
  values: Record<string, string>
}

export type TimelineEvent = {
  encounter: string
  date: string
  label: string
  image_count: number
  image_labels: string[]
}

export type GalleryEntityResponse = {
  archive_type?: 'query' | 'gallery'
  entity_id: string
  metadata_summary: Record<string, string>
  metadata_rows?: MetadataRow[]
  timeline?: TimelineEvent[]
  encounters: EncounterOption[]
  images: ImageDescriptor[]
}

export type IdReviewOption = {
  entity_id: string
  label: string
  location: string
  last_observation_date: string
  metadata: Record<string, string>
}

export type IdReviewOptionsResponse = {
  archive_type: 'query' | 'gallery'
  options: IdReviewOption[]
}

export type FirstOrderQueryOption = {
  query_id: string
  state: 'not_attempted' | 'pinned' | 'attempted' | 'matched'
  last_observation_date?: string | null
  last_location?: string | null
  easy_match_score: number
  quality: Record<string, number | null>
  metadata?: Record<string, string>
}

export type FirstOrderQueryOptionsResponse = {
  queries: FirstOrderQueryOption[]
}

export type FirstOrderPreset = 'all' | 'colors' | 'text' | 'arms_patterns' | 'megastar'

export type FirstOrderSearchRequest = {
  query_id: string
  top_k?: number
  preset?: FirstOrderPreset
  query_image_id?: string
}

export type FirstOrderSearchResponse = {
  query_id: string
  preset: FirstOrderPreset
  query_image_id?: string | null
  candidates: Array<{
    entity_id: string
    score: number
    k_contrib: number
    field_breakdown: Record<string, number>
    preferred_image_id?: string | null
  }>
}

export type BatchUploadImportSource =
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

export async function getLocationSites(): Promise<LocationSitesResponse> {
  const res = await fetch('/api/locations/sites')
  return parseJsonOrThrow<LocationSitesResponse>(res)
}

export async function getMetadataSchema(): Promise<MetadataSchemaResponse> {
  const res = await fetch('/api/schema/metadata')
  return parseJsonOrThrow<MetadataSchemaResponse>(res)
}

export async function submitEntry(req: SubmissionRequest): Promise<SubmissionResponse> {
  const form = new FormData()
  form.append('payload', JSON.stringify({
    target_type: req.target_type,
    target_mode: req.target_mode,
    target_id: req.target_id,
    encounter_date: req.encounter_date,
    encounter_suffix: req.encounter_suffix ?? '',
    metadata: req.metadata,
  }))
  for (const file of req.files) {
    form.append('files', file)
  }
  const res = await fetch('/api/submissions', {
    method: 'POST',
    body: form,
  })
  return parseJsonOrThrow<SubmissionResponse>(res)
}

export async function uploadBatchZip(file: File): Promise<BatchUploadUploadResponse> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch('/api/batch-upload/uploads', { method: 'POST', body: form })
  return parseJsonOrThrow<BatchUploadUploadResponse>(res)
}

export async function uploadBatchFolder(files: File[]): Promise<BatchUploadUploadResponse> {
  const form = new FormData()
  for (const file of files) {
    const relativePath = (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name
    form.append('files', file, relativePath)
  }
  const res = await fetch('/api/batch-upload/folder-uploads', { method: 'POST', body: form })
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

export async function getIdReviewOptions(archiveType: 'query' | 'gallery'): Promise<IdReviewOptionsResponse> {
  const res = await fetch(`/api/id-review/options/${archiveType}`)
  return parseJsonOrThrow<IdReviewOptionsResponse>(res)
}

export async function getIdReviewEntity(archiveType: 'query' | 'gallery', entityId: string): Promise<GalleryEntityResponse> {
  const res = await fetch(`/api/id-review/entities/${archiveType}/${encodeURIComponent(entityId)}`)
  return parseJsonOrThrow<GalleryEntityResponse>(res)
}

export async function getGalleryEntity(entityId: string): Promise<GalleryEntityResponse> {
  return getIdReviewEntity('gallery', entityId)
}

export async function getFirstOrderQueries(): Promise<FirstOrderQueryOptionsResponse> {
  const res = await fetch('/api/first-order/queries')
  return parseJsonOrThrow<FirstOrderQueryOptionsResponse>(res)
}

export async function getFirstOrderMedia(targetType: 'query' | 'gallery', entityId: string): Promise<FirstOrderMediaResponse> {
  const encoded = encodeURIComponent(entityId)
  const path = targetType === 'query'
    ? `/api/first-order/queries/${encoded}/media`
    : `/api/first-order/candidates/${encoded}/media`
  const res = await fetch(path)
  return parseJsonOrThrow<FirstOrderMediaResponse>(res)
}

export async function runFirstOrderSearch(req: FirstOrderSearchRequest): Promise<FirstOrderSearchResponse> {
  const res = await fetch('/api/first-order/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  return parseJsonOrThrow<FirstOrderSearchResponse>(res)
}
