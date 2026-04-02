export type SessionResponse = {
  authenticated_email: string
  capabilities: Record<string, boolean>
}

export type SchemaField = {
  name: string
  display_name: string
  field_type: string
  group: string
  group_display_name: string
  required: boolean
  tooltip: string
  options: { label: string; value: string | number }[]
  vocabulary: string[]
  mobile_widget: string
}

export type MetadataSchemaResponse = { fields: SchemaField[] }

export type ImageDescriptor = {
  image_id: string
  label: string
  fullres_url: string
  preview_url: string
  width?: number
  height?: number
}

export type ImageWindow = {
  offset: number
  count: number
  total: number
  items: ImageDescriptor[]
  next_offset?: number | null
}

export type ArchiveEntityResponse = {
  entity_type: 'gallery' | 'query'
  entity_id: string
  metadata_summary: Record<string, string>
  image_window: ImageWindow
}

export type EntitySuggestionResponse = {
  entity_type: 'gallery' | 'query'
  query: string
  items: string[]
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

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init)
  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || `Request failed: ${res.status}`)
  }
  return res.json() as Promise<T>
}

export const getSession = () => api<SessionResponse>('/api/session')
export const getMetadataSchema = () => api<MetadataSchemaResponse>('/api/schema/metadata')
export const lookupEntity = (entityId: string, entityType: 'gallery' | 'query' = 'gallery') => api<ArchiveEntityResponse>(`/api/archive/entities/${encodeURIComponent(entityId)}?entity_type=${entityType}`)
export const getEntityImages = (entityId: string, entityType: 'gallery' | 'query', offset: number, limit = 4) => api<ImageWindow>(`/api/archive/entities/${encodeURIComponent(entityId)}/images?entity_type=${entityType}&offset=${offset}&limit=${limit}`)
export const suggestEntities = (entityType: 'gallery' | 'query', query: string, limit = 8) => api<EntitySuggestionResponse>(`/api/archive/suggest?entity_type=${entityType}&query=${encodeURIComponent(query)}&limit=${limit}`)

export async function submitObservation(payload: Record<string, unknown>, files: File[]): Promise<SubmissionResponse> {
  const form = new FormData()
  form.append('payload', JSON.stringify(payload))
  for (const file of files) form.append('files', file)
  const res = await fetch('/api/submissions', { method: 'POST', body: form })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || `Submit failed: ${res.status}`)
  }
  return res.json()
}
