import type { MetadataDraft } from './components/MetadataSheet'

type MobileSubmissionReadinessInput = {
  draft: MetadataDraft
  fileCount: number
  submitting: boolean
}

export function mobileSubmissionReadiness({ draft, fileCount, submitting }: MobileSubmissionReadinessInput) {
  const locationReady = Boolean(draft.values.location?.trim())
  const disabled = submitting || !draft.ready || !locationReady || fileCount === 0
  let summary = 'Open metadata, fill the required targeting info, then tap Ready.'
  if (draft.ready && !locationReady) {
    summary = 'Location is required before upload.'
  } else if (draft.ready) {
    summary = `Ready for ${draft.targetType} / ${draft.targetId || 'missing target'} on ${draft.encounterDate}`
  }
  return { disabled, locationReady, summary }
}
