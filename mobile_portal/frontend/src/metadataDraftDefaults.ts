import type { MetadataDraft } from './components/MetadataSheet'

export function initialDraftForMetadataSheet(
  metadataDraft: MetadataDraft,
  inferredTargetType: 'query' | 'gallery',
  inferredTargetId: string,
): MetadataDraft {
  return {
    ...metadataDraft,
    targetType: metadataDraft.targetType || inferredTargetType,
    targetId: metadataDraft.targetId || inferredTargetId,
  }
}
