import type { SchemaField } from './api/client'

export type MetadataFieldGroup = {
  displayName: string
  fields: SchemaField[]
}

export function orderedMetadataGroups(fields: SchemaField[]): [string, MetadataFieldGroup][] {
  const groups = new Map<string, MetadataFieldGroup>()
  for (const field of fields) {
    if (!groups.has(field.group)) groups.set(field.group, { displayName: field.group_display_name, fields: [] })
    groups.get(field.group)!.fields.push(field)
  }
  return Array.from(groups.entries()).sort(([left], [right]) => {
    const leftRank = groupRank(left)
    const rightRank = groupRank(right)
    return leftRank - rightRank
  })
}

function groupRank(groupName: string): number {
  if (groupName === 'location') return 0
  if (groupName === 'health') return 1
  return 2
}
