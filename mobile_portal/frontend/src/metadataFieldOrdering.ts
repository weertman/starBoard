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
    if (left === 'location') return -1
    if (right === 'location') return 1
    return 0
  })
}
