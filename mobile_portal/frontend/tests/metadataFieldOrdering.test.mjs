import assert from 'node:assert/strict'
import { orderedMetadataGroups } from '../src/metadataFieldOrdering.ts'

const fields = [
  { name: 'arm_color', display_name: 'Arm color', group: 'color', group_display_name: 'Color' },
  { name: 'health_codes', display_name: 'Health coding', group: 'health', group_display_name: 'Health Coding' },
  { name: 'location', display_name: 'Location', group: 'location', group_display_name: 'Location' },
  { name: 'notes', display_name: 'Notes', group: 'notes', group_display_name: 'Notes' },
]

const groups = orderedMetadataGroups(fields)
assert.equal(groups[0][0], 'location')
assert.equal(groups[0][1].fields[0].name, 'location')
assert.equal(groups[1][0], 'health')
assert.equal(groups[1][1].fields[0].name, 'health_codes')
assert.equal(groups[2][0], 'color')

console.log('metadataFieldOrdering tests passed')
