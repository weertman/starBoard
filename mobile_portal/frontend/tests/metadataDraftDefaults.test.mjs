import assert from 'node:assert/strict'
import { initialDraftForMetadataSheet } from '../src/metadataDraftDefaults.ts'

const baseDraft = {
  targetType: 'query',
  targetMode: 'create',
  targetId: '',
  encounterDate: '2026-04-24',
  encounterSuffix: '',
  values: {},
  ready: false,
}

{
  const result = initialDraftForMetadataSheet(baseDraft, 'gallery', '')
  assert.equal(result.targetType, 'query')
  assert.equal(result.targetMode, 'create')
}

{
  const readyQuery = { ...baseDraft, ready: true, targetType: 'query', targetMode: 'create', targetId: 'Maybe eggo' }
  const result = initialDraftForMetadataSheet(readyQuery, 'gallery', 'anchovy')
  assert.equal(result.targetType, 'query')
  assert.equal(result.targetMode, 'create')
  assert.equal(result.targetId, 'Maybe eggo')
}

{
  const result = initialDraftForMetadataSheet(baseDraft, 'query', '')
  assert.equal(result.targetType, 'query')
  assert.equal(result.targetMode, 'create')
}

console.log('metadataDraftDefaults tests passed')
