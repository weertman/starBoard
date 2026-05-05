import assert from 'node:assert/strict'
import { mobileSubmissionReadiness } from '../src/submissionReadiness.ts'

const baseDraft = {
  targetType: 'query',
  targetMode: 'create',
  targetId: 'q1',
  encounterDate: '2026-04-24',
  encounterSuffix: '',
  values: {},
  ready: true,
}

{
  const result = mobileSubmissionReadiness({ draft: baseDraft, fileCount: 1, submitting: false })
  assert.equal(result.disabled, true)
  assert.equal(result.summary, 'Location is required before upload.')
}

{
  const result = mobileSubmissionReadiness({ draft: { ...baseDraft, values: { location: 'Dock' } }, fileCount: 1, submitting: false })
  assert.equal(result.disabled, false)
  assert.equal(result.summary, 'Ready for query / q1 on 2026-04-24')
}

console.log('submissionReadiness tests passed')
