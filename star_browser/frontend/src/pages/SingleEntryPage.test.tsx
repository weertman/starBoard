import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { SingleEntryPage } from './SingleEntryPage'

vi.mock('../api/client', () => ({
  getMetadataSchema: vi.fn(),
  submitEntry: vi.fn(),
}))

import { getMetadataSchema, submitEntry } from '../api/client'

const mockedGetMetadataSchema = vi.mocked(getMetadataSchema)
const mockedSubmitEntry = vi.mocked(submitEntry)

const schemaResponse = {
  fields: [
    {
      name: 'location',
      display_name: 'Location',
      field_type: 'text_history',
      group: 'location',
      group_display_name: 'Location',
      required: false,
      tooltip: 'Written description of the star\'s location',
      min_value: null,
      max_value: null,
      options: [],
      vocabulary: ['Dock', 'Pier'],
      mobile_widget: 'location',
    },
    {
      name: 'num_apparent_arms',
      display_name: 'Number of apparent arms',
      field_type: 'numeric_int',
      group: 'numeric',
      group_display_name: 'Numeric Measurements',
      required: false,
      tooltip: 'Number of visually apparent arms',
      min_value: 0,
      max_value: 30,
      options: [],
      vocabulary: [],
      mobile_widget: 'number',
    },
    {
      name: 'health_observation',
      display_name: 'Health observation',
      field_type: 'text_free',
      group: 'notes',
      group_display_name: 'Notes',
      required: false,
      tooltip: 'Observations about the star\'s health',
      min_value: null,
      max_value: null,
      options: [],
      vocabulary: [],
      mobile_widget: 'textarea',
    },
  ],
}

describe('SingleEntryPage', () => {
  afterEach(() => {
    cleanup()
  })

  beforeEach(() => {
    mockedGetMetadataSchema.mockReset()
    mockedSubmitEntry.mockReset()
    mockedGetMetadataSchema.mockResolvedValue(schemaResponse)
    mockedSubmitEntry.mockResolvedValue({
      status: 'accepted',
      entity_type: 'query',
      entity_id: 'q1',
      encounter_folder: '04_01_26',
      accepted_images: 1,
      skipped_images: 0,
      archive_paths_written: ['/archive/queries/q1/04_01_26/capture.jpg'],
      message: 'Submission incorporated into archive',
    })
  })

  it('loads schema fields and renders grouped form controls', async () => {
    render(<SingleEntryPage />)

    expect(await screen.findByRole('heading', { name: 'Location' })).toBeInTheDocument()
    expect(screen.getByLabelText('Location')).toBeInTheDocument()
    expect(screen.getByLabelText('Number of apparent arms')).toBeInTheDocument()
    expect(screen.getByLabelText('Health observation')).toBeInTheDocument()
  })

  it('submits target info, metadata, and files through the client API', async () => {
    const user = userEvent.setup()
    render(<SingleEntryPage />)

    await screen.findByRole('heading', { name: 'Location' })
    await user.type(screen.getByLabelText('Target ID'), 'q1')
    await user.type(screen.getByLabelText('Location'), 'Dock')
    await user.type(screen.getByLabelText('Number of apparent arms'), '12')
    await user.type(screen.getByLabelText('Health observation'), 'Looks healthy')

    const file = new File(['image-bytes'], 'capture.jpg', { type: 'image/jpeg' })
    await user.upload(screen.getByLabelText('Images'), file)
    await user.click(screen.getByRole('button', { name: 'Submit entry' }))

    await waitFor(() => {
      expect(mockedSubmitEntry).toHaveBeenCalledTimes(1)
    })

    const payload = mockedSubmitEntry.mock.calls[0][0]
    expect(payload.target_type).toBe('query')
    expect(payload.target_mode).toBe('create')
    expect(payload.target_id).toBe('q1')
    expect(payload.metadata.location).toBe('Dock')
    expect(payload.metadata.num_apparent_arms).toBe('12')
    expect(payload.metadata.health_observation).toBe('Looks healthy')
    expect(payload.files).toHaveLength(1)
    expect(await screen.findByText(/Submission incorporated into archive/)).toBeInTheDocument()
  })
})
