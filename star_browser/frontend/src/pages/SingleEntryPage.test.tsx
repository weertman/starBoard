import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

vi.mock('../components/LocationSiteMap', () => ({
  LocationSiteMap: ({ onPick, sites }: { onPick: (lat: number, lon: number) => void; sites: Array<{ name: string }> }) => (
    <div>
      <div title="Location map">Mock map</div>
      <div>{sites.map((site) => site.name).join(', ')}</div>
      <button type="button" aria-label="Location map selector" onClick={() => onPick(48.5, -123.25)}>pick mock point</button>
    </div>
  ),
}))

import { SingleEntryPage } from './SingleEntryPage'

vi.mock('../api/client', () => ({
  getLocationSites: vi.fn(),
  getMetadataSchema: vi.fn(),
  submitEntry: vi.fn(),
}))

import { getLocationSites, getMetadataSchema, submitEntry } from '../api/client'

const mockedGetLocationSites = vi.mocked(getLocationSites)
const mockedGetMetadataSchema = vi.mocked(getMetadataSchema)
const mockedSubmitEntry = vi.mocked(submitEntry)

const schemaResponse = {
  fields: [
    {
      name: 'stripe_color',
      display_name: 'Stripe color',
      field_type: 'color_categorical',
      group: 'stripe',
      group_display_name: 'Stripe Morphology',
      required: false,
      tooltip: 'General color of arm stripes',
      min_value: null,
      max_value: null,
      options: [],
      vocabulary: ['Red', 'Orange'],
      mobile_widget: 'color_select',
    },
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
      name: 'latitude',
      display_name: 'Latitude',
      field_type: 'numeric_float',
      group: 'location',
      group_display_name: 'Location',
      required: false,
      tooltip: 'Latitude in decimal degrees',
      min_value: -90,
      max_value: 90,
      options: [],
      vocabulary: [],
      mobile_widget: 'number',
    },
    {
      name: 'longitude',
      display_name: 'Longitude',
      field_type: 'numeric_float',
      group: 'location',
      group_display_name: 'Location',
      required: false,
      tooltip: 'Longitude in decimal degrees',
      min_value: -180,
      max_value: 180,
      options: [],
      vocabulary: [],
      mobile_widget: 'number',
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
    mockedGetLocationSites.mockReset()
    mockedGetMetadataSchema.mockReset()
    mockedSubmitEntry.mockReset()
    mockedGetLocationSites.mockResolvedValue({
      sites: [
        { name: 'Dock', latitude: 48.546, longitude: -123.013 },
        { name: 'Pier', latitude: 48.5, longitude: -123.2 },
      ],
    })
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

    const headings = await screen.findAllByRole('heading', { level: 2 })
    expect(headings[0]).toHaveTextContent('Location')
    const savedLocations = screen.getByLabelText('Saved locations') as HTMLSelectElement
    expect(savedLocations.tagName).toBe('SELECT')
    expect(savedLocations.options[0].textContent).toBe('Add new…')
    expect(savedLocations.options[0].value).toBe('__new__')
    expect(screen.getByRole('button', { name: 'Add new location' })).toBeInTheDocument()
    expect(screen.queryByLabelText('Location')).not.toBeInTheDocument()
    expect(screen.getByLabelText('Latitude')).toBeInTheDocument()
    expect(screen.getByLabelText('Longitude')).toBeInTheDocument()
    const map = screen.getByTitle('Location map')
    expect(map).toBeInTheDocument()
    expect(screen.getByText('Dock, Pier')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Pick coordinates on map' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Pick coordinates on map' })).toBeInTheDocument()
    expect(screen.getByLabelText('Number of apparent arms')).toBeInTheDocument()
    expect(screen.getByLabelText('Health observation')).toBeInTheDocument()
  })

  it('clicking the contained map populates latitude and longitude', async () => {
    const user = userEvent.setup()
    render(<SingleEntryPage />)

    await screen.findByRole('heading', { name: 'Location' })
    await user.click(screen.getByRole('button', { name: 'Pick coordinates on map' }))
    await user.click(screen.getByLabelText('Location map selector'))

    expect(screen.getByLabelText('Latitude')).toHaveValue(48.5)
    expect(screen.getByLabelText('Longitude')).toHaveValue(-123.25)
  })

  it('submits target info, metadata, and files through the client API', async () => {
    const user = userEvent.setup()
    render(<SingleEntryPage />)

    await screen.findByRole('heading', { name: 'Location' })
    await user.type(screen.getByLabelText('Target ID'), 'q1')
    await user.selectOptions(screen.getByLabelText('Saved locations'), 'Dock')
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
