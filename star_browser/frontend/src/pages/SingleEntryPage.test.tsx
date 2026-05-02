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
      name: 'short_arm_code',
      display_name: 'Short arm coding',
      field_type: 'morphometric_code',
      group: 'short_arm',
      group_display_name: 'Short Arm Coding',
      required: false,
      tooltip: 'Arm positions and severity of short arms',
      min_value: null,
      max_value: null,
      options: [],
      vocabulary: [],
      mobile_widget: 'short_arm_code',
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

  it('shows collapsible single entry instructions with discrete sections', async () => {
    const user = userEvent.setup()
    render(<SingleEntryPage />)

    await screen.findByRole('heading', { name: 'Single Entry' })
    const instructionsToggle = screen.getByText('How to use Single Entry')
    expect(instructionsToggle).toBeVisible()
    expect(screen.getByRole('heading', { name: '1. Target and encounter', hidden: true })).not.toBeVisible()

    await user.click(instructionsToggle)

    expect(screen.getByRole('heading', { name: '1. Target and encounter' })).toBeVisible()
    expect(screen.getByText('Choose Queries or Gallery, choose create/append mode, then enter the target ID.')).toBeVisible()
    expect(screen.getByText('Set the encounter date and optional encounter suffix before submitting.')).toBeVisible()
    expect(screen.getByRole('heading', { name: '2. Location and metadata' })).toBeVisible()
    expect(screen.getByText('Use a saved location or Add new location, then verify latitude/longitude on the map.')).toBeVisible()
    expect(screen.getByText('Fill in the observation metadata fields that apply to this entry.')).toBeVisible()
    expect(screen.getByRole('heading', { name: '3. Images and review' })).toBeVisible()
    expect(screen.getByText('Choose image files from this computer.')).toBeVisible()
    expect(screen.getByText('Review the selected filenames before submitting.')).toBeVisible()
    expect(screen.getByRole('heading', { name: '4. Submit entry' })).toBeVisible()
    expect(screen.getByText('Click Submit entry to archive only after the target, metadata, and selected images look correct.')).toBeVisible()
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
    expect(screen.getByLabelText('Upload images from this computer')).toBeInTheDocument()
    expect(screen.queryByText('Selected image files are uploaded from your browser into the chosen archive ID. There is no server-folder path mode here.')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Submit entry to archive' })).toBeDisabled()
    expect(screen.queryByText(/server folder path/i)).not.toBeInTheDocument()
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

  it('shows a selected-file review before archive submit', async () => {
    const user = userEvent.setup()
    render(<SingleEntryPage />)

    await screen.findByRole('heading', { name: 'Location' })
    const files = [
      new File(['image-a'], 'capture-a.jpg', { type: 'image/jpeg' }),
      new File(['image-b'], 'capture-b.jpg', { type: 'image/jpeg' }),
    ]
    await user.upload(screen.getByLabelText('Upload images from this computer'), files)

    expect(screen.getByRole('heading', { name: 'Review selected image files' })).toBeInTheDocument()
    expect(screen.getByText('2 file(s) selected from this computer.')).toBeInTheDocument()
    expect(screen.getByText('capture-a.jpg')).toBeInTheDocument()
    expect(screen.getByText('capture-b.jpg')).toBeInTheDocument()
  })

  it('renders short arm coding like the desktop app and serializes entries', async () => {
    const user = userEvent.setup()
    render(<SingleEntryPage />)

    await screen.findByRole('heading', { name: 'Short Arm Coding' })
    expect(screen.queryByRole('textbox', { name: 'Short arm coding' })).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: '+ Add short arm' }))
    const position = screen.getByLabelText('Short arm 1 position')
    const severity = screen.getByLabelText('Short arm 1 severity')
    expect(position).toHaveValue(1)
    expect(severity).toHaveValue('very_tiny')

    await user.click(position)
    await user.keyboard('{Control>}a{/Control}7')
    await user.selectOptions(severity, 'small')

    await user.type(screen.getByLabelText('Target ID'), 'q1')
    await user.selectOptions(screen.getByLabelText('Saved locations'), 'Dock')
    const file = new File(['image-bytes'], 'capture.jpg', { type: 'image/jpeg' })
    await user.upload(screen.getByLabelText('Upload images from this computer'), file)
    await user.click(screen.getByRole('button', { name: 'Submit entry to archive' }))

    await waitFor(() => {
      expect(mockedSubmitEntry).toHaveBeenCalledTimes(1)
    })
    expect(mockedSubmitEntry.mock.calls[0][0].metadata.short_arm_code).toBe('small(7)')
  })

  it('requires a location before submitting browser entry upload', async () => {
    const user = userEvent.setup()
    render(<SingleEntryPage />)

    await screen.findByRole('heading', { name: 'Location' })
    await user.type(screen.getByLabelText('Target ID'), 'q1')
    await user.upload(screen.getByLabelText('Upload images from this computer'), new File(['image-bytes'], 'capture.jpg', { type: 'image/jpeg' }))

    expect(screen.getByRole('button', { name: 'Submit entry to archive' })).toBeDisabled()
    expect(screen.getByText('Location is required before upload.')).toBeInTheDocument()
    expect(mockedSubmitEntry).not.toHaveBeenCalled()
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
    await user.upload(screen.getByLabelText('Upload images from this computer'), file)
    await user.click(screen.getByRole('button', { name: 'Submit entry to archive' }))

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
