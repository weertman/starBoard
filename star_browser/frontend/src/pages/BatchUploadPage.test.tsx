import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, waitFor, within } from '@testing-library/react'
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

vi.mock('../api/client', () => ({
  uploadBatchZip: vi.fn(),
  discoverBatchUpload: vi.fn(),
  executeBatchUpload: vi.fn(),
  previewBatchServerPath: vi.fn(),
  getLocationSites: vi.fn(),
}))

import { BatchUploadPage } from './BatchUploadPage'
import { discoverBatchUpload, executeBatchUpload, getLocationSites, previewBatchServerPath, uploadBatchZip } from '../api/client'

const mockedUploadBatchZip = vi.mocked(uploadBatchZip)
const mockedDiscoverBatchUpload = vi.mocked(discoverBatchUpload)
const mockedExecuteBatchUpload = vi.mocked(executeBatchUpload)
const mockedPreviewBatchServerPath = vi.mocked(previewBatchServerPath)
const mockedGetLocationSites = vi.mocked(getLocationSites)

const baseDiscoverResponse = {
  plan_id: 'plan_123',
  target_archive: 'gallery' as const,
  requested_discovery_mode: 'flat' as const,
  resolved_discovery_mode: 'flat' as const,
  summary: {
    detected_rows: 1,
    detected_ids: 1,
    total_images: 2,
    new_ids: 1,
    existing_ids: 0,
    warnings: 0,
    errors: 0,
  },
  rows: [
    {
      row_id: 'row_001',
      original_detected_id: 'anchovy',
      transformed_target_id: 'anchovy',
      action: 'create_new' as const,
      target_exists: false,
      group_name: null,
      encounter_folder_name: null,
      encounter_date: null,
      encounter_suffix: null,
      image_count: 2,
      sample_labels: ['a.jpg', 'b.jpg'],
      source_ref: 'anchovy',
      warnings: [],
    },
  ],
  warnings: [],
  errors: [],
}

describe('BatchUploadPage', () => {
  beforeEach(() => {
    mockedUploadBatchZip.mockReset()
    mockedDiscoverBatchUpload.mockReset()
    mockedExecuteBatchUpload.mockReset()
    mockedPreviewBatchServerPath.mockReset()
    mockedGetLocationSites.mockReset()
    mockedGetLocationSites.mockResolvedValue({
      sites: [
        { name: 'Dock', latitude: 48.546, longitude: -123.013 },
        { name: 'Pier', latitude: 48.5, longitude: -123.2 },
      ],
    })
    mockedUploadBatchZip.mockResolvedValue({
      upload_token: 'upload_123',
      file_count: 2,
      root_entries: ['anchovy'],
    })
    mockedExecuteBatchUpload.mockResolvedValue({
      status: 'ok',
      plan_id: 'plan_123',
      batch_id: 'batch_123',
      target_archive: 'gallery',
      summary: {
        executed_rows: 1,
        created_ids: 1,
        appended_ids: 0,
        accepted_images: 2,
        skipped_images: 0,
        rows_with_errors: 0,
      },
      rows: [
        {
          row_id: 'row_001',
          target_id: 'anchovy',
          action: 'create_new',
          accepted_images: 2,
          skipped_images: 0,
          encounter_folder: '04_21_26',
          archive_paths_written: ['anchovy/04_21_26/a.jpg'],
          warnings: [],
          errors: [],
        },
      ],
      message: 'Batch upload completed.',
    })
    mockedDiscoverBatchUpload.mockResolvedValue(baseDiscoverResponse)
    mockedPreviewBatchServerPath.mockResolvedValue({
      path: '/data/trip_upload',
      exists: true,
      is_directory: true,
      resolved_discovery_mode: 'encounters',
      immediate_entries: ['anchovy'],
      importable_images: 2,
    })
    vi.spyOn(window, 'confirm').mockReturnValue(true)
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  async function stageAndDiscover(user: ReturnType<typeof userEvent.setup>) {
    const zip = new File(['zip-bytes'], 'bundle.zip', { type: 'application/zip' })
    await user.upload(screen.getByLabelText('Source zip bundle'), zip)
    await user.click(screen.getByRole('button', { name: 'Upload zip' }))
    await screen.findByText(/Files staged:/)
    await user.click(screen.getByRole('button', { name: 'Discover IDs' }))
    await screen.findByRole('heading', { name: '3. Preview plan' })
  }

  it('shows source structure guidance and discovers from a validated server folder path', async () => {
    const user = userEvent.setup()
    render(<BatchUploadPage />)

    expect(screen.getByText(/Accepted source layouts/)).toBeInTheDocument()
    expect(screen.getAllByText('ID / date / images')[0]).toBeInTheDocument()

    await user.click(screen.getByLabelText('Select server folder path source'))
    await user.type(screen.getByRole('textbox', { name: 'Server folder path' }), '/data/trip_upload')
    await user.click(screen.getByRole('button', { name: 'Preview server path' }))

    await screen.findByText(/Server path ready/)
    expect(mockedPreviewBatchServerPath).toHaveBeenCalledWith({ path: '/data/trip_upload', discovery_mode: 'auto' })

    await user.click(screen.getByRole('button', { name: 'Discover IDs' }))

    expect(mockedDiscoverBatchUpload).toHaveBeenCalledWith(expect.objectContaining({
      import_source: { type: 'server_path', path: '/data/trip_upload' },
    }))
  })

  it('hides flat-only encounter controls outside flat mode and shows the today warning in flat mode', async () => {
    const user = userEvent.setup()
    render(<BatchUploadPage />)

    expect(screen.getByLabelText('Flat encounter date')).toBeInTheDocument()
    expect(screen.getByText("Today's date selected")).toBeInTheDocument()

    await user.selectOptions(screen.getByLabelText('Discovery mode'), 'encounters')

    expect(screen.queryByLabelText('Flat encounter date')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('Flat encounter suffix')).not.toBeInTheDocument()
  })

  it('sends batch location values in the discover request', async () => {
    const user = userEvent.setup()
    render(<BatchUploadPage />)

    await user.selectOptions(await screen.findByLabelText('Saved locations'), 'Dock')
    await user.type(screen.getByLabelText('Latitude'), '48.5')
    await user.type(screen.getByLabelText('Longitude'), '-123.1')
    await stageAndDiscover(user)

    expect(mockedDiscoverBatchUpload).toHaveBeenCalledWith(expect.objectContaining({
      batch_location: { location: 'Dock', latitude: '48.5', longitude: '-123.1' },
    }))
  })

  it('renders batch location entry like Single Entry with saved locations, add-new, and map picking', async () => {
    const user = userEvent.setup()
    render(<BatchUploadPage />)

    const savedLocations = await screen.findByLabelText('Saved locations') as HTMLSelectElement
    expect(savedLocations.tagName).toBe('SELECT')
    expect(savedLocations.options[0].textContent).toBe('Add new…')
    expect(savedLocations.options[0].value).toBe('__new__')
    expect(screen.getByRole('button', { name: 'Add new location' })).toBeInTheDocument()
    expect(screen.queryByLabelText('Batch location')).not.toBeInTheDocument()
    expect(screen.getByLabelText('Latitude')).toBeInTheDocument()
    expect(screen.getByLabelText('Longitude')).toBeInTheDocument()
    expect(screen.getByTitle('Location map')).toBeInTheDocument()
    expect(screen.getByText('Dock, Pier')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Add new location' }))
    await user.type(screen.getByLabelText('Location'), 'New Reef')
    expect(screen.getByLabelText('Location')).toHaveValue('New Reef')

    await user.click(screen.getByRole('button', { name: 'Pick coordinates on map' }))
    await user.click(screen.getByLabelText('Location map selector'))
    expect(screen.getByLabelText('Latitude')).toHaveValue(48.5)
    expect(screen.getByLabelText('Longitude')).toHaveValue(-123.25)
  })

  it('requires confirmation before appending to existing IDs', async () => {
    const user = userEvent.setup()
    mockedDiscoverBatchUpload.mockResolvedValue({
      ...baseDiscoverResponse,
      summary: { ...baseDiscoverResponse.summary, new_ids: 0, existing_ids: 1 },
      rows: [
        {
          ...baseDiscoverResponse.rows[0],
          action: 'append_existing',
          target_exists: true,
        },
      ],
    })
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(false)
    render(<BatchUploadPage />)

    await stageAndDiscover(user)
    await user.click(screen.getByRole('button', { name: 'Start batch upload' }))

    expect(confirmSpy).toHaveBeenCalledWith(expect.stringContaining('already exist'))
    expect(mockedExecuteBatchUpload).not.toHaveBeenCalled()
  })

  it('invalidates a discovered plan when discovery settings change', async () => {
    const user = userEvent.setup()
    render(<BatchUploadPage />)

    await stageAndDiscover(user)
    expect(screen.getByRole('button', { name: 'Start batch upload' })).toBeEnabled()

    await user.type(screen.getByLabelText('ID prefix'), 'new_')

    expect(screen.queryByRole('button', { name: 'Start batch upload' })).not.toBeInTheDocument()
    expect(screen.getByText(/Settings changed. Rediscover IDs before executing/)).toBeInTheDocument()
  })

  it('renders discover warnings and row-level execute results', async () => {
    const user = userEvent.setup()
    mockedDiscoverBatchUpload.mockResolvedValue({
      ...baseDiscoverResponse,
      summary: { ...baseDiscoverResponse.summary, warnings: 1 },
      warnings: [{ code: 'source_warning', message: 'Some files were ignored', row_id: null }],
    })
    render(<BatchUploadPage />)

    await stageAndDiscover(user)
    expect(screen.getByText('Some files were ignored')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Start batch upload' }))

    await screen.findByText('Row results')
    const results = screen.getByRole('table', { name: 'Batch upload row results' })
    expect(within(results).getByText('anchovy')).toBeInTheDocument()
    expect(within(results).getByText('04_21_26')).toBeInTheDocument()
    expect(within(results).getByText('2')).toBeInTheDocument()
  })
})
