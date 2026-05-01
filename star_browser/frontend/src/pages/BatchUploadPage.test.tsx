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

function makeZipFile(paths: string[], name = 'bundle.zip') {
  const chunks: Uint8Array[] = []
  const centralChunks: Uint8Array[] = []
  let offset = 0
  const encoder = new TextEncoder()
  const pushChunk = (chunk: Uint8Array) => {
    chunks.push(chunk)
    offset += chunk.length
  }
  const writeU16 = (view: DataView, pos: number, value: number) => view.setUint16(pos, value, true)
  const writeU32 = (view: DataView, pos: number, value: number) => view.setUint32(pos, value, true)

  for (const path of paths) {
    const nameBytes = encoder.encode(path)
    const localOffset = offset
    const local = new Uint8Array(30 + nameBytes.length)
    const localView = new DataView(local.buffer)
    writeU32(localView, 0, 0x04034b50)
    writeU16(localView, 4, 20)
    writeU16(localView, 26, nameBytes.length)
    local.set(nameBytes, 30)
    pushChunk(local)

    const central = new Uint8Array(46 + nameBytes.length)
    const centralView = new DataView(central.buffer)
    writeU32(centralView, 0, 0x02014b50)
    writeU16(centralView, 4, 20)
    writeU16(centralView, 6, 20)
    writeU16(centralView, 28, nameBytes.length)
    writeU32(centralView, 42, localOffset)
    central.set(nameBytes, 46)
    centralChunks.push(central)
  }

  const centralOffset = offset
  let centralSize = 0
  for (const central of centralChunks) {
    pushChunk(central)
    centralSize += central.length
  }
  const eocd = new Uint8Array(22)
  const eocdView = new DataView(eocd.buffer)
  writeU32(eocdView, 0, 0x06054b50)
  writeU16(eocdView, 8, paths.length)
  writeU16(eocdView, 10, paths.length)
  writeU32(eocdView, 12, centralSize)
  writeU32(eocdView, 16, centralOffset)
  chunks.push(eocd)

  const file = new File(chunks, name, { type: 'application/zip' })
  Object.defineProperty(file, 'arrayBuffer', {
    value: async () => {
      const out = new Uint8Array(chunks.reduce((total, chunk) => total + chunk.length, 0))
      let pos = 0
      for (const chunk of chunks) {
        out.set(chunk, pos)
        pos += chunk.length
      }
      return out.buffer
    },
  })
  return file
}

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
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  async function stageAndDiscover(user: ReturnType<typeof userEvent.setup>) {
    await user.click(screen.getByLabelText('Use zip upload'))
    const zip = makeZipFile(['trip_upload/anchovy/04_21_26/a.jpg', 'trip_upload/anchovy/04_21_26/b.jpg'])
    await user.upload(screen.getByLabelText('Source zip bundle'), zip)
    await user.click(screen.getByRole('button', { name: 'Test zip structure' }))
    await screen.findByText(/Zip structure looks valid/i)
    await user.click(screen.getByRole('button', { name: 'Prepare zip for preview' }))
    await screen.findByText(/Files ready for preview:/)
    await user.click(screen.getByRole('button', { name: 'Preview IDs and metadata' }))
    await screen.findByRole('heading', { name: 'Review selected IDs and metadata' })
    expect(screen.getByRole('heading', { name: '3. Submit IDs' })).toBeInTheDocument()
  }

  it('uses preview-first wording and reserves upload wording for the final push', async () => {
    render(<BatchUploadPage />)

    expect(screen.getByRole('heading', { name: '3. Submit IDs' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Submit selected IDs for upload' })).toBeDisabled()
    expect(screen.queryByRole('heading', { name: '2. Discover IDs' })).not.toBeInTheDocument()
  })

  it('shows collapsible stepwise batch upload instructions at the top', async () => {
    const user = userEvent.setup()
    render(<BatchUploadPage />)

    const instructionsToggle = screen.getByText('How to use Batch Upload')
    expect(instructionsToggle).toBeVisible()
    expect(screen.getByText('Choose the source: use Server folder path when the images are already on this starBoard machine; use Upload zip only when you need to bring files in from another computer.')).not.toBeVisible()

    await user.click(instructionsToggle)

    expect(screen.getByText('Choose the source: use Server folder path when the images are already on this starBoard machine; use Upload zip only when you need to bring files in from another computer.')).toBeVisible()
    expect(screen.getByText('Use Auto discovery for normal batches. Use Flat for ID / images folders, With Encounters for ID / date / images folders, and Grouped for group / ID / date / images field exports.')).toBeVisible()
    expect(screen.getByText('For zip sources, click Test zip structure first, then Prepare zip for preview; this catches root-level images or mismatched folder layouts before anything is uploaded.')).toBeVisible()
    expect(screen.getByText('Preview IDs and metadata to build the review table. This is still read-only: it does not write images, metadata, or IDs into Gallery or Queries.')).toBeVisible()
    expect(screen.getByText('Review every row: target ID, create-vs-append action, encounter date/suffix, image count, sample filenames, warnings, and whether the row is selected.')).toBeVisible()
    expect(screen.getByText('Submit selected IDs only after the review table looks correct; this final step writes the selected rows into the chosen archive.')).toBeVisible()
  })

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

    await user.click(screen.getByRole('button', { name: 'Preview IDs and metadata' }))

    expect(mockedDiscoverBatchUpload).toHaveBeenCalledWith(expect.objectContaining({
      import_source: { type: 'server_path', path: '/data/trip_upload' },
    }))
  })

  it('tests zip source structure locally before upload is allowed', async () => {
    const user = userEvent.setup()
    render(<BatchUploadPage />)

    await user.click(screen.getByLabelText('Use zip upload'))
    const zip = makeZipFile(['trip_upload/anchovy/04_21_26/a.jpg', 'trip_upload/anchovy/04_21_26/b.jpg'])
    await user.upload(screen.getByLabelText('Source zip bundle'), zip)

    expect(screen.getByRole('button', { name: 'Prepare zip for preview' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'Test zip structure' }))

    expect(await screen.findByText(/Zip structure looks valid/i)).toBeInTheDocument()
    expect(screen.getByText(/Resolved mode:/i)).toBeInTheDocument()
    expect(screen.getByText(/Importable images:/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Prepare zip for preview' })).toBeEnabled()
    expect(mockedUploadBatchZip).not.toHaveBeenCalled()
  })

  it('accepts a single-ID zip root during local structure testing', async () => {
    const user = userEvent.setup()
    render(<BatchUploadPage />)

    await user.click(screen.getByLabelText('Use zip upload'))
    const zip = makeZipFile(['anchovy/a.jpg', 'anchovy/b.jpg'])
    await user.upload(screen.getByLabelText('Source zip bundle'), zip)
    await user.click(screen.getByRole('button', { name: 'Test zip structure' }))

    expect(await screen.findByText(/Zip structure looks valid/i)).toBeInTheDocument()
    expect(screen.getByText(/root entries: anchovy/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Prepare zip for preview' })).toBeEnabled()
  })

  it('shows elapsed time while an operation is running and success after upload', async () => {
    const user = userEvent.setup()
    let resolveUpload: (value: Awaited<ReturnType<typeof uploadBatchZip>>) => void = () => undefined
    mockedUploadBatchZip.mockReturnValue(new Promise((resolve) => { resolveUpload = resolve }))
    render(<BatchUploadPage />)

    await user.click(screen.getByLabelText('Use zip upload'))
    const zip = makeZipFile(['trip_upload/anchovy/a.jpg'])
    await user.upload(screen.getByLabelText('Source zip bundle'), zip)
    await user.click(screen.getByRole('button', { name: 'Test zip structure' }))
    await screen.findByText(/Zip structure looks valid/i)
    await user.click(screen.getByRole('button', { name: 'Prepare zip for preview' }))
    const status = await screen.findByRole('status')
    expect(status).toHaveTextContent(/Preparing preview source/i)
    await waitFor(() => expect(screen.getByText(/Elapsed: [1-9]\d*s/i)).toBeInTheDocument(), { timeout: 2500 })

    resolveUpload({ upload_token: 'upload_123', file_count: 1, root_entries: ['anchovy'] })
    await screen.findByText(/Preview source ready/i)
    expect(screen.getByText(/Files ready for preview:/)).toBeInTheDocument()
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
    await user.click(screen.getByRole('button', { name: 'Submit selected IDs for upload' }))

    expect(confirmSpy).toHaveBeenCalledWith(expect.stringContaining('already exist'))
    expect(mockedExecuteBatchUpload).not.toHaveBeenCalled()
  })

  it('invalidates a discovered plan when discovery settings change', async () => {
    const user = userEvent.setup()
    render(<BatchUploadPage />)

    await stageAndDiscover(user)
    expect(screen.getByRole('button', { name: 'Submit selected IDs for upload' })).toBeEnabled()

    await user.type(screen.getByLabelText('ID prefix'), 'new_')

    expect(screen.getByRole('button', { name: 'Submit selected IDs for upload' })).toBeDisabled()
    expect(screen.getByText(/Settings changed. Preview IDs and metadata again before submitting IDs for upload/)).toBeInTheDocument()
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

    await user.click(screen.getByRole('button', { name: 'Submit selected IDs for upload' }))

    await screen.findByText('Row results')
    const results = screen.getByRole('table', { name: 'Batch upload row results' })
    expect(within(results).getByText('anchovy')).toBeInTheDocument()
    expect(within(results).getByText('04_21_26')).toBeInTheDocument()
    expect(within(results).getByText('2')).toBeInTheDocument()
  })
})
