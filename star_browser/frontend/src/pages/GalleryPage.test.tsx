import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { GalleryPage } from './GalleryPage'

vi.mock('../api/client', () => ({
  getIdReviewEntity: vi.fn(),
  getIdReviewOptions: vi.fn(),
}))

import { getIdReviewEntity, getIdReviewOptions } from '../api/client'

const mockedGetIdReviewEntity = vi.mocked(getIdReviewEntity)
const mockedGetIdReviewOptions = vi.mocked(getIdReviewOptions)

const galleryResponse = {
  archive_type: 'gallery' as const,
  entity_id: 'entity_001',
  metadata_summary: { location: 'Friday Harbor' },
  metadata_rows: [
    { row_index: 1, source: 'gallery_metadata.csv', values: { location: 'Friday Harbor', sex: 'female' } },
    { row_index: 2, source: 'gallery_metadata.csv', values: { location: 'Cattle Point', sex: 'male' } },
  ],
  timeline: [
    { encounter: 'enc_a', date: '2026-04-01', label: 'A — 2026-04-01', image_count: 2, image_labels: ['Image A1', 'Image A2'] },
    { encounter: 'enc_b', date: '2026-04-02', label: 'B — 2026-04-02', image_count: 1, image_labels: ['Image B1'] },
  ],
  encounters: [
    { encounter: 'enc_a', date: '2026-04-01', label: 'A — 2026-04-01' },
    { encounter: 'enc_b', date: '2026-04-02', label: 'B — 2026-04-02' },
  ],
  images: [
    {
      image_id: 'img_a1',
      label: 'Image A1',
      encounter: 'enc_a',
      preview_url: '/preview/a1.jpg',
      fullres_url: '/full/a1.jpg',
    },
    {
      image_id: 'img_a2',
      label: 'Image A2',
      encounter: 'enc_a',
      preview_url: '/preview/a2.jpg',
      fullres_url: '/full/a2.jpg',
    },
    {
      image_id: 'img_b1',
      label: 'Image B1',
      encounter: 'enc_b',
      preview_url: '/preview/b1.jpg',
      fullres_url: '/full/b1.jpg',
    },
  ],
}

const queryOptionsResponse = {
  archive_type: 'query' as const,
  options: [
    { entity_id: 'query_friday_001', label: 'query_friday_001 — Friday Harbor — 2026-04-01', location: 'Friday Harbor', last_observation_date: '2026-04-01', metadata: { location: 'Friday Harbor', sex: 'female' } },
    { entity_id: 'query_cattle_002', label: 'query_cattle_002 — Cattle Point — 2026-04-02', location: 'Cattle Point', last_observation_date: '2026-04-02', metadata: { location: 'Cattle Point', sex: 'male' } },
  ],
}

describe('GalleryPage', () => {
  afterEach(() => {
    cleanup()
  })

  beforeEach(() => {
    mockedGetIdReviewEntity.mockReset()
    mockedGetIdReviewEntity.mockResolvedValue(galleryResponse)
    mockedGetIdReviewOptions.mockReset()
    mockedGetIdReviewOptions.mockResolvedValue(queryOptionsResponse)
  })

  it('is labeled ID Review and lets users choose query or gallery IDs', async () => {
    const user = userEvent.setup()
    render(<GalleryPage />)

    expect(screen.getByRole('heading', { name: 'ID Review' })).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Gallery Review' })).not.toBeInTheDocument()
    expect(screen.getByLabelText('Review ID type')).toHaveValue('query')
    expect(screen.queryByPlaceholderText('Enter query or gallery ID')).not.toBeInTheDocument()
    expect(await screen.findByRole('listbox', { name: 'Available IDs' })).toBeInTheDocument()

    await user.selectOptions(screen.getByLabelText('Review ID type'), 'gallery')

    expect(mockedGetIdReviewOptions).toHaveBeenCalledWith('gallery')
  })

  it('loads query IDs selected from the available ID list', async () => {
    const user = userEvent.setup()
    render(<GalleryPage />)

    await screen.findByRole('option', { name: 'query_friday_001 — Friday Harbor — 2026-04-01' })
    await user.click(screen.getByRole('option', { name: 'query_friday_001 — Friday Harbor — 2026-04-01' }))
    await user.click(screen.getByRole('button', { name: 'Load ID' }))

    expect(mockedGetIdReviewEntity).toHaveBeenCalledWith('query', 'query_friday_001')
  })

  it('lets users search and filter a scrollable ID combo box before loading an ID', async () => {
    const user = userEvent.setup()
    render(<GalleryPage />)

    expect(await screen.findByText('Available IDs')).toBeInTheDocument()
    expect(mockedGetIdReviewOptions).toHaveBeenCalledWith('query')
    expect(screen.getByRole('listbox', { name: 'Available IDs' })).toBeInTheDocument()
    expect(screen.getByText('query_friday_001 — Friday Harbor — 2026-04-01')).toBeInTheDocument()
    expect(screen.getByText('query_cattle_002 — Cattle Point — 2026-04-02')).toBeInTheDocument()

    await user.type(screen.getByLabelText('Search IDs'), 'friday')
    expect(screen.getByText('query_friday_001 — Friday Harbor — 2026-04-01')).toBeInTheDocument()
    expect(screen.queryByText('query_cattle_002 — Cattle Point — 2026-04-02')).not.toBeInTheDocument()

    await user.clear(screen.getByLabelText('Search IDs'))
    await user.selectOptions(screen.getByLabelText('Location filter'), 'Cattle Point')
    expect(screen.queryByText('query_friday_001 — Friday Harbor — 2026-04-01')).not.toBeInTheDocument()
    expect(screen.getByText('query_cattle_002 — Cattle Point — 2026-04-02')).toBeInTheDocument()

    await user.selectOptions(screen.getByLabelText('Location filter'), '__all__')
    await user.type(screen.getByLabelText('Observed from'), '2026-04-02')
    expect(screen.queryByText('query_friday_001 — Friday Harbor — 2026-04-01')).not.toBeInTheDocument()
    expect(screen.getByText('query_cattle_002 — Cattle Point — 2026-04-02')).toBeInTheDocument()

    await user.click(screen.getByRole('option', { name: 'query_cattle_002 — Cattle Point — 2026-04-02' }))
    expect(screen.queryByPlaceholderText('Enter query or gallery ID')).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Load ID' }))
    expect(mockedGetIdReviewEntity).toHaveBeenCalledWith('query', 'query_cattle_002')
  })

  it('shows images above metadata rows and timeline for the selected ID', async () => {
    const user = userEvent.setup()
    render(<GalleryPage />)

    await screen.findByRole('option', { name: 'query_friday_001 — Friday Harbor — 2026-04-01' })
    await user.click(screen.getByRole('option', { name: 'query_friday_001 — Friday Harbor — 2026-04-01' }))
    await user.click(screen.getByRole('button', { name: 'Load ID' }))

    const imagesHeading = await screen.findByRole('heading', { name: 'Images' })
    const metadataHeading = screen.getByRole('heading', { name: 'Metadata' })
    const timelineHeading = screen.getByRole('heading', { name: 'Timeline' })
    expect(imagesHeading.compareDocumentPosition(metadataHeading) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(imagesHeading.compareDocumentPosition(timelineHeading) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(screen.getByText('Latest metadata')).toBeInTheDocument()
    expect(screen.getByText('All metadata rows')).toBeInTheDocument()
    expect(screen.getByText('Row 1 · gallery_metadata.csv')).toBeInTheDocument()
    expect(screen.getAllByText(/sex:/).length).toBeGreaterThan(0)
    expect(screen.getByText('female')).toBeInTheDocument()
    expect(screen.getByText('2026-04-01')).toBeInTheDocument()
    expect(screen.getByText('2 images')).toBeInTheDocument()
    expect(screen.getByText('Image A1, Image A2')).toBeInTheDocument()
  })

  it('lets users zoom, pan, rotate, and reset the selected ID image', async () => {
    const user = userEvent.setup()
    const addEventListenerSpy = vi.spyOn(window, 'addEventListener')
    render(<GalleryPage />)

    await screen.findByRole('option', { name: 'query_friday_001 — Friday Harbor — 2026-04-01' })
    await user.click(screen.getByRole('option', { name: 'query_friday_001 — Friday Harbor — 2026-04-01' }))
    await user.click(screen.getByRole('button', { name: 'Load ID' }))

    const viewer = await screen.findByLabelText('Interactive image viewer')
    const image = screen.getByRole('img', { name: 'Image A1' })
    expect(screen.getByText('Wheel to zoom. Drag to pan. Hold R and drag to rotate.')).toBeInTheDocument()
    expect(addEventListenerSpy).toHaveBeenCalledWith('wheel', expect.any(Function), expect.objectContaining({ capture: true, passive: false }))

    vi.spyOn(viewer, 'getBoundingClientRect').mockReturnValue({
      x: 10,
      y: 10,
      left: 10,
      top: 10,
      right: 510,
      bottom: 410,
      width: 500,
      height: 400,
      toJSON: () => ({}),
    })
    const wheelEvent = new WheelEvent('wheel', { clientX: 100, clientY: 100, deltaY: -300, bubbles: true, cancelable: true })
    window.dispatchEvent(wheelEvent)
    expect(wheelEvent.defaultPrevented).toBe(true)
    await waitFor(() => {
      expect(image).toHaveStyle({ transform: 'translate(0px, 0px) rotate(0deg) scale(1.3)' })
    })

    fireEvent.mouseDown(viewer, { clientX: 100, clientY: 100 })
    fireEvent.mouseMove(window, { clientX: 130, clientY: 120 })
    fireEvent.mouseUp(window)
    expect(image).toHaveStyle({ transform: 'translate(30px, 20px) rotate(0deg) scale(1.3)' })

    fireEvent.keyDown(window, { key: 'r' })
    fireEvent.mouseDown(viewer, { clientX: 100, clientY: 100 })
    fireEvent.mouseMove(window, { clientX: 130, clientY: 100 })
    fireEvent.mouseUp(window)
    fireEvent.keyUp(window, { key: 'r' })
    expect(image).toHaveStyle({ transform: 'translate(30px, 20px) rotate(9deg) scale(1.3)' })

    await user.click(screen.getByRole('button', { name: 'Reset image view' }))
    expect(image).toHaveStyle({ transform: 'translate(0px, 0px) rotate(0deg) scale(1)' })
  })

  it('filters the image list by encounter', async () => {
    const user = userEvent.setup()
    render(<GalleryPage />)

    await screen.findByRole('option', { name: 'query_friday_001 — Friday Harbor — 2026-04-01' })
    await user.click(screen.getByRole('option', { name: 'query_friday_001 — Friday Harbor — 2026-04-01' }))
    await user.click(screen.getByRole('button', { name: 'Load ID' }))

    await screen.findByRole('img', { name: 'Image A1' })
    await user.selectOptions(screen.getByLabelText('Encounter filter'), 'enc_b')

    await waitFor(() => {
      expect(screen.queryByText('Image A1')).not.toBeInTheDocument()
    })
    expect(screen.queryByText('Image A2')).not.toBeInTheDocument()
    expect(screen.getByRole('img', { name: 'Image B1' })).toBeInTheDocument()
  })

  it('resets the selected image to the first filtered result when the encounter filter changes', async () => {
    const user = userEvent.setup()
    render(<GalleryPage />)

    await screen.findByRole('option', { name: 'query_friday_001 — Friday Harbor — 2026-04-01' })
    await user.click(screen.getByRole('option', { name: 'query_friday_001 — Friday Harbor — 2026-04-01' }))
    await user.click(screen.getByRole('button', { name: 'Load ID' }))

    await screen.findByRole('img', { name: 'Image A1' })
    await user.click(screen.getByRole('button', { name: /Image A2/i }))
    expect(screen.getByRole('img')).toHaveAttribute('src', '/preview/a2.jpg')

    await user.selectOptions(screen.getByLabelText('Encounter filter'), 'enc_b')

    await waitFor(() => {
      expect(screen.getByRole('img')).toHaveAttribute('src', '/preview/b1.jpg')
    })
  })
})
