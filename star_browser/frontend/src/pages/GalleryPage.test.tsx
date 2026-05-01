import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { GalleryPage } from './GalleryPage'

vi.mock('../api/client', () => ({
  getIdReviewEntity: vi.fn(),
}))

import { getIdReviewEntity } from '../api/client'

const mockedGetIdReviewEntity = vi.mocked(getIdReviewEntity)

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

describe('GalleryPage', () => {
  afterEach(() => {
    cleanup()
  })

  beforeEach(() => {
    mockedGetIdReviewEntity.mockReset()
    mockedGetIdReviewEntity.mockResolvedValue(galleryResponse)
  })

  it('is labeled ID Review and lets users choose query or gallery IDs', async () => {
    const user = userEvent.setup()
    render(<GalleryPage />)

    expect(screen.getByRole('heading', { name: 'ID Review' })).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Gallery Review' })).not.toBeInTheDocument()
    expect(screen.getByLabelText('Review ID type')).toHaveValue('query')
    expect(screen.getByPlaceholderText('Enter query or gallery ID')).toBeInTheDocument()

    await user.selectOptions(screen.getByLabelText('Review ID type'), 'gallery')
    await user.type(screen.getByPlaceholderText('Enter query or gallery ID'), 'entity_001')
    await user.click(screen.getByRole('button', { name: 'Load ID' }))

    expect(mockedGetIdReviewEntity).toHaveBeenCalledWith('gallery', 'entity_001')
  })

  it('loads query IDs through ID Review', async () => {
    const user = userEvent.setup()
    render(<GalleryPage />)

    await user.type(screen.getByPlaceholderText('Enter query or gallery ID'), 'query_001')
    await user.click(screen.getByRole('button', { name: 'Load ID' }))

    expect(mockedGetIdReviewEntity).toHaveBeenCalledWith('query', 'query_001')
  })

  it('shows images, metadata rows, and timeline for the selected ID', async () => {
    const user = userEvent.setup()
    render(<GalleryPage />)

    await user.type(screen.getByPlaceholderText('Enter query or gallery ID'), 'entity_001')
    await user.click(screen.getByRole('button', { name: 'Load ID' }))

    expect(await screen.findByRole('heading', { name: 'Images' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Metadata' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Timeline' })).toBeInTheDocument()
    expect(screen.getByText('Latest metadata')).toBeInTheDocument()
    expect(screen.getByText('All metadata rows')).toBeInTheDocument()
    expect(screen.getByText('Row 1 · gallery_metadata.csv')).toBeInTheDocument()
    expect(screen.getAllByText(/sex:/).length).toBeGreaterThan(0)
    expect(screen.getByText('female')).toBeInTheDocument()
    expect(screen.getByText('2026-04-01')).toBeInTheDocument()
    expect(screen.getByText('2 images')).toBeInTheDocument()
    expect(screen.getByText('Image A1, Image A2')).toBeInTheDocument()
  })

  it('filters the image list by encounter', async () => {
    const user = userEvent.setup()
    render(<GalleryPage />)

    await user.type(screen.getByPlaceholderText('Enter query or gallery ID'), 'entity_001')
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

    await user.type(screen.getByPlaceholderText('Enter query or gallery ID'), 'entity_001')
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
