import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { FirstOrderPage } from './FirstOrderPage'

vi.mock('../api/client', () => ({
  getFirstOrderQueries: vi.fn(),
  getFirstOrderMedia: vi.fn(),
  runFirstOrderSearch: vi.fn(),
}))

import { getFirstOrderMedia, getFirstOrderQueries, runFirstOrderSearch } from '../api/client'

const mockedGetFirstOrderQueries = vi.mocked(getFirstOrderQueries)
const mockedGetFirstOrderMedia = vi.mocked(getFirstOrderMedia)
const mockedRunFirstOrderSearch = vi.mocked(runFirstOrderSearch)

const queryOptions = [
  {
    query_id: 'query_a',
    state: 'pinned' as const,
    last_observation_date: '2026-01-02',
    last_location: 'Eagle Point',
    easy_match_score: 0.75,
    quality: { madreporite_visibility: 0.75, anus_visibility: null, postural_visibility: 1 },
  },
  {
    query_id: 'query_b',
    state: 'attempted' as const,
    last_observation_date: '2026-01-03',
    last_location: 'Cattle Point',
    easy_match_score: 0.5,
    quality: { madreporite_visibility: null, anus_visibility: null, postural_visibility: null },
  },
  {
    query_id: 'query_matched',
    state: 'matched' as const,
    last_observation_date: '2026-01-01',
    last_location: 'Friday Harbor',
    easy_match_score: 1,
    quality: { madreporite_visibility: 1, anus_visibility: 1, postural_visibility: 1 },
  },
]

describe('FirstOrderPage query selector', () => {
  afterEach(() => {
    cleanup()
  })

  beforeEach(() => {
    mockedGetFirstOrderQueries.mockReset()
    mockedGetFirstOrderMedia.mockReset()
    mockedRunFirstOrderSearch.mockReset()
    mockedGetFirstOrderQueries.mockResolvedValue({ queries: queryOptions })
    mockedGetFirstOrderMedia.mockImplementation(async (targetType, entityId) => ({
      target_type: targetType,
      entity_id: entityId,
      images: [0, 1].map((index) => ({
        image_id: `${targetType}:${entityId}:${index}`,
        label: index === 0 ? `${entityId}.jpg` : `${entityId}_detail.jpg`,
        encounter: index === 0 ? '01_02_26_a' : '01_03_26_b',
        preview_url: `/preview/${targetType}/${entityId}_${index}.jpg`,
        fullres_url: `/full/${targetType}/${entityId}_${index}.jpg`,
        is_best: index === 0,
      })),
    }))
    mockedRunFirstOrderSearch.mockResolvedValue({
      query_id: 'query_b',
      preset: 'all',
      candidates: [{ entity_id: 'gallery_1', score: 0.9, k_contrib: 2, field_breakdown: { location: 1 } }],
    })
  })

  it('is labeled Query Matcher instead of First-order Search', () => {
    render(<FirstOrderPage />)

    expect(screen.getByRole('heading', { name: 'Query Matcher' })).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'First-order Search' })).not.toBeInTheDocument()
  })

  it('loads query options with desktop-style state and quality indicators', async () => {
    render(<FirstOrderPage />)

    await screen.findByDisplayValue('query_a')

    expect(screen.getAllByText('Pinned').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Attempted').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Matched').length).toBeGreaterThan(0)
    expect(screen.getByText(/3 of 3 queries shown/i)).toBeInTheDocument()
    expect(screen.getAllByText('●').length).toBeGreaterThan(0)
  })

  it('filters query options by contains search and uses the typed query for search', async () => {
    const user = userEvent.setup()
    render(<FirstOrderPage />)

    const queryInput = await screen.findByLabelText('Query')
    await user.clear(queryInput)
    await user.type(queryInput, 'matched')

    expect(screen.getByText(/1 of 3 queries shown/i)).toBeInTheDocument()
    expect(screen.getByText('query_matched')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Search' }))

    await waitFor(() => {
      expect(mockedRunFirstOrderSearch).toHaveBeenCalledWith({ query_id: 'query_matched', top_k: 10, preset: 'all' })
    })
  })

  it('supports previous and next query navigation', async () => {
    const user = userEvent.setup()
    render(<FirstOrderPage />)

    await screen.findByDisplayValue('query_a')
    await user.click(screen.getByTitle('Previous query in list'))
    expect(screen.getByLabelText('Query')).toHaveValue('query_b')
    await user.click(screen.getByTitle('Next query in list'))
    expect(screen.getByLabelText('Query')).toHaveValue('query_a')
  })

  it('renders every matching query row inside the scrollable selector', async () => {
    const manyOptions = Array.from({ length: 35 }, (_, index) => ({
      query_id: `query_${String(index).padStart(2, '0')}`,
      state: 'not_attempted' as const,
      last_observation_date: '2026-01-01',
      last_location: 'Friday Harbor',
      easy_match_score: 0,
      quality: { madreporite_visibility: null, anus_visibility: null, postural_visibility: null },
    }))
    mockedGetFirstOrderQueries.mockResolvedValueOnce({ queries: manyOptions })

    render(<FirstOrderPage />)

    await screen.findByDisplayValue('query_00')

    expect(screen.getByText(/35 of 35 queries shown/i)).toBeInTheDocument()
    expect(screen.queryByText(/rows rendered/i)).not.toBeInTheDocument()
    expect(screen.getByText('query_34')).toBeInTheDocument()
  })

  it('filters visible queries with state and quality controls', async () => {
    const user = userEvent.setup()
    render(<FirstOrderPage />)

    await screen.findByDisplayValue('query_a')
    await user.click(screen.getByLabelText('Matched'))

    expect(screen.getByText(/1 of 3 queries shown/i)).toBeInTheDocument()
    expect(screen.getByText('query_matched')).toBeInTheDocument()
    expect(screen.queryByText('query_a')).not.toBeInTheDocument()

    await user.click(screen.getByLabelText('Matched'))
    await user.click(screen.getByLabelText('With any quality marker'))

    expect(screen.getByText(/2 of 3 queries shown/i)).toBeInTheDocument()
    expect(screen.getAllByText('query_a').length).toBeGreaterThan(0)
    expect(screen.getByText('query_matched')).toBeInTheDocument()
    expect(screen.queryByText('query_b')).not.toBeInTheDocument()
  })

  it('filters by date range and last location, then rank-orders by date or existing easy match', async () => {
    const user = userEvent.setup()
    render(<FirstOrderPage />)

    await screen.findByDisplayValue('query_a')

    await user.type(screen.getByLabelText('Observed from'), '2026-01-02')
    await user.type(screen.getByLabelText('Observed through'), '2026-01-03')
    await user.selectOptions(screen.getByLabelText('Last location'), 'Cattle Point')

    expect(screen.getByText(/1 of 3 queries shown/i)).toBeInTheDocument()
    expect(screen.getByText('query_b')).toBeInTheDocument()
    expect(screen.queryByText('query_a')).not.toBeInTheDocument()

    await user.selectOptions(screen.getByLabelText('Last location'), '')
    await user.selectOptions(screen.getByLabelText('Rank order'), 'existing_easy_match')
    const rowsByEasyMatch = screen.getAllByRole('button').filter((button) => button.textContent?.includes('query_'))
    expect(rowsByEasyMatch[0]).toHaveTextContent('query_a')
    expect(rowsByEasyMatch[1]).toHaveTextContent('query_b')

    await user.selectOptions(screen.getByLabelText('Rank order'), 'date_time')
    const rowsByDate = screen.getAllByRole('button').filter((button) => button.textContent?.includes('query_'))
    expect(rowsByDate[0]).toHaveTextContent('query_b')
    expect(rowsByDate[1]).toHaveTextContent('query_a')
  })

  it('offers MegaStar as a preset and shows its numeric field in the Top-K breakdown', async () => {
    const user = userEvent.setup()
    mockedRunFirstOrderSearch.mockResolvedValueOnce({
      query_id: 'query_a',
      preset: 'megastar',
      candidates: [
        { entity_id: 'gallery_from_megastar', score: 0.91, k_contrib: 1, field_breakdown: { megastar: 0.91 } },
      ],
    })
    render(<FirstOrderPage />)

    await screen.findByDisplayValue('query_a')
    expect(screen.getByRole('option', { name: 'MegaStar' })).toBeInTheDocument()
    expect(screen.queryByLabelText('MegaStar ranking')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('MegaStar query image')).not.toBeInTheDocument()
    expect(screen.queryByText(/MegaStar image lookup/i)).not.toBeInTheDocument()

    await user.selectOptions(screen.getByLabelText('Preset'), 'megastar')
    await user.click(screen.getByRole('button', { name: 'Search' }))

    await waitFor(() => {
      expect(mockedRunFirstOrderSearch).toHaveBeenCalledWith({ query_id: 'query_a', top_k: 10, preset: 'megastar' })
    })
    expect(await screen.findByText('gallery_from_megastar')).toBeInTheDocument()
    expect(screen.getByText(/megastar: 0\.910/i)).toBeInTheDocument()
  })

  it('shows query imagery beside one active proposal with quick proposal navigation', async () => {
    const user = userEvent.setup()
    mockedRunFirstOrderSearch.mockResolvedValueOnce({
      query_id: 'query_a',
      preset: 'all',
      candidates: [
        { entity_id: 'gallery_visual_1', score: 0.93, k_contrib: 2, field_breakdown: { location: 1, megastar: 0.86 } },
        { entity_id: 'gallery_visual_2', score: 0.81, k_contrib: 1, field_breakdown: { color: 0.72 } },
      ],
    })

    render(<FirstOrderPage />)

    const selectorQueryImage = await screen.findByAltText('Selected query query_a image query_a.jpg')
    expect(selectorQueryImage).toHaveAttribute('loading', 'lazy')
    expect(mockedGetFirstOrderMedia).toHaveBeenCalledWith('query', 'query_a')

    await user.click(screen.getByRole('button', { name: 'Search' }))

    expect(await screen.findByRole('region', { name: 'Query Matcher side-by-side comparison' })).toBeInTheDocument()
    expect(screen.getByRole('region', { name: 'Query image comparison panel' })).toBeInTheDocument()
    expect(screen.getByRole('region', { name: 'Active proposal comparison panel' })).toBeInTheDocument()
    expect(screen.getAllByText('Wheel to zoom. Drag to pan. Hold R and drag to rotate.').length).toBeGreaterThanOrEqual(2)
    const queryViewer = screen.getByLabelText('Query matcher query image viewer')
    const proposalViewer = screen.getByLabelText('Query matcher proposal image viewer')
    const comparisonQueryImages = screen.getAllByAltText('Selected query query_a image query_a.jpg')
    const activeQueryImage = comparisonQueryImages.find((image) => image !== selectorQueryImage)
    expect(activeQueryImage).toBeTruthy()
    expect(activeQueryImage).not.toHaveAttribute('loading')
    expect(activeQueryImage?.closest('a')).toBeNull()
    const firstProposalImage = await screen.findByAltText('Rank 1 gallery_visual_1 image gallery_visual_1.jpg')
    expect(firstProposalImage).not.toHaveAttribute('loading')
    expect(firstProposalImage.closest('a')).toBeNull()
    expect(queryViewer).toHaveStyle({ height: '640px' })
    expect(proposalViewer).toHaveStyle({ height: '640px' })

    vi.spyOn(queryViewer, 'getBoundingClientRect').mockReturnValue({
      x: 10,
      y: 10,
      left: 10,
      top: 10,
      right: 510,
      bottom: 650,
      width: 500,
      height: 640,
      toJSON: () => ({}),
    })
    const wheelEvent = new WheelEvent('wheel', { clientX: 100, clientY: 100, deltaY: -300, bubbles: true, cancelable: true })
    window.dispatchEvent(wheelEvent)
    expect(wheelEvent.defaultPrevented).toBe(true)
    await waitFor(() => {
      expect(activeQueryImage).toHaveStyle({ transform: 'translate(0px, 0px) rotate(0deg) scale(1.3)' })
    })
    fireEvent.mouseDown(proposalViewer, { clientX: 100, clientY: 100 })
    fireEvent.mouseMove(window, { clientX: 130, clientY: 120 })
    fireEvent.mouseUp(window)
    expect(firstProposalImage).toHaveStyle({ transform: 'translate(30px, 20px) rotate(0deg) scale(1)' })
    expect(screen.getByText('Query image 1 of 2')).toBeInTheDocument()
    expect(screen.getByText('Proposal image 1 of 2')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Previous query image' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'Next query image' }))
    expect(await screen.findByAltText('Selected query query_a image query_a_detail.jpg')).toBeInTheDocument()
    expect(screen.getByText('Query image 2 of 2')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Next query image' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'Next proposal image' }))
    expect(await screen.findByAltText('Rank 1 gallery_visual_1 image gallery_visual_1_detail.jpg')).toBeInTheDocument()
    expect(screen.getByText('Proposal image 2 of 2')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Next proposal image' })).toBeDisabled()
    expect(screen.getByText('Proposal 1 of 2')).toBeInTheDocument()
    expect(screen.getByText('gallery_visual_1')).toBeInTheDocument()
    expect(screen.getByText(/location: 1\.000/i)).toBeInTheDocument()
    expect(screen.getByText(/megastar: 0\.860/i)).toBeInTheDocument()
    expect(screen.queryByRole('table')).not.toBeInTheDocument()
    expect(screen.queryByRole('region', { name: 'First-order visual lineup' })).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Next proposal' }))

    const secondProposalImage = await screen.findByAltText('Rank 2 gallery_visual_2 image gallery_visual_2.jpg')
    expect(secondProposalImage).not.toHaveAttribute('loading')
    expect(secondProposalImage.closest('a')).toBeNull()
    expect(screen.getByText('Proposal 2 of 2')).toBeInTheDocument()
    expect(screen.getByText('gallery_visual_2')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Next proposal' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Previous proposal' })).toBeEnabled()
    expect(mockedGetFirstOrderMedia).toHaveBeenCalledWith('gallery', 'gallery_visual_1')
    expect(mockedGetFirstOrderMedia).toHaveBeenCalledWith('gallery', 'gallery_visual_2')
  })

  it('refreshes query options while preserving the selected query when it still exists', async () => {
    const user = userEvent.setup()
    mockedGetFirstOrderQueries
      .mockResolvedValueOnce({ queries: queryOptions })
      .mockResolvedValueOnce({ queries: queryOptions.slice(1) })
    render(<FirstOrderPage />)

    await screen.findByDisplayValue('query_a')
    await user.click(screen.getByTitle('Previous query in list'))
    expect(screen.getByLabelText('Query')).toHaveValue('query_b')
    await user.click(screen.getByRole('button', { name: 'Refresh' }))

    await waitFor(() => {
      expect(mockedGetFirstOrderQueries).toHaveBeenCalledTimes(2)
    })
    expect(screen.getByLabelText('Query')).toHaveValue('query_b')
  })
})
