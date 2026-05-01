import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'

import App from './App'

vi.mock('./pages/SingleEntryPage', () => ({ SingleEntryPage: () => <div>Single Entry Page</div> }))
vi.mock('./pages/BatchUploadPage', () => ({ BatchUploadPage: () => <div>Batch Upload Page</div> }))
vi.mock('./pages/GalleryPage', () => ({ GalleryPage: () => <div>ID Review Page</div> }))
vi.mock('./pages/FirstOrderPage', () => ({ FirstOrderPage: () => <div>Query Matcher Page</div> }))

describe('App navigation', () => {
  afterEach(() => {
    cleanup()
    window.history.pushState({}, '', '/')
  })

  it('opens Batch Upload directly for the /batch-upload URL', () => {
    window.history.pushState({}, '', '/batch-upload')

    render(<App />)

    expect(screen.getByText('Batch Upload Page')).toBeInTheDocument()
  })

  it('labels the gallery/query review tab as ID Review', () => {
    render(<App />)

    expect(screen.getByRole('button', { name: 'ID Review' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Gallery Review' })).not.toBeInTheDocument()
  })

  it('labels the first-order matching tab as Query Matcher', () => {
    render(<App />)

    expect(screen.getByRole('button', { name: 'Query Matcher' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'First-order Search' })).not.toBeInTheDocument()
  })

  it('does not expose MegaStar as a separate top-level tab', () => {
    render(<App />)

    expect(screen.queryByRole('button', { name: 'MegaStar Search' })).not.toBeInTheDocument()
    expect(screen.queryByText('MegaStar Standalone Page')).not.toBeInTheDocument()
  })
})
