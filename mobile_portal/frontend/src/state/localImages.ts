import { useEffect, useMemo, useState } from 'react'

export type LocalPreview = { file: File; url: string }

export function useLocalImages() {
  const [files, setFiles] = useState<File[]>([])
  const [selectedIndex, setSelectedIndex] = useState(0)

  const previews = useMemo<LocalPreview[]>(() => files.map((file) => ({ file, url: URL.createObjectURL(file) })), [files])

  useEffect(() => {
    return () => {
      previews.forEach((preview) => URL.revokeObjectURL(preview.url))
    }
  }, [previews])

  function addFiles(next: FileList | null) {
    if (!next) return
    setFiles((prev) => [...prev, ...Array.from(next)])
  }

  function removeAt(index: number) {
    setFiles((prev) => prev.filter((_, i) => i !== index))
    setSelectedIndex((prev) => Math.max(0, prev > index ? prev - 1 : prev))
  }

  function clear() {
    setFiles([])
    setSelectedIndex(0)
  }

  function select(index: number) {
    setSelectedIndex(index)
  }

  return {
    files,
    previews,
    selectedIndex,
    selectedPreview: previews[selectedIndex],
    addFiles,
    removeAt,
    clear,
    select,
  }
}
