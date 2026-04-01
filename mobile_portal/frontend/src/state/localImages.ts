import { useMemo, useState } from 'react'

export function useLocalImages() {
  const [files, setFiles] = useState<File[]>([])

  const previews = useMemo(() => files.map((file) => ({ file, url: URL.createObjectURL(file) })), [files])

  function addFiles(next: FileList | null) {
    if (!next) return
    setFiles((prev) => [...prev, ...Array.from(next)])
  }

  function removeAt(index: number) {
    setFiles((prev) => prev.filter((_, i) => i !== index))
  }

  function clear() {
    setFiles([])
  }

  return { files, previews, addFiles, removeAt, clear }
}
