import { useCallback, useState } from 'react'
import { Database, Search } from 'lucide-react'
import { IndexingPanel } from '@/components/IndexingPanel'

export default function IndexPage() {
  const [isIndexing, setIsIndexing] = useState(false)

  const handleDone = useCallback(() => setIsIndexing(false), [])

  return (
    <div className="max-w-5xl mx-auto p-5 lg:p-8 space-y-5">
      <div>
        <h1 className="font-display text-2xl font-semibold text-bright flex items-center gap-3">
          <Database className="w-6 h-6 text-accent" />
          Photo Indexing
        </h1>
        <p className="text-sm text-ghost mt-1 flex items-center gap-2">
          Build and inspect embeddings here. Search remains available on the Search page while indexing runs.
          <Search className="w-3.5 h-3.5 text-dim" />
        </p>
      </div>

      <IndexingPanel
        onIndexingStart={() => setIsIndexing(true)}
        onIndexingComplete={handleDone}
      />

      {isIndexing && (
        <div className="glass border border-accent/20 rounded-2xl p-4">
          <p className="text-xs text-ghost leading-relaxed">
            Indexing runs in the backend. You can switch to Search and continue querying the already indexed images.
            New images become searchable as batches are committed.
          </p>
        </div>
      )}
    </div>
  )
}
