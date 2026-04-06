import { useState, useEffect, useCallback, useRef } from 'react'
import { SlidersHorizontal, Zap, Cpu } from 'lucide-react'
import { SearchBar }    from '@/components/SearchBar'
import { FilterSidebar, type FilterState } from '@/components/FilterSidebar'
import { PhotoGrid }    from '@/components/PhotoGrid'
import { IndexingPanel }from '@/components/IndexingPanel'
import { search, getFolders, getStats, type MediaResult, type FolderItem } from '@/lib/api'
import { cn } from '@/lib/utils'

export default function SearchPage() {
  const [sidebarOpen,    setSidebarOpen]    = useState(false)
  const [results,        setResults]        = useState<MediaResult[] | null>(null)
  const [isSearching,    setIsSearching]    = useState(false)
  const [isIndexing,     setIsIndexing]     = useState(false)
  const [folders,        setFolders]        = useState<FolderItem[]>([])
  const [totalIndexed,   setTotalIndexed]   = useState(0)
  const [currentPage,    setCurrentPage]    = useState(1)
  const [totalPages,     setTotalPages]     = useState(0)
  const [lastQuery,      setLastQuery]      = useState('')
  const [searchBackend,  setSearchBackend]  = useState<string>('')
  const [filters, setFilters] = useState<FilterState>({
    fileTypes: ['image', 'video'], folders: [], dateRange: null,
  })
  const abortRef = useRef<AbortController | null>(null)

  const loadMeta = useCallback(async () => {
    try {
      const [f, s] = await Promise.all([getFolders(), getStats()])
      setFolders(f)
      setTotalIndexed(s.images ?? 0)
    } catch {}
  }, [])

  useEffect(() => { loadMeta() }, [loadMeta])

  const doSearch = useCallback(async (query: string, page = 1) => {
    if (!query.trim()) return
    if (abortRef.current) abortRef.current.abort()
    abortRef.current = new AbortController()
    setIsSearching(true); setLastQuery(query)

    try {
      const activeType   = filters.fileTypes.length === 1 ? filters.fileTypes[0] : null
      const activeFolder = filters.folders.length === 1   ? filters.folders[0]   : null
      const res = await search({
        q:         query,
        type:      activeType,
        folder:    activeFolder,
        date_from: filters.dateRange?.min ? new Date(filters.dateRange.min).toISOString().split('T')[0] : null,
        date_to:   filters.dateRange?.max ? new Date(filters.dateRange.max).toISOString().split('T')[0] : null,
        top_k: 100, page,
        signal: abortRef.current.signal,
      })
      setResults(res.results)
      setCurrentPage(res.page)
      setTotalPages(res.total_pages)
      setSearchBackend(res.backend ?? '')
    } catch (e: any) {
      if (e.name !== 'AbortError') { console.error(e); setResults([]) }
    } finally {
      setIsSearching(false)
    }
  }, [filters])

  useEffect(() => { if (lastQuery) doSearch(lastQuery, 1) }, [filters])

  return (
    <div className="flex min-h-[calc(100vh-3.5rem)]">
      <FilterSidebar
        folders={folders.map(f => f.folder)}
        filters={filters}
        onFiltersChange={setFilters}
        resultCount={results?.length ?? 0}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <div className="flex-1 min-w-0">
        <div className="p-6 lg:p-8 space-y-8 max-w-screen-2xl mx-auto">

          {/* Search bar row */}
          <div className="flex items-start gap-3">
            <button
              onClick={() => setSidebarOpen(true)}
              className="lg:hidden mt-3 p-2.5 rounded-xl glass border border-border/50 text-soft hover:text-bright hover:border-accent/30 transition-all flex-shrink-0">
              <SlidersHorizontal className="w-4 h-4" />
            </button>
            <div className="flex-1">
              <SearchBar
                onSearch={q => { setCurrentPage(1); doSearch(q, 1) }}
                onClear={() => { setResults(null); setLastQuery(''); setCurrentPage(1) }}
                isSearching={isSearching}
                disabled={isIndexing}
              />
            </div>
          </div>

          {/* Indexing panel */}
          <div className="max-w-2xl mx-auto">
            <IndexingPanel
              onIndexingStart={() => setIsIndexing(true)}
              onIndexingComplete={() => { setIsIndexing(false); loadMeta() }}
            />
          </div>

          {/* Results area */}
          <div>
            {results !== null && results.length > 0 && (
              <div className="mb-4 flex items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                  <p className="text-sm text-ghost">
                    <span className="text-soft font-medium">{results.length}</span> results for{' '}
                    <span className="text-bright font-medium">"{lastQuery}"</span>
                  </p>
                  {searchBackend && (
                    <span className={cn(
                      'flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border',
                      searchBackend === 'faiss'
                        ? 'bg-accent/10 border-accent/20 text-accent'
                        : 'bg-surface border-border/30 text-ghost',
                    )}>
                      {searchBackend === 'faiss' ? <Zap className="w-3 h-3" /> : <Cpu className="w-3 h-3" />}
                      {searchBackend}
                    </span>
                  )}
                </div>

                {totalPages > 1 && (
                  <div className="flex items-center gap-2 text-sm">
                    <button
                      disabled={currentPage <= 1 || isSearching}
                      onClick={() => doSearch(lastQuery, currentPage - 1)}
                      className="px-3 py-1 rounded-lg glass border border-border/40 text-soft
                                 hover:border-accent/30 hover:text-bright disabled:opacity-30 transition-all">
                      ← Prev
                    </button>
                    <span className="text-ghost text-xs font-mono">{currentPage} / {totalPages}</span>
                    <button
                      disabled={currentPage >= totalPages || isSearching}
                      onClick={() => doSearch(lastQuery, currentPage + 1)}
                      className="px-3 py-1 rounded-lg glass border border-border/40 text-soft
                                 hover:border-accent/30 hover:text-bright disabled:opacity-30 transition-all">
                      Next →
                    </button>
                  </div>
                )}
              </div>
            )}

            {totalIndexed === 0 && !isIndexing ? (
              <EmptyState type="no-index" />
            ) : results === null ? (
              <EmptyState type="no-search" />
            ) : (
              <PhotoGrid photos={results} isLoading={isSearching} />
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function EmptyState({ type }: { type: 'no-index' | 'no-search' }) {
  return (
    <div className="flex flex-col items-center justify-center py-24 text-center animate-fade-in">
      <div className="w-24 h-24 rounded-3xl glass border border-border/40 flex items-center justify-center mb-6">
        <span className="text-4xl">{type === 'no-index' ? '📂' : '🔍'}</span>
      </div>
      <h3 className="font-display text-xl font-semibold text-soft mb-2">
        {type === 'no-index' ? 'No photos indexed yet' : 'Ready to search'}
      </h3>
      <p className="text-sm text-ghost max-w-xs leading-relaxed">
        {type === 'no-index'
          ? 'Use the Photo Indexer above to index a directory of photos.'
          : 'Type a query above to search your indexed photos using AI.'}
      </p>
    </div>
  )
}
