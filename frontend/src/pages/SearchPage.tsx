import { useCallback, useEffect, useRef, useState } from 'react'
import { Calendar, ChevronDown, Cpu, Filter, Folder, SlidersHorizontal, X, Zap } from 'lucide-react'
import { SearchBar } from '@/components/SearchBar'
import { PhotoGrid } from '@/components/PhotoGrid'
import {
  search,
  getGallery,
  getFolders,
  getStats,
  getSearchHistory,
  clearSearchHistory,
  deleteSearchHistoryItem,
  type MediaResult,
  type SearchHistoryItem,
} from '@/lib/api'
import { cn } from '@/lib/utils'

interface Filters {
  folder: string
  dateRange: 'all' | 'today' | 'week' | 'month' | 'year' | 'custom'
  customFrom: string
  customTo: string
  exts: string[]
}

const DATE_OPTIONS: {id: Filters['dateRange']; label: string}[] = [
  { id: 'all', label: 'All dates' },
  { id: 'today', label: 'Today' },
  { id: 'week', label: 'Last 7 days' },
  { id: 'month', label: 'Last 30 days' },
  { id: 'year', label: 'This year' },
  { id: 'custom', label: 'Custom range' },
]

const IMAGE_EXTS = [
  { id: 'jpg', label: 'JPG' },
  { id: 'jpeg', label: 'JPEG' },
  { id: 'png', label: 'PNG' },
  { id: 'webp', label: 'WEBP' },
  { id: 'heic', label: 'HEIC' },
  { id: 'heif', label: 'HEIF' },
  { id: 'gif', label: 'GIF' },
  { id: 'bmp', label: 'BMP' },
  { id: 'tiff', label: 'TIFF' },
]

function isoStart(dateStr: string) {
  if (!dateStr) return null
  const d = new Date(dateStr)
  d.setHours(0, 0, 0, 0)
  return d.toISOString()
}

function isoEnd(dateStr: string) {
  if (!dateStr) return null
  const d = new Date(dateStr)
  d.setHours(23, 59, 59, 999)
  return d.toISOString()
}

function dateBounds(filters: Filters) {
  const range = filters.dateRange
  if (range === 'all') return {}
  const end = new Date()
  const start = new Date()
  if (range === 'today') start.setHours(0, 0, 0, 0)
  if (range === 'week') start.setDate(start.getDate() - 7)
  if (range === 'month') start.setDate(start.getDate() - 30)
  if (range === 'year') start.setMonth(0, 1), start.setHours(0, 0, 0, 0)
  if (range === 'custom') {
    const from = isoStart(filters.customFrom)
    const to = isoEnd(filters.customTo)
    return { date_from: from ?? undefined, date_to: to ?? undefined }
  }
  return { date_from: start.toISOString(), date_to: end.toISOString() }
}

export default function SearchPage() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<MediaResult[] | null>(null)
  const [isSearching, setIsSearching] = useState(false)
  const [folders, setFolders] = useState<{folder:string;count:number}[]>([])
  const [totalIndexed, setTotalIndexed] = useState(0)
  const [currentPage, setCurrentPage] = useState(1)
  const [totalPages, setTotalPages] = useState(0)
  const [lastQuery, setLastQuery] = useState('')
  const [backend, setBackend] = useState('')
  const [searchInfo, setSearchInfo] = useState('')
  const [filters, setFilters] = useState<Filters>({
    folder: '',
    dateRange: 'all',
    customFrom: '',
    customTo: '',
    exts: [],
  })
  const [history, setHistory] = useState<SearchHistoryItem[]>([])
  const [viewMode, setViewMode] = useState<'search' | 'gallery'>('search')
  const [galleryFolder, setGalleryFolder] = useState('')
  const [galleryTotal, setGalleryTotal] = useState(0)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [folderOpen, setFolderOpen] = useState(true)
  const [dateOpen, setDateOpen] = useState(true)
  const [typeOpen, setTypeOpen] = useState(true)
  const abortRef = useRef<AbortController | null>(null)

  const loadMeta = useCallback(async () => {
    try {
      const [f, s] = await Promise.all([getFolders(), getStats()])
      setFolders(f)
      setTotalIndexed(s.images ?? 0)
    } catch {}
  }, [])

  useEffect(() => { loadMeta() }, [loadMeta])

  const refreshHistory = useCallback(async () => {
    try {
      const res = await getSearchHistory(3)
      setHistory(res.items)
    } catch {}
  }, [])

  useEffect(() => { refreshHistory() }, [refreshHistory])

  const doSearch = useCallback(async (q: string, page = 1, f = filters) => {
    if (!q.trim()) return
    if (abortRef.current) abortRef.current.abort()
    abortRef.current = new AbortController()
    setIsSearching(true)
    setLastQuery(q)
    try {
      const dates = dateBounds(f)
      const ext = f.exts.length > 0 ? f.exts.join(',') : null
      const res = await search({
        q,
        page,
        type: 'image',
        folder: f.folder || null,
        top_k: 100,
        signal: abortRef.current.signal,
        ext,
        ...dates,
      })
      setResults(res.results)
      setViewMode('search')
      setGalleryFolder('')
      setGalleryTotal(0)
      setCurrentPage(res.page)
      setTotalPages(res.total_pages)
      setBackend(res.backend ?? '')
      const parts = [`${res.total} results`, res.backend ?? 'cosine', 'Adaptive Fusion']
      if (f.dateRange !== 'all') parts.push(DATE_OPTIONS.find(d => d.id === f.dateRange)?.label || '')
      if (f.folder) parts.push(f.folder.split(/[/\\]/).pop() || f.folder)
      if (f.exts.length > 0) parts.push(`types: ${f.exts.join(', ')}`)
      setSearchInfo(parts.filter(Boolean).join(' + '))
    } catch (e: any) {
      if (e.name !== 'AbortError') { console.error(e); setResults([]) }
    } finally {
      setIsSearching(false)
      refreshHistory()
    }
  }, [filters, refreshHistory])

  const doGallery = useCallback(async (folder: string, page = 1, f = filters) => {
    if (abortRef.current) abortRef.current.abort()
    abortRef.current = new AbortController()
    setIsSearching(true)
    setViewMode('gallery')
    setGalleryFolder(folder)
    setLastQuery('')
    setQuery('')
    try {
      const ext = f.exts.length > 0 ? f.exts.join(',') : null
      const res = await getGallery({
        folder: folder || null,
        type: 'image',
        page,
        page_size: 48,
        signal: abortRef.current.signal,
        ext,
      })
      setResults(res.results)
      setCurrentPage(res.page)
      setTotalPages(res.total_pages)
      setGalleryTotal(res.total)
      setBackend('gallery')
      const folderName = folder ? (folder.split(/[/\\]/).pop() || folder) : 'All folders'
      setSearchInfo(`Gallery: ${folderName} + photos`)
    } catch (e: any) {
      if (e.name !== 'AbortError') { console.error(e); setResults([]); setGalleryTotal(0) }
    } finally {
      setIsSearching(false)
    }
  }, [filters])

  const applyFilters = useCallback((next: Filters) => {
    setFilters(next)
    if (viewMode === 'gallery') doGallery(next.folder, 1, next)
    else if (lastQuery) doSearch(lastQuery, 1, next)
  }, [viewMode, lastQuery, doGallery, doSearch])

  const toggleFolder = useCallback((folder: string) => {
    const nextFolder = filters.folder === folder ? '' : folder
    const next = { ...filters, folder: nextFolder }
    applyFilters(next)
    doGallery(nextFolder, 1, next)
  }, [filters, applyFilters, doGallery])

  const toggleExt = useCallback((ext: string) => {
    const nextExts = filters.exts.includes(ext)
      ? filters.exts.filter(e => e !== ext)
      : [...filters.exts, ext]
    applyFilters({ ...filters, exts: nextExts })
  }, [filters, applyFilters])

  const handleHistoryPick = useCallback((q: string) => {
    setQuery(q)
    setCurrentPage(1)
    doSearch(q, 1)
  }, [doSearch])

  const handleClearHistory = useCallback(async () => {
    try {
      await clearSearchHistory()
      refreshHistory()
    } catch {}
  }, [refreshHistory])

  const handleDeleteHistoryItem = useCallback(async (id: number) => {
    try {
      await deleteSearchHistoryItem(id)
      refreshHistory()
    } catch {}
  }, [refreshHistory])

  const activeFilterCount = (filters.folder ? 1 : 0)
    + (filters.dateRange !== 'all' ? 1 : 0)
    + (filters.exts.length > 0 ? 1 : 0)

  return (
    <div className="flex min-h-[calc(100vh-3.5rem)]">
      {sidebarOpen && (
        <div className="fixed inset-0 bg-void/60 z-40 lg:hidden backdrop-blur-sm" onClick={() => setSidebarOpen(false)} />
      )}

      <aside className={cn(
        'fixed lg:sticky top-14 left-0 z-50 lg:z-auto w-72 h-[calc(100vh-3.5rem)]',
        'glass-heavy border-r border-border/40 flex flex-col transition-transform duration-300',
        sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0',
      )}>
        <div className="flex items-center justify-between px-5 py-4 border-b border-border/30">
          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-accent" />
            <span className="text-sm font-semibold text-bright uppercase tracking-wide">Filters</span>
          </div>
          <div className="flex items-center gap-2">
            {activeFilterCount > 0 && (
              <button onClick={() => applyFilters({ folder: '', dateRange: 'all', customFrom: '', customTo: '', exts: [] })}
                className="text-xs text-dim hover:text-soft transition-colors">Clear</button>
            )}
            <button onClick={() => setSidebarOpen(false)} className="lg:hidden p-1 rounded hover:bg-lift text-ghost">
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-5 space-y-5">
          <div className="space-y-2">
            <button onClick={() => setDateOpen(o => !o)}
              className="flex items-center justify-between w-full text-sm font-medium text-soft hover:text-bright transition-colors">
              <span className="flex items-center gap-2"><Calendar className="w-3.5 h-3.5 text-dim" />Date Range</span>
              <ChevronDown className={cn('w-3.5 h-3.5 transition-transform', dateOpen && 'rotate-180')} />
            </button>
            {dateOpen && (
              <div className="space-y-1">
                {DATE_OPTIONS.map(opt => (
                  <button key={opt.id} onClick={() => applyFilters({ ...filters, dateRange: opt.id })}
                    className={cn('w-full flex items-center gap-3 px-2 py-2 rounded-lg text-left hover:bg-lift transition-colors',
                      filters.dateRange === opt.id && 'bg-accent/10 text-accent')}>
                    <CheckBox checked={filters.dateRange === opt.id} />
                    <span className="text-sm text-soft">{opt.label}</span>
                  </button>
                ))}
                {filters.dateRange === 'custom' && (
                  <div className="mt-2 space-y-2 px-2">
                    <div className="space-y-1">
                      <label className="text-[11px] text-ghost">From</label>
                      <input
                        type="date"
                        value={filters.customFrom}
                        onChange={e => applyFilters({ ...filters, customFrom: e.target.value, dateRange: 'custom' })}
                        className="w-full px-2 py-1.5 rounded-lg bg-surface border border-border/50 text-xs text-bright"
                      />
                    </div>
                    <div className="space-y-1">
                      <label className="text-[11px] text-ghost">To</label>
                      <input
                        type="date"
                        value={filters.customTo}
                        onChange={e => applyFilters({ ...filters, customTo: e.target.value, dateRange: 'custom' })}
                        className="w-full px-2 py-1.5 rounded-lg bg-surface border border-border/50 text-xs text-bright"
                      />
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          <div className="space-y-2">
            <button onClick={() => setTypeOpen(o => !o)}
              className="flex items-center justify-between w-full text-sm font-medium text-soft hover:text-bright transition-colors">
              <span className="flex items-center gap-2"><Filter className="w-3.5 h-3.5 text-dim" />File Types</span>
              <ChevronDown className={cn('w-3.5 h-3.5 transition-transform', typeOpen && 'rotate-180')} />
            </button>
            {typeOpen && (
              <div className="space-y-1">
                <button
                  onClick={() => applyFilters({ ...filters, exts: [] })}
                  className={cn('w-full flex items-center gap-3 px-2 py-2 rounded-lg text-left hover:bg-lift transition-colors',
                    filters.exts.length === 0 && 'bg-accent/10 text-accent')}>
                  <CheckBox checked={filters.exts.length === 0} />
                  <span className="text-sm text-soft">All image types</span>
                </button>
                {IMAGE_EXTS.map(ext => (
                  <button key={ext.id} onClick={() => toggleExt(ext.id)}
                    className={cn('w-full flex items-center gap-3 px-2 py-2 rounded-lg text-left hover:bg-lift transition-colors',
                      filters.exts.includes(ext.id) && 'bg-accent/10 text-accent')}>
                    <CheckBox checked={filters.exts.includes(ext.id)} />
                    <span className="text-sm text-soft">{ext.label}</span>
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="space-y-2">
            <button onClick={() => setFolderOpen(o => !o)}
              className="flex items-center justify-between w-full text-sm font-medium text-soft hover:text-bright transition-colors">
              <span className="flex items-center gap-2"><Folder className="w-3.5 h-3.5 text-dim" />Folders</span>
              <ChevronDown className={cn('w-3.5 h-3.5 transition-transform', folderOpen && 'rotate-180')} />
            </button>
            {folderOpen && (
              <div className="max-h-[26rem] overflow-y-auto space-y-0.5 pr-1">
                <button onClick={() => toggleFolder('')}
                  className={cn('w-full flex items-center gap-3 px-2 py-2 rounded-lg text-left hover:bg-lift transition-colors',
                    !filters.folder && 'bg-accent/10')}>
                  <CheckBox checked={!filters.folder} />
                  <span className="text-sm text-soft flex-1">All folders</span>
                  <span className="text-xs text-ghost">{totalIndexed.toLocaleString()}</span>
                </button>
                {folders.map(({ folder, count }) => (
                  <button key={folder} onClick={() => toggleFolder(folder)}
                    title={filters.folder === folder ? 'Click to unselect this folder' : folder}
                    className={cn('w-full flex items-center gap-3 px-2 py-2 rounded-lg text-left hover:bg-lift transition-colors',
                      filters.folder === folder && 'bg-accent/10')}>
                    <CheckBox checked={filters.folder === folder} />
                    <Folder className="w-4 h-4 text-dim shrink-0" />
                    <span className="text-sm text-soft truncate flex-1">{folder.split(/[/\\]/).pop() || folder}</span>
                    <span className="text-xs text-ghost shrink-0">{count}</span>
                  </button>
                ))}
              </div>
            )}
          </div>

          {searchInfo && (
            <div className="pt-4 border-t border-border/20">
              <p className="text-xs text-ghost leading-relaxed">{searchInfo}</p>
            </div>
          )}
        </div>
      </aside>

      <div className="flex-1 min-w-0">
        <div className="p-5 lg:p-8 space-y-7 max-w-screen-2xl mx-auto">
          <div className="flex items-start gap-3">
            <button onClick={() => setSidebarOpen(true)}
              className={cn('lg:hidden mt-3 p-2.5 rounded-xl glass border text-soft hover:text-bright transition-all shrink-0',
                activeFilterCount > 0 ? 'border-accent/40 text-accent bg-accent/10' : 'border-border/50 hover:border-accent/30')}>
              <SlidersHorizontal className="w-4 h-4" />
            </button>
            <div className="flex-1">
              <SearchBar
                value={query}
                onChange={setQuery}
                onSearch={q => { setCurrentPage(1); doSearch(q, 1) }}
                onClear={() => {
                  setResults(null); setLastQuery(''); setCurrentPage(1); setSearchInfo('')
                  setViewMode('search'); setGalleryFolder(''); setGalleryTotal(0)
                }}
                isSearching={isSearching}
                showExamples={false}
                history={history}
                onHistorySelect={handleHistoryPick}
                onClearHistory={handleClearHistory}
                onDeleteHistoryItem={handleDeleteHistoryItem}
              />
            </div>
          </div>

          <div>
            {results !== null && results.length > 0 && (
              <div className="mb-4 flex items-center justify-between flex-wrap gap-3">
                <div className="flex items-center gap-3 flex-wrap">
                  <p className="text-sm text-ghost">
                    {viewMode === 'gallery' ? (
                      <>
                        Showing <span className="text-soft font-medium">{results.length}</span>
                        {' '}of <span className="text-soft font-medium">{galleryTotal}</span> photos in{' '}
                        <span className="text-bright font-medium">
                          {galleryFolder ? (galleryFolder.split(/[/\\]/).pop() || galleryFolder) : 'All folders'}
                        </span>
                      </>
                    ) : (
                      <>
                        Showing <span className="text-soft font-medium">{results.length}</span> results for{' '}
                        <span className="text-bright font-medium">"{lastQuery}"</span>
                      </>
                    )}
                  </p>
                  {backend && (
                    <span className={cn('flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border',
                      backend === 'faiss'
                        ? 'bg-accent/10 border-accent/20 text-accent'
                        : 'bg-surface border-border/30 text-ghost')}>
                      {backend === 'faiss' ? <Zap className="w-3 h-3" /> : <Cpu className="w-3 h-3" />}
                      {backend}
                    </span>
                  )}
                </div>
                {totalPages > 1 && (
                  <div className="flex items-center gap-2 text-sm">
                    <button disabled={currentPage <= 1 || isSearching}
                      onClick={() => viewMode === 'gallery' ? doGallery(galleryFolder, currentPage - 1) : doSearch(lastQuery, currentPage - 1)}
                      className="px-3 py-1 rounded-lg glass border border-border/40 text-soft hover:border-accent/30 hover:text-bright disabled:opacity-30 transition-all">
                      Prev
                    </button>
                    <span className="text-ghost text-xs font-mono">{currentPage}/{totalPages}</span>
                    <button disabled={currentPage >= totalPages || isSearching}
                      onClick={() => viewMode === 'gallery' ? doGallery(galleryFolder, currentPage + 1) : doSearch(lastQuery, currentPage + 1)}
                      className="px-3 py-1 rounded-lg glass border border-border/40 text-soft hover:border-accent/30 hover:text-bright disabled:opacity-30 transition-all">
                      Next
                    </button>
                  </div>
                )}
              </div>
            )}

            {totalIndexed === 0 ? (
              <EmptyState type="no-index" />
            ) : results === null ? (
              <EmptyState type="no-search" />
            ) : (
              <PhotoGrid photos={results} isLoading={isSearching} variant={viewMode === 'gallery' ? 'gallery' : 'search'} />
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function CheckBox({ checked }: { checked:boolean }) {
  return (
    <span className={cn('w-5 h-5 rounded-md border flex items-center justify-center shrink-0 transition-all',
      checked ? 'bg-accent border-accent' : 'border-border')}>
      {checked && (
        <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 12 12">
          <path d="M2 6l3 3 5-5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      )}
    </span>
  )
}

function EmptyState({ type }: { type: 'no-index' | 'no-search' }) {
  return (
    <div className="flex flex-col items-center justify-center py-28 text-center animate-fade-in">
      <div className="w-24 h-24 rounded-3xl glass border border-border/40 flex items-center justify-center mb-5">
        <span className="text-5xl">{type === 'no-index' ? 'IDX' : 'AI'}</span>
      </div>
      <h3 className="font-display text-xl font-semibold text-soft mb-2">
        {type === 'no-index' ? 'No photos indexed yet' : 'Ready to search'}
      </h3>
      <p className="text-sm text-ghost max-w-xs leading-relaxed">
        {type === 'no-index'
          ? 'Open the Index page to index a directory of photos.'
          : 'Type a query or choose a folder to browse your indexed photos.'}
      </p>
    </div>
  )
}
