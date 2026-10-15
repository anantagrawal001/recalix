import { useState, useCallback, useEffect, useRef } from 'react'
import { FolderOpen, Play, RotateCcw, Check, AlertCircle, Loader2, Database, Server, Zap, ChevronUp, ChevronDown, Info } from 'lucide-react'
import { cn } from '@/lib/utils'
import { startIndexing, getIndexStatus, clearIndex, clearIndexFolder, checkHealth, selectDirectory, buildFaissIndex, type IndexStatus } from '@/lib/api'

// Fix #6: tooltip component for destructive actions
function Tooltip({ text, children }: { text: string; children?: React.ReactNode }) {
  const [show, setShow] = useState(false)
  return (
    <div className="relative inline-flex items-center">
      {children}
      <button
        type="button"
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        onFocus={() => setShow(true)}
        onBlur={() => setShow(false)}
        className="ml-1 text-ghost hover:text-dim transition-colors">
        <Info className="w-3 h-3" />
      </button>
      {show && (
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 z-50
                        w-56 p-2 rounded-lg bg-card border border-border/60 shadow-lift
                        text-xs text-soft leading-relaxed pointer-events-none">
          {text}
          <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-card" />
        </div>
      )}
    </div>
  )
}

export function IndexingPanel({ onIndexingComplete, onIndexingStart }:
  { onIndexingComplete: () => void; onIndexingStart: () => void }) {
  const [dir,       setDir]       = useState('')
  const [status,    setStatus]    = useState<IndexStatus | null>(null)
  const [online,    setOnline]    = useState<boolean | null>(null)
  const [error,     setError]     = useState('')
  const [reindex,   setReindex]   = useState(false)
  const [collapsed, setCollapsed] = useState(false)
  const [faissMsg,  setFaissMsg]  = useState('')
  const [clearFolder, setClearFolder] = useState('')
  const [clearMsg, setClearMsg] = useState('')
  const [isClearingFolder, setIsClearingFolder] = useState(false)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => { checkHealth().then(setOnline) }, [])
  useEffect(() => { getIndexStatus().then(setStatus).catch(() => {}) }, [])

  const startPoll = useCallback(() => {
    if (pollRef.current) return
    pollRef.current = setInterval(async () => {
      try {
        const s = await getIndexStatus(); setStatus(s)
        if (s.status === 'done' || s.status === 'error') {
          clearInterval(pollRef.current!); pollRef.current = null
          if (s.status === 'done') onIndexingComplete()
        }
      } catch { clearInterval(pollRef.current!); pollRef.current = null }
    }, 800)
  }, [onIndexingComplete])

  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current) }, [])

  const handleStart = useCallback(async () => {
    if (!dir.trim()) { setError('Enter a directory path'); return }
    setError('')
    try { onIndexingStart(); await startIndexing(dir.trim(), reindex); startPoll() }
    catch (e) { setError(e instanceof Error ? e.message : 'Failed') }
  }, [dir, reindex, onIndexingStart, startPoll])

  const handleClear = useCallback(async () => {
    if (!confirm(
      'Clear all indexed data?\n\n' +
      'This will delete:\n' +
      '• All embeddings (embeddings.npy)\n' +
      '• All metadata (metadata.db)\n' +
      '• All thumbnails\n\n' +
      'Your actual photos are NOT deleted.\n' +
      'You will need to re-index to search again.'
    )) return
    await clearIndex()
    const s = await getIndexStatus(); setStatus(s); onIndexingComplete()
  }, [onIndexingComplete])

  const handleBrowse = useCallback(async () => {
    const d = await selectDirectory(); if (d) setDir(d)
  }, [])

  const handleBuildFaiss = useCallback(async () => {
    try { await buildFaissIndex(); setFaissMsg('FAISS build started…') }
    catch (e) { setFaissMsg(`Failed: ${e instanceof Error ? e.message : e}`) }
    setTimeout(() => setFaissMsg(''), 5000)
  }, [])

  const handleClearFolder = useCallback(async () => {
    if (!clearFolder.trim()) { setClearMsg('Enter a folder path to clear'); return }
    if (!confirm(
      'Clear indexed data for this folder?\n\n' +
      'This will delete embeddings, thumbnails, and metadata for this folder (and subfolders).\n' +
      'Your actual photo files are NOT deleted.'
    )) return
    setClearMsg('')
    setIsClearingFolder(true)
    try {
      const res = await clearIndexFolder(clearFolder.trim(), true)
      setClearMsg(`Cleared ${res.deleted ?? 0} items.`)
      const s = await getIndexStatus(); setStatus(s); onIndexingComplete()
    } catch (e) {
      setClearMsg(e instanceof Error ? e.message : 'Failed')
    } finally {
      setIsClearingFolder(false)
    }
  }, [clearFolder, onIndexingComplete])

  const isRunning = status?.status === 'scanning' || status?.status === 'indexing'
  const pct       = status && status.total > 0 ? Math.round((status.processed / status.total) * 100) : 0
  const indexed   = status?.db_stats?.images ?? 0

  return (
    <div className="glass border border-border/50 rounded-2xl p-5 space-y-4 shadow-card">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-accent to-purple-500 flex items-center justify-center shadow-glow-sm">
            <Database className="w-4 h-4 text-white" />
          </div>
          <div>
            <h3 className="font-display font-semibold text-bright text-sm">Photo Indexer</h3>
            <p className="text-xs text-ghost">{indexed > 0 ? `${indexed.toLocaleString()} images indexed` : 'No photos indexed yet'}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <div className={cn('flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border',
            online === null ? 'bg-surface border-border text-ghost' :
            online          ? 'bg-success/10 border-success/20 text-success' :
                              'bg-error/10 border-error/20 text-error')}>
            <Server className="w-3 h-3" />
            {online === null ? '…' : online ? 'Online' : 'Offline'}
          </div>
          <button onClick={() => setCollapsed(c => !c)} className="p-1 rounded-lg hover:bg-lift text-ghost">
            {collapsed ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {!collapsed && (
        <>
          {online === false && (
            <div className="flex items-start gap-2 p-3 rounded-xl bg-error/10 border border-error/20 text-error text-xs">
              <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
              <div>
                <p className="font-medium">Backend not reachable</p>
                <p className="opacity-70 font-mono mt-0.5">Run start.bat to start the backend</p>
              </div>
            </div>
          )}

          {/* Directory input */}
          <div className="space-y-2">
            <label className="text-xs font-medium text-soft">Directory Path</label>
            <div className="flex gap-2">
              <input type="text" value={dir} onChange={e => setDir(e.target.value)}
                placeholder="C:\Users\You\Pictures  or  /home/user/Photos"
                disabled={isRunning}
                onKeyDown={e => e.key === 'Enter' && handleStart()}
                className="flex-1 px-3 py-2 rounded-xl bg-surface border border-border/50 text-xs text-bright
                           font-mono placeholder:text-ghost focus:outline-none focus:border-accent/50 disabled:opacity-50" />
              <button onClick={handleBrowse} disabled={isRunning || !online}
                className="px-3 py-2 rounded-xl bg-lift border border-border/50 text-soft
                           hover:text-bright hover:border-accent/30 disabled:opacity-40 transition-all" title="Browse for folder">
                <FolderOpen className="w-4 h-4" />
              </button>
              <button onClick={handleStart} disabled={isRunning || !dir.trim() || !online}
                className="px-4 py-2 rounded-xl bg-gradient-to-br from-accent to-purple-500 text-white text-sm font-medium
                           flex items-center gap-2 shadow-glow-sm hover:opacity-90 active:scale-95
                           disabled:opacity-40 disabled:active:scale-100 transition-all">
                {isRunning ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                {isRunning ? 'Indexing…' : 'Index'}
              </button>
            </div>

            {/* Fix #6: info on force reindex */}
            <div className="flex items-center gap-2">
              <label className="flex items-center gap-2 text-xs text-ghost cursor-pointer select-none">
                <input type="checkbox" checked={reindex} onChange={e => setReindex(e.target.checked)} disabled={isRunning} />
                Force re-index
              </label>
              <Tooltip text="Deletes ALL existing index data (embeddings, thumbnails, database) and re-indexes everything from scratch. Use when images have changed or index is corrupted. Your actual photos are never deleted.">
                <span />
              </Tooltip>
            </div>
          </div>

          {/* Clear specific folder */}
          <div className="space-y-2">
            <label className="text-xs font-medium text-soft">Clear Folder Index</label>
            <div className="flex gap-2">
              <input
                type="text"
                value={clearFolder}
                onChange={e => setClearFolder(e.target.value)}
                placeholder="C:\\Users\\You\\Pictures\\Subfolder"
                disabled={isRunning || isClearingFolder}
                className="flex-1 px-3 py-2 rounded-xl bg-surface border border-border/50 text-xs text-bright
                           font-mono placeholder:text-ghost focus:outline-none focus:border-accent/50 disabled:opacity-50"
              />
              <button
                onClick={handleClearFolder}
                disabled={isRunning || isClearingFolder || !clearFolder.trim()}
                className="px-3 py-2 rounded-xl bg-lift border border-border/50 text-soft
                           hover:text-bright hover:border-accent/30 disabled:opacity-40 transition-all">
                {isClearingFolder ? <Loader2 className="w-4 h-4 animate-spin" /> : <RotateCcw className="w-4 h-4" />}
              </button>
            </div>
            {clearMsg && <div className="text-xs text-ghost">{clearMsg}</div>}
          </div>

          {error && <div className="flex items-center gap-2 text-xs text-error"><AlertCircle className="w-3.5 h-3.5" />{error}</div>}

          {/* Progress */}
          {isRunning && (
            <div className="space-y-2 animate-fade-in">
              <div className="flex justify-between text-xs">
                <span className="flex items-center gap-2 text-soft">
                  <Loader2 className="w-3.5 h-3.5 text-accent animate-spin" />
                  {status?.status === 'scanning' ? 'Scanning files…' : 'Embedding & indexing…'}
                </span>
                <span className="text-ghost font-mono">{status?.processed}/{status?.total} ({pct}%)</span>
              </div>
              <div className="h-1.5 rounded-full bg-surface overflow-hidden">
                <div className="h-full bg-gradient-to-r from-accent to-purple-500 rounded-full transition-all duration-300"
                  style={{ width: `${pct}%` }} />
              </div>
              {status?.current_file && <p className="text-xs text-ghost font-mono truncate">{status.current_file}</p>}
              {status?.detail && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2 rounded-xl bg-void/50 border border-border/30 p-3">
                  <Detail label="Stage" value={status.detail.stage || status.status} />
                  <Detail label="Batch" value={status.detail.batch_size != null ? String(status.detail.batch_size) : 'n/a'} />
                  <Detail label="Embedding" value={status.detail.embedding_index != null ? `#${status.detail.embedding_index}` : 'pending'} />
                  <Detail label="Size" value={status.detail.width && status.detail.height ? `${status.detail.width}x${status.detail.height}` : 'n/a'} />
                  <Detail label="OCR chars" value={status.detail.ocr_chars != null ? String(status.detail.ocr_chars) : '0'} />
                  <Detail label="Document" value={status.detail.is_document ? 'yes' : 'no'} />
                  <Detail label="Screenshot" value={status.detail.is_screenshot ? 'yes' : 'no'} />
                  <Detail label="Device" value={status.detail.device || 'auto'} />
                </div>
              )}
              <div className="flex gap-4 text-xs">
                <span className="text-success">+{status?.new_indexed} new</span>
                <span className="text-ghost">⏭ {status?.skipped} skipped</span>
                {(status?.errors ?? 0) > 0 && <span className="text-error">✗ {status?.errors} errors</span>}
              </div>
            </div>
          )}

          {status?.status === 'done' && (
            <div className="flex items-start gap-2 p-3 rounded-xl bg-success/10 border border-success/20 text-success text-xs animate-fade-in">
              <Check className="w-4 h-4 mt-0.5 shrink-0" />
              <div><p className="font-medium">Indexing complete</p><p className="opacity-70 mt-0.5">{status.message}</p></div>
            </div>
          )}
          {status?.status === 'error' && (
            <div className="flex items-start gap-2 p-3 rounded-xl bg-error/10 border border-error/20 text-error text-xs animate-fade-in">
              <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
              <div><p className="font-medium">Indexing failed</p><p className="opacity-70 mt-0.5">{status.message}</p></div>
            </div>
          )}

          {/* Footer actions with tooltips */}
          <div className="flex items-center justify-between pt-1">
            <div className="flex items-center">
              <button onClick={handleBuildFaiss} disabled={!indexed || !online}
                className="flex items-center gap-1.5 text-xs text-dim hover:text-accent disabled:opacity-30 transition-colors">
                <Zap className="w-3.5 h-3.5" />Build FAISS Index
              </button>
              <Tooltip text="Builds a FAISS Approximate Nearest Neighbor index for faster search. Recommended when you have 1000+ photos. Auto-enables for large collections. Safe to run anytime — does not delete photos or embeddings." />
            </div>
            {faissMsg && <span className="text-xs text-ghost">{faissMsg}</span>}
            {indexed > 0 && (
              <div className="flex items-center">
                <button onClick={handleClear} disabled={isRunning}
                  className="flex items-center gap-1.5 text-xs text-dim hover:text-error disabled:opacity-30 transition-colors">
                  <RotateCcw className="w-3.5 h-3.5" />Clear Index
                </button>
                <Tooltip text="⚠ DESTRUCTIVE: Deletes all index data — embeddings, thumbnails, and database. Your actual photo files are safe. You will need to re-index to use search again." />
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}

function Detail({ label, value }: { label:string; value:string }) {
  return (
    <div className="min-w-0">
      <p className="text-[11px] text-ghost">{label}</p>
      <p className="text-xs text-soft font-mono truncate">{value}</p>
    </div>
  )
}
