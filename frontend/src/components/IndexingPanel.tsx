import { useState, useCallback, useEffect, useRef } from 'react'
import { FolderOpen, Play, RotateCcw, Check, AlertCircle, Loader2, Database, ChevronDown, ChevronUp, Server, Zap } from 'lucide-react'
import { cn } from '@/lib/utils'
import { startIndexing, getIndexStatus, clearIndex, checkHealth, selectDirectory, buildFaissIndex, type IndexStatus } from '@/lib/api'

interface Props { onIndexingComplete: () => void; onIndexingStart: () => void }

export function IndexingPanel({ onIndexingComplete, onIndexingStart }: Props) {
  const [dir,       setDir]       = useState('')
  const [status,    setStatus]    = useState<IndexStatus | null>(null)
  const [online,    setOnline]    = useState<boolean | null>(null)
  const [error,     setError]     = useState('')
  const [reindex,   setReindex]   = useState(false)
  const [collapsed, setCollapsed] = useState(false)
  const [faissMsg,  setFaissMsg]  = useState('')
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => { checkHealth().then(setOnline) }, [])
  useEffect(() => { getIndexStatus().then(setStatus).catch(() => {}) }, [])

  const startPoll = useCallback(() => {
    if (pollRef.current) return
    pollRef.current = setInterval(async () => {
      try {
        const s = await getIndexStatus()
        setStatus(s)
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
    if (!confirm('Clear all indexed data? This cannot be undone.')) return
    await clearIndex(); const s = await getIndexStatus(); setStatus(s); onIndexingComplete()
  }, [onIndexingComplete])

  const handleBrowse = useCallback(async () => {
    const d = await selectDirectory(); if (d) setDir(d)
  }, [])

  const handleBuildFaiss = useCallback(async () => {
    try { await buildFaissIndex(); setFaissMsg('FAISS build started…') }
    catch (e) { setFaissMsg(`Failed: ${e instanceof Error ? e.message : e}`) }
    setTimeout(() => setFaissMsg(''), 4000)
  }, [])

  const isRunning   = status?.status === 'scanning' || status?.status === 'indexing'
  const pct         = status && status.total > 0 ? Math.round((status.processed / status.total) * 100) : 0
  const indexed     = status?.db_stats?.images ?? 0

  return (
    <div className="glass border border-border/50 rounded-2xl p-5 space-y-4 shadow-card">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-accent-grad flex items-center justify-center shadow-glow-sm">
            <Database className="w-4 h-4 text-white" />
          </div>
          <div>
            <h3 className="font-display font-semibold text-bright text-sm">Photo Indexer</h3>
            <p className="text-xs text-ghost">
              {indexed > 0 ? `${indexed.toLocaleString()} images indexed` : 'No photos indexed yet'}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <div className={cn(
            'flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border',
            online === null  ? 'bg-surface border-border text-ghost' :
            online           ? 'bg-success/10 border-success/20 text-success' :
                               'bg-error/10 border-error/20 text-error',
          )}>
            <Server className="w-3 h-3" />
            {online === null ? 'Checking…' : online ? 'Online' : 'Offline'}
          </div>
          <button onClick={() => setCollapsed(c => !c)} className="p-1 rounded-lg hover:bg-lift text-ghost">
            {collapsed ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {!collapsed && (
        <>
          {/* Offline warning */}
          {online === false && (
            <div className="flex items-start gap-3 p-3 rounded-xl bg-error/10 border border-error/20 text-error text-xs">
              <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
              <div>
                <p className="font-medium">Backend not reachable</p>
                <p className="text-error/70 mt-0.5 font-mono">uvicorn app.main:app --port 8000</p>
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
                className="flex-1 px-3 py-2 rounded-xl bg-surface border border-border/50 text-sm text-bright
                           placeholder:text-ghost focus:outline-none focus:border-accent/50
                           disabled:opacity-50 font-mono text-xs" />
              <button onClick={handleBrowse} disabled={isRunning || !online}
                className="px-3 py-2 rounded-xl bg-lift border border-border/50 text-soft hover:text-bright
                           hover:border-accent/30 disabled:opacity-40 transition-all">
                <FolderOpen className="w-4 h-4" />
              </button>
              <button onClick={handleStart} disabled={isRunning || !dir.trim() || !online}
                className="px-4 py-2 rounded-xl bg-accent-grad text-white text-sm font-medium
                           flex items-center gap-2 shadow-glow-sm hover:opacity-90 active:scale-95
                           disabled:opacity-40 disabled:active:scale-100 transition-all">
                {isRunning ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                {isRunning ? 'Indexing…' : 'Index'}
              </button>
            </div>

            <label className="flex items-center gap-2 text-xs text-ghost cursor-pointer select-none">
              <input type="checkbox" checked={reindex} onChange={e => setReindex(e.target.checked)}
                disabled={isRunning} className="rounded border-border bg-surface" />
              Force re-index (clears all data)
            </label>
          </div>

          {error && (
            <div className="flex items-center gap-2 text-xs text-error">
              <AlertCircle className="w-3.5 h-3.5" /> {error}
            </div>
          )}

          {/* Progress */}
          {isRunning && (
            <div className="space-y-2 animate-fade-in">
              <div className="flex justify-between text-xs">
                <span className="flex items-center gap-2 text-soft">
                  <Loader2 className="w-3.5 h-3.5 text-accent animate-spin" />
                  {status?.status === 'scanning' ? 'Scanning…' : 'Indexing…'}
                </span>
                <span className="text-ghost font-mono">{status?.processed} / {status?.total} ({pct}%)</span>
              </div>
              {/* Progress bar */}
              <div className="h-1.5 rounded-full bg-surface overflow-hidden">
                <div className="h-full bg-accent-grad rounded-full transition-all duration-300"
                  style={{ width: `${pct}%` }} />
              </div>
              {status?.current_file && (
                <p className="text-xs text-ghost font-mono truncate">{status.current_file}</p>
              )}
              <div className="flex gap-4 text-xs">
                <span className="text-success">+{status?.new_indexed} new</span>
                <span className="text-ghost">⏭ {status?.skipped} skipped</span>
                {(status?.errors ?? 0) > 0 && <span className="text-error">✗ {status?.errors} errors</span>}
              </div>
            </div>
          )}

          {/* Done */}
          {status?.status === 'done' && (
            <div className="flex items-start gap-3 p-3 rounded-xl bg-success/10 border border-success/20 text-success text-xs animate-fade-in">
              <Check className="w-4 h-4 mt-0.5 flex-shrink-0" />
              <div>
                <p className="font-medium">Indexing complete</p>
                <p className="opacity-70 mt-0.5">{status.message}</p>
              </div>
            </div>
          )}

          {/* Error */}
          {status?.status === 'error' && (
            <div className="flex items-start gap-3 p-3 rounded-xl bg-error/10 border border-error/20 text-error text-xs animate-fade-in">
              <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
              <div>
                <p className="font-medium">Indexing failed</p>
                <p className="opacity-70 mt-0.5">{status.message}</p>
              </div>
            </div>
          )}

          {/* Footer actions */}
          <div className="flex items-center justify-between pt-1">
            <button onClick={handleBuildFaiss} disabled={!indexed || !online}
              className="flex items-center gap-1.5 text-xs text-dim hover:text-accent
                         disabled:opacity-30 transition-colors">
              <Zap className="w-3.5 h-3.5" />
              Build FAISS Index
            </button>
            {faissMsg && <span className="text-xs text-ghost">{faissMsg}</span>}
            {indexed > 0 && (
              <button onClick={handleClear} disabled={isRunning}
                className="flex items-center gap-1.5 text-xs text-dim hover:text-error disabled:opacity-30 transition-colors">
                <RotateCcw className="w-3.5 h-3.5" /> Clear Index
              </button>
            )}
          </div>
        </>
      )}
    </div>
  )
}
