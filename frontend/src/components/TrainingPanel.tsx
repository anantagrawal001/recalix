import { useState, useCallback, useEffect, useRef } from 'react'
import { BrainCircuit, Play, Trash2, Zap, ChevronDown, AlertCircle, CheckCircle, Loader2, TerminalSquare, RefreshCw } from 'lucide-react'
import { cn } from '@/lib/utils'
import {
  startTraining, getTrainingStatus, getFusionStatus, deleteModel,
  createTrainingStream, type TrainingStatus, type FusionStatus,
} from '@/lib/api'

const MODES = [
  { id: 'basic',     label: 'Basic',     desc: 'Random negatives. Fast & reliable. Good first run.',              color: 'text-info' },
  { id: 'baseline',  label: 'Baseline',  desc: 'Trains vs fixed baselines. Generates research comparison table.', color: 'text-soft' },
  { id: 'semi_hard', label: 'Semi-Hard', desc: '50% random + 50% semi-hard negatives. RECOMMENDED.',             color: 'text-accent' },
  { id: 'hard',      label: 'Hard',      desc: 'Full hard negative mining. Highest potential, needs stability.',  color: 'text-warning' },
] as const

type ModeId = typeof MODES[number]['id']

function LogLine({ line, i }: { line: string; i: number }) {
  const isError   = line.includes('ERROR') || line.includes('✗')
  const isSuccess = line.includes('✓') || line.includes('complete')
  const isWarn    = line.includes('⚠') || line.includes('spike')
  const isMetric  = line.includes('R@') || line.includes('recall') || line.includes('loss=')
  const isSection = line.includes('===') || line.includes('───')

  return (
    <div className={cn(
      'font-mono text-xs leading-relaxed log-line px-3 py-0.5 rounded',
      isError   ? 'text-error bg-error/5'   :
      isSuccess ? 'text-success'             :
      isWarn    ? 'text-warning'             :
      isMetric  ? 'text-accent'              :
      isSection ? 'text-bright font-semibold':
                  'text-ghost',
    )} style={{ animationDelay: `${Math.min(i, 20) * 0.02}s` }}>
      {line}
    </div>
  )
}

export function TrainingPanel() {
  const [mode,       setMode]       = useState<ModeId>('semi_hard')
  const [epochs,     setEpochs]     = useState(20)
  const [lr,         setLr]         = useState(0.001)
  const [batchSize,  setBatchSize]  = useState(64)
  const [status,     setStatus]     = useState<TrainingStatus | null>(null)
  const [fusion,     setFusion]     = useState<FusionStatus | null>(null)
  const [logs,       setLogs]       = useState<string[]>([])
  const [showConfig, setShowConfig] = useState(false)
  const [error,      setError]      = useState('')
  const logRef  = useRef<HTMLDivElement>(null)
  const esRef   = useRef<EventSource | null>(null)

  const loadStatus = useCallback(async () => {
    try { setStatus(await getTrainingStatus()) } catch {}
    try { setFusion(await getFusionStatus()) } catch {}
  }, [])

  useEffect(() => { loadStatus() }, [loadStatus])

  // Auto-scroll logs to bottom
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [logs])

  const handleStart = useCallback(async () => {
    setError(''); setLogs([])
    try {
      await startTraining(mode, epochs, lr, batchSize)

      // Close previous SSE
      if (esRef.current) { esRef.current.close(); esRef.current = null }

      const es = createTrainingStream()
      esRef.current = es

      es.onmessage = (ev) => {
        const data = JSON.parse(ev.data)
        if (data.log)       setLogs(prev => [...prev.slice(-1000), data.log])
        if (data.heartbeat) loadStatus()
        if (data.done) {
          es.close(); esRef.current = null; loadStatus()
        }
      }
      es.onerror = () => { es.close(); esRef.current = null; loadStatus() }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to start training')
    }
  }, [mode, epochs, lr, batchSize, loadStatus])

  const handleDelete = useCallback(async () => {
    if (!confirm('Delete the trained model? Search will revert to fixed weights.')) return
    try { await deleteModel(); await loadStatus() }
    catch (e) { setError(e instanceof Error ? e.message : 'Failed') }
  }, [loadStatus])

  const isRunning = status?.status === 'running'
  const epochPct  = status && status.total_epochs > 0
    ? Math.round((status.epoch / status.total_epochs) * 100) : 0

  return (
    <div className="space-y-6">
      {/* ── Fusion model status card ──────────────────────────────────────── */}
      {fusion && (
        <div className={cn(
          'glass border rounded-2xl p-4 flex items-center gap-4',
          fusion.fusion_active ? 'border-success/30 bg-success/5' : 'border-border/50',
        )}>
          <div className={cn(
            'w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0',
            fusion.fusion_active ? 'bg-success/20' : 'bg-surface',
          )}>
            <BrainCircuit className={cn('w-5 h-5', fusion.fusion_active ? 'text-success' : 'text-ghost')} />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-bright">
              {fusion.fusion_active ? 'Fusion Model Active' : 'Using Fixed Weights (No Model)'}
            </p>
            <p className="text-xs text-ghost truncate">
              {fusion.fusion_active
                ? `Query-adaptive scoring · device: ${fusion.device}`
                : `Fallback: visual=${fusion.fallback_weights.visual}, text=${fusion.fallback_weights.text}, filename=${fusion.fallback_weights.filename}`}
            </p>
          </div>
          {fusion.fusion_active && (
            <button onClick={handleDelete}
              className="p-2 rounded-lg hover:bg-error/15 text-ghost hover:text-error transition-colors flex-shrink-0"
              title="Delete model">
              <Trash2 className="w-4 h-4" />
            </button>
          )}
          <button onClick={loadStatus} className="p-2 rounded-lg hover:bg-lift text-ghost transition-colors flex-shrink-0">
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* ── Mode selector ─────────────────────────────────────────────────── */}
      <div className="glass border border-border/50 rounded-2xl p-5 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="font-display font-semibold text-bright flex items-center gap-2">
            <Zap className="w-4 h-4 text-accent" />
            Training Mode
          </h3>
          <button onClick={() => setShowConfig(c => !c)}
            className="flex items-center gap-1.5 text-xs text-dim hover:text-soft transition-colors">
            Advanced <ChevronDown className={cn('w-3.5 h-3.5 transition-transform', showConfig && 'rotate-180')} />
          </button>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {MODES.map(m => (
            <button key={m.id} onClick={() => setMode(m.id)} disabled={isRunning}
              className={cn(
                'p-3 rounded-xl border text-left transition-all duration-150',
                mode === m.id
                  ? 'bg-accent/10 border-accent/40 shadow-glow-sm'
                  : 'bg-surface border-border/40 hover:border-accent/20 hover:bg-lift',
                'disabled:opacity-50 disabled:cursor-not-allowed',
              )}>
              <div className="flex items-center gap-2 mb-1">
                <div className={cn(
                  'w-2 h-2 rounded-full',
                  mode === m.id ? 'bg-accent' : 'bg-border',
                )} />
                <span className={cn('text-sm font-medium', mode === m.id ? m.color : 'text-soft')}>{m.label}</span>
              </div>
              <p className="text-xs text-ghost pl-4 leading-relaxed">{m.desc}</p>
            </button>
          ))}
        </div>

        {/* Advanced config */}
        {showConfig && (
          <div className="grid grid-cols-3 gap-3 pt-2 border-t border-border/30 animate-fade-in">
            {[
              { label: 'Epochs',     value: epochs,    set: setEpochs,    min: 1,  max: 100, step: 1,     type: 'int'   },
              { label: 'Batch Size', value: batchSize, set: setBatchSize, min: 8,  max: 256, step: 8,     type: 'int'   },
              { label: 'Learn Rate', value: lr,        set: setLr,        min: 1e-5, max:0.01,step:1e-5, type: 'float' },
            ].map(({ label, value, set, min, max, step, type }) => (
              <div key={label} className="space-y-1">
                <label className="text-xs text-ghost">{label}</label>
                <input type="number" value={value} step={step} min={min} max={max}
                  onChange={e => (set as any)(type === 'float' ? parseFloat(e.target.value) : parseInt(e.target.value))}
                  disabled={isRunning}
                  className="w-full px-3 py-1.5 rounded-lg bg-surface border border-border/50 text-xs text-bright
                             font-mono focus:outline-none focus:border-accent/50 disabled:opacity-50" />
              </div>
            ))}
          </div>
        )}

        {error && (
          <div className="flex items-center gap-2 text-xs text-error">
            <AlertCircle className="w-3.5 h-3.5" /> {error}
          </div>
        )}

        {/* Start button */}
        <button onClick={handleStart} disabled={isRunning}
          className="w-full py-2.5 rounded-xl bg-accent-grad text-white text-sm font-semibold
                     flex items-center justify-center gap-2 shadow-glow hover:opacity-90 active:scale-[0.99]
                     disabled:opacity-40 disabled:cursor-not-allowed disabled:active:scale-100 transition-all">
          {isRunning
            ? <><Loader2 className="w-4 h-4 animate-spin" /> Training {status?.mode} — epoch {status?.epoch}/{status?.total_epochs}</>
            : <><Play className="w-4 h-4" /> Start Training</>}
        </button>

        {/* Epoch progress */}
        {isRunning && (
          <div className="space-y-1.5 animate-fade-in">
            <div className="h-1 rounded-full bg-surface overflow-hidden">
              <div className="h-full bg-accent-grad rounded-full transition-all duration-500"
                style={{ width: `${epochPct}%` }} />
            </div>
            <div className="flex justify-between text-xs text-ghost font-mono">
              <span>loss: {status?.loss ?? '—'}</span>
              <span>recall@5: {status?.recall_at5 ? (status.recall_at5 * 100).toFixed(1) + '%' : '—'}</span>
            </div>
          </div>
        )}

        {/* Done / error */}
        {status?.status === 'done' && (
          <div className="flex items-center gap-2 p-3 rounded-xl bg-success/10 border border-success/20 text-success text-xs animate-fade-in">
            <CheckCircle className="w-4 h-4 flex-shrink-0" />
            Training complete — model saved & loaded into search engine
          </div>
        )}
        {status?.status === 'error' && (
          <div className="flex items-center gap-2 p-3 rounded-xl bg-error/10 border border-error/20 text-error text-xs animate-fade-in">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            {status.message}
          </div>
        )}
      </div>

      {/* ── Live log terminal ─────────────────────────────────────────────── */}
      <div className="glass border border-border/50 rounded-2xl overflow-hidden">
        <div className="flex items-center gap-2 px-4 py-2.5 border-b border-border/30 bg-surface/50">
          <TerminalSquare className="w-4 h-4 text-accent" />
          <span className="text-xs font-mono font-medium text-soft">Training Log</span>
          <div className="ml-auto flex items-center gap-1.5">
            {isRunning && <div className="w-2 h-2 rounded-full bg-success animate-pulse" />}
            <button onClick={() => setLogs([])}
              className="text-xs text-ghost hover:text-soft transition-colors">Clear</button>
          </div>
        </div>
        <div ref={logRef}
          className="h-72 overflow-y-auto p-3 bg-void/60 space-y-0.5"
          style={{ fontVariantNumeric: 'tabular-nums' }}>
          {logs.length === 0 ? (
            <p className="text-xs text-ghost font-mono px-1 py-2">
              {isRunning ? 'Waiting for logs…' : 'Start training to see live logs here.'}
            </p>
          ) : (
            logs.map((line, i) => <LogLine key={i} line={line} i={i} />)
          )}
        </div>
      </div>

      {/* ── CLI instructions ──────────────────────────────────────────────── */}
      <div className="glass border border-border/40 rounded-2xl p-4 space-y-2">
        <p className="text-xs font-medium text-soft flex items-center gap-1.5">
          <TerminalSquare className="w-3.5 h-3.5 text-dim" />
          Run training from CLI (no backend needed):
        </p>
        <div className="bg-void rounded-xl p-3 font-mono text-xs text-ghost space-y-1">
          <p><span className="text-dim">cd</span> <span className="text-bright">recalix_v9/training</span></p>
          <p><span className="text-dim">python</span> <span className="text-accent">train.py</span> <span className="text-warning">--mode semi_hard --epochs 20</span></p>
          <p className="text-ghost/50 pt-1">Other modes: basic | baseline | semi_hard | hard</p>
        </div>
      </div>
    </div>
  )
}
