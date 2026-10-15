import { useEffect, useState, useMemo } from 'react'
import { TrainingPanel } from '@/components/TrainingPanel'
import {
  Activity,
  Archive,
  BarChart3,
  BookOpen,
  BrainCircuit,
  ChevronDown,
  ChevronRight,
  Clock,
  Database,
  FileText,
  GitCompare,
  Hash,
  Layers,
  LineChart,
  Percent,
  TrendingUp,
} from 'lucide-react'
import { assetUrl, getArtifacts, type TrainingArtifact } from '@/lib/api'
import { cn } from '@/lib/utils'

type Metric = { epoch: number; loss: number; val_acc: number }
type ProcessStep = { stage: string; detail: string; [key: string]: any }

const MODE_TABS = [
  { id: 'all',       label: 'All Runs',   color: 'text-bright',  bg: 'bg-white/10',         border: 'border-white/20' },
  { id: 'basic',     label: 'Basic',      color: 'text-info',    bg: 'bg-blue-500/10',      border: 'border-blue-500/30' },
  { id: 'baseline',  label: 'Baseline',   color: 'text-soft',    bg: 'bg-slate-400/10',     border: 'border-slate-400/30' },
  { id: 'semi_hard', label: 'Semi-Hard',  color: 'text-accent',  bg: 'bg-purple-500/10',    border: 'border-purple-500/30' },
  { id: 'hard',      label: 'Hard',       color: 'text-warning', bg: 'bg-amber-500/10',     border: 'border-amber-500/30' },
] as const

type ModeFilter = typeof MODE_TABS[number]['id']

export default function TrainingPage() {
  const [artifacts, setArtifacts] = useState<TrainingArtifact[]>([])
  const [modeFilter, setModeFilter] = useState<ModeFilter>('all')
  const [selectedId, setSelectedId] = useState<string | null>(null)

  useEffect(() => {
    getArtifacts().then(r => setArtifacts(r.artifacts)).catch(() => {})
  }, [])

  // Count runs per mode
  const modeCounts = useMemo(() => {
    const counts: Record<string, number> = { all: artifacts.length }
    for (const a of artifacts) {
      counts[a.mode] = (counts[a.mode] || 0) + 1
    }
    return counts
  }, [artifacts])

  // Filter and select
  const filtered = useMemo(() => {
    if (modeFilter === 'all') return artifacts
    return artifacts.filter(a => a.mode === modeFilter)
  }, [artifacts, modeFilter])

  const selectedArtifact = useMemo(() => {
    if (!selectedId) return filtered[0] ?? null
    return filtered.find(a => a.session_id === selectedId) ?? filtered[0] ?? null
  }, [filtered, selectedId])

  return (
    <div className="max-w-7xl mx-auto p-5 lg:p-8 space-y-5">
      <div>
        <h1 className="font-display text-2xl font-semibold text-bright flex items-center gap-3">
          <BrainCircuit className="w-6 h-6 text-accent" />
          Model Training
        </h1>
        <p className="text-sm text-ghost mt-1">
          Train the Query-Conditioned Fusion model and audit how examples, baselines, and gains change across runs.
        </p>
      </div>

      <div className="glass border border-accent/20 rounded-2xl p-4 space-y-2">
        <div className="flex items-center gap-2 text-sm font-medium text-accent">
          <BookOpen className="w-4 h-4" />
          Research Context
        </div>
        <p className="text-xs text-ghost leading-relaxed">
          The fusion model predicts per-query weights across visual, OCR, filename, and structural signals.
          A negative gain can happen when the fixed visual-heavy baseline is already very strong, when synthetic
          positives are noisy, or when evaluation only rewards global pairwise accuracy. The run history below now
          shows the graph, examples, baselines, gain, and the data-building process for each approach.
        </p>
      </div>

      <TrainingPanel onTrainingComplete={() => {
        getArtifacts().then(r => setArtifacts(r.artifacts)).catch(() => {})
      }} />

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* RESEARCH-GRADE TRAINING HISTORY                                    */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      <div className="glass border border-border/50 rounded-2xl overflow-hidden">
        {/* ── Mode Tabs ─────────────────────────────────────────────── */}
        <div className="flex items-center gap-1 px-4 py-3 border-b border-border/30 overflow-x-auto">
          <Archive className="w-4 h-4 text-dim shrink-0 mr-1" />
          <span className="text-xs font-medium text-soft shrink-0 mr-3">Training History</span>
          {MODE_TABS.map(tab => {
            const count = modeCounts[tab.id] || 0
            const active = modeFilter === tab.id
            return (
              <button
                key={tab.id}
                onClick={() => { setModeFilter(tab.id); setSelectedId(null) }}
                className={cn(
                  'px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-150 shrink-0',
                  'flex items-center gap-1.5',
                  active
                    ? `${tab.bg} ${tab.border} border ${tab.color} shadow-sm`
                    : 'text-ghost hover:text-soft hover:bg-surface border border-transparent'
                )}
              >
                {tab.label}
                {count > 0 && (
                  <span className={cn(
                    'text-[10px] px-1.5 py-0.5 rounded-full font-mono',
                    active ? 'bg-white/10' : 'bg-surface'
                  )}>
                    {count}
                  </span>
                )}
              </button>
            )
          })}
        </div>

        {filtered.length === 0 ? (
          <p className="text-xs text-ghost px-4 py-8 text-center">
            {modeFilter === 'all'
              ? 'No training sessions yet. Start training to create a record.'
              : `No ${modeFilter.replace('_', '-')} runs yet.`}
          </p>
        ) : (
          <div className="flex flex-col lg:flex-row">
            {/* ── Run History Table ──────────────────────────────── */}
            <div className="lg:w-[380px] border-b lg:border-b-0 lg:border-r border-border/30 overflow-y-auto max-h-[28rem]">
              <table className="w-full text-xs">
                <thead className="sticky top-0 bg-void/90 backdrop-blur-sm z-10">
                  <tr className="border-b border-border/30">
                    <th className="px-3 py-2 text-left text-ghost font-medium">Date</th>
                    <th className="px-2 py-2 text-left text-ghost font-medium">Mode</th>
                    <th className="px-2 py-2 text-right text-ghost font-medium">Acc</th>
                    <th className="px-2 py-2 text-right text-ghost font-medium">Gain</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((a) => {
                    const result = a.result || {}
                    const bestAcc = Number(result.best_val_acc ?? result.best_fusion_acc ?? 0)
                    const gain = Number(result.gain ?? result.gain_vs_best_fixed ?? 0)
                    const isSelected = selectedArtifact?.session_id === a.session_id
                    const modeTab = MODE_TABS.find(t => t.id === a.mode)

                    return (
                      <tr
                        key={a.session_id}
                        onClick={() => setSelectedId(a.session_id)}
                        className={cn(
                          'cursor-pointer transition-colors border-b border-border/10',
                          isSelected
                            ? 'bg-accent/10 border-l-2 border-l-accent'
                            : 'hover:bg-lift/40 border-l-2 border-l-transparent'
                        )}
                      >
                        <td className="px-3 py-2.5 font-mono text-ghost whitespace-nowrap">
                          {a.finished_at
                            ? new Date(a.finished_at).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' })
                            : a.session_id}
                        </td>
                        <td className="px-2 py-2.5">
                          <span className={cn('text-[10px] font-semibold uppercase tracking-wide', modeTab?.color || 'text-soft')}>
                            {a.mode.replace('_', '-')}
                          </span>
                        </td>
                        <td className="px-2 py-2.5 text-right font-mono text-bright">
                          {bestAcc > 0 ? `${(bestAcc * 100).toFixed(1)}%` : '—'}
                        </td>
                        <td className={cn(
                          'px-2 py-2.5 text-right font-mono',
                          gain >= 0 ? 'text-success' : 'text-warning'
                        )}>
                          {gain !== 0 ? `${gain >= 0 ? '+' : ''}${(gain * 100).toFixed(2)}%` : '—'}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>

            {/* ── Selected Run Detail Panel ─────────────────────── */}
            <div className="flex-1 min-w-0 p-4 overflow-y-auto max-h-[28rem]">
              {selectedArtifact ? (
                <RunDetailPanel artifact={selectedArtifact} />
              ) : (
                <p className="text-xs text-ghost py-8 text-center">Select a run to view details</p>
              )}
            </div>
          </div>
        )}

        {/* ── Bottom Progress Dashboard ─────────────────────────── */}
        {selectedArtifact && (
          <div className="border-t border-border/30">
            <RunProgressDashboard artifact={selectedArtifact} />
          </div>
        )}
      </div>
    </div>
  )
}


// ═══════════════════════════════════════════════════════════════════════════════
// RUN DETAIL PANEL — shown when a run is selected from the table
// ═══════════════════════════════════════════════════════════════════════════════

function RunDetailPanel({ artifact: a }: { artifact: TrainingArtifact }) {
  const result = a.result || {}
  const dataset = a.summary?.dataset || {}
  const metrics = ((a.metrics || a.summary?.metrics || []) as Metric[])
  const baselines = a.baselines || a.summary?.baselines || {}
  const process = ((a.process || a.summary?.process || []) as ProcessStep[])
  const gain = Number(result.gain ?? result.gain_vs_best_fixed ?? 0)
  const bestAcc = Number(result.best_val_acc ?? result.best_fusion_acc ?? 0)
  const fixedAcc = Number(result.best_fixed_acc ?? 0)
  const trainExamples = Number(dataset.train_examples ?? 0)
  const valExamples = Number(dataset.val_examples ?? 0)
  const bestLoss = result.best_loss != null ? Number(result.best_loss) : null

  const [showEpochTable, setShowEpochTable] = useState(false)
  const [showProcess, setShowProcess] = useState(false)

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-bright uppercase tracking-wide">
            {a.mode.replace('_', '-')}
          </span>
          <span className="text-xs text-ghost">
            — {a.epochs} epoch{a.epochs !== 1 ? 's' : ''} · LR {a.lr} · Batch {a.batch_size}
          </span>
        </div>
        <div className="text-[11px] text-ghost font-mono">
          {a.finished_at ? new Date(a.finished_at).toLocaleString() : a.session_id}
        </div>
      </div>

      {/* Key Metrics Row */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-2">
        <MetricTile icon={Database} label="Train / Val" value={`${trainExamples.toLocaleString()} / ${valExamples.toLocaleString()}`} tone="text-info" />
        <MetricTile icon={Activity} label="Best Accuracy" value={bestAcc ? `${(bestAcc * 100).toFixed(1)}%` : '—'} tone="text-success" />
        <MetricTile icon={GitCompare} label="Fixed Baseline" value={fixedAcc ? `${(fixedAcc * 100).toFixed(1)}%` : '—'} tone="text-soft" />
        <MetricTile icon={TrendingUp} label="Gain" value={gain !== 0 ? `${gain >= 0 ? '+' : ''}${(gain * 100).toFixed(2)}%` : '—'} tone={gain >= 0 ? 'text-success' : 'text-warning'} />
        <MetricTile icon={LineChart} label="Best Loss" value={bestLoss != null ? bestLoss.toFixed(4) : '—'} tone="text-accent" />
      </div>

      {/* Graph + Baselines */}
      <div className="grid lg:grid-cols-[1.5fr_1fr] gap-3">
        {/* Learning Curve */}
        <div className="glass border border-border/40 rounded-xl overflow-hidden bg-void/30">
          <div className="px-3 py-2 border-b border-border/25 flex items-center justify-between">
            <span className="text-xs font-mono text-accent flex items-center gap-1.5">
              <BarChart3 className="w-3.5 h-3.5" />Learning Curve
            </span>
            <span className="text-[11px] text-ghost">{metrics.length} epoch{metrics.length !== 1 ? 's' : ''}</span>
          </div>
          <div className="p-3 min-h-52 flex items-center justify-center">
            {a.learning_curve_url ? (
              <img
                src={assetUrl(a.learning_curve_url)}
                alt={`${a.mode} learning curve`}
                className="w-full max-h-72 object-contain rounded-lg bg-white"
              />
            ) : metrics.length > 0 ? (
              <InlineMetricChart metrics={metrics} />
            ) : (
              <p className="text-xs text-ghost font-mono">No graph data saved for this run.</p>
            )}
          </div>
        </div>

        {/* Baseline Comparison Table */}
        <div className="glass border border-border/40 rounded-xl overflow-hidden bg-void/30">
          <div className="px-3 py-2 border-b border-border/25">
            <span className="text-xs font-mono text-accent flex items-center gap-1.5">
              <GitCompare className="w-3.5 h-3.5" />Baseline Comparison
            </span>
          </div>
          <div className="p-3">
            {Object.keys(baselines).length > 0 ? (
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-border/25">
                    <th className="text-left py-1.5 text-ghost font-medium">Strategy</th>
                    <th className="text-right py-1.5 text-ghost font-medium">Accuracy</th>
                    <th className="text-right py-1.5 text-ghost font-medium">vs Trained</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(baselines).sort(([,a],[,b]) => Number(b) - Number(a)).map(([name, value]) => {
                    const acc = Number(value)
                    const diff = bestAcc - acc
                    return (
                      <tr key={name} className="border-b border-border/10">
                        <td className="py-1.5 text-soft">{name}</td>
                        <td className="py-1.5 text-right font-mono text-bright">{(acc * 100).toFixed(1)}%</td>
                        <td className={cn('py-1.5 text-right font-mono', diff >= 0 ? 'text-success' : 'text-warning')}>
                          {diff !== 0 ? `${diff >= 0 ? '+' : ''}${(diff * 100).toFixed(2)}%` : '—'}
                        </td>
                      </tr>
                    )
                  })}
                  {/* Trained model row */}
                  <tr className="bg-accent/5">
                    <td className="py-1.5 text-accent font-semibold">Trained (fusion)</td>
                    <td className="py-1.5 text-right font-mono text-accent font-semibold">
                      {bestAcc ? `${(bestAcc * 100).toFixed(1)}%` : '—'}
                    </td>
                    <td className="py-1.5 text-right font-mono text-accent">—</td>
                  </tr>
                </tbody>
              </table>
            ) : (
              <p className="text-xs text-ghost font-mono py-4">No baseline data for this run.</p>
            )}
          </div>
        </div>
      </div>

      {/* Collapsible: Epoch-by-Epoch Table */}
      {metrics.length > 0 && (
        <div className="glass border border-border/40 rounded-xl overflow-hidden">
          <button
            onClick={() => setShowEpochTable(s => !s)}
            className="flex items-center justify-between w-full px-3 py-2 text-xs font-medium text-soft hover:text-bright transition-colors"
          >
            <span className="flex items-center gap-1.5">
              <Hash className="w-3.5 h-3.5 text-dim" />
              Epoch-by-Epoch Metrics ({metrics.length} epochs)
            </span>
            {showEpochTable ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
          </button>
          {showEpochTable && (
            <div className="border-t border-border/25 max-h-60 overflow-y-auto">
              <table className="w-full text-xs">
                <thead className="sticky top-0 bg-void/90 backdrop-blur-sm">
                  <tr className="border-b border-border/25">
                    <th className="px-3 py-1.5 text-left text-ghost font-medium">Epoch</th>
                    <th className="px-3 py-1.5 text-right text-ghost font-medium">Loss</th>
                    <th className="px-3 py-1.5 text-right text-ghost font-medium">Val Accuracy</th>
                    <th className="px-3 py-1.5 text-right text-ghost font-medium">Δ Loss</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.map((m, i) => {
                    const prevLoss = i > 0 ? metrics[i - 1].loss : m.loss
                    const delta = m.loss - prevLoss
                    return (
                      <tr key={m.epoch} className="border-b border-border/10 hover:bg-lift/30">
                        <td className="px-3 py-1.5 font-mono text-soft">{m.epoch}</td>
                        <td className="px-3 py-1.5 text-right font-mono text-bright">{Number(m.loss).toFixed(4)}</td>
                        <td className="px-3 py-1.5 text-right font-mono text-success">
                          {m.val_acc ? `${(Number(m.val_acc) * 100).toFixed(1)}%` : '—'}
                        </td>
                        <td className={cn(
                          'px-3 py-1.5 text-right font-mono',
                          delta < 0 ? 'text-success' : delta > 0 ? 'text-warning' : 'text-ghost'
                        )}>
                          {i > 0 ? `${delta >= 0 ? '+' : ''}${delta.toFixed(4)}` : '—'}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Collapsible: Approach Process */}
      {process.length > 0 && (
        <div className="glass border border-border/40 rounded-xl overflow-hidden">
          <button
            onClick={() => setShowProcess(s => !s)}
            className="flex items-center justify-between w-full px-3 py-2 text-xs font-medium text-soft hover:text-bright transition-colors"
          >
            <span className="flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-dim" />
              Approach Process ({process.length} steps)
            </span>
            {showProcess ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
          </button>
          {showProcess && (
            <div className="border-t border-border/25 p-3 space-y-2 max-h-60 overflow-y-auto">
              {process.map((p, i) => (
                <div key={`${p.stage}-${i}`} className="rounded-lg bg-surface/60 border border-border/25 p-2">
                  <p className="text-[11px] uppercase tracking-wide text-soft font-semibold">{p.stage.replace('_', ' ')}</p>
                  <p className="text-xs text-ghost leading-relaxed mt-1">{p.detail}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Terminal log link */}
      {a.terminal_log_url && (
        <a
          href={assetUrl(a.terminal_log_url)}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg bg-accent/10 border border-accent/20 text-accent hover:bg-accent/15 transition-colors"
        >
          <FileText className="w-3.5 h-3.5" />
          View Full Terminal Log
        </a>
      )}
    </div>
  )
}


// ═══════════════════════════════════════════════════════════════════════════════
// RUN PROGRESS DASHBOARD — bottom section with ALL data for the selected run
// ═══════════════════════════════════════════════════════════════════════════════

function RunProgressDashboard({ artifact: a }: { artifact: TrainingArtifact }) {
  const result = a.result || {}
  const dataset = a.summary?.dataset || {}
  const buildInfo = a.summary?.build_info || result.build_info || result || {}

  // Extract all available stats
  const stats: [string, string | number][] = []

  // Dataset coverage
  const totalEmb = Number(buildInfo.total_embeddings ?? dataset.total_embeddings ?? 0)
  const negPool = Number(buildInfo.negative_pool_size ?? buildInfo.candidate_images ?? 0)
  const coveragePct = Number(buildInfo.negative_pool_coverage_pct ?? 0)
  const trainEx = Number(dataset.train_examples ?? buildInfo.examples ?? 0)
  const valEx = Number(dataset.val_examples ?? 0)
  const indexedRows = Number(buildInfo.indexed_rows ?? 0)
  const splitFrac = Number(buildInfo.split_fraction ?? 0)

  if (totalEmb) stats.push(['Total Embeddings', totalEmb.toLocaleString()])
  if (indexedRows) stats.push(['Indexed Rows', indexedRows.toLocaleString()])
  if (negPool) stats.push(['Negative Pool Size', negPool.toLocaleString()])
  if (coveragePct) stats.push(['Pool Coverage', `${coveragePct}%`])
  if (splitFrac) stats.push(['Train Split', `${(splitFrac * 100).toFixed(0)}%`])
  if (trainEx) stats.push(['Train Examples', trainEx.toLocaleString()])
  if (valEx) stats.push(['Val Examples', valEx.toLocaleString()])

  // Training hyperparams
  stats.push(['Mode', a.mode.replace('_', '-').toUpperCase()])
  stats.push(['Epochs', String(a.epochs)])
  stats.push(['Learning Rate', String(a.lr)])
  stats.push(['Batch Size', String(a.batch_size)])

  // Results
  const bestLoss = result.best_loss != null ? Number(result.best_loss) : null
  const bestAcc = Number(result.best_val_acc ?? result.best_fusion_acc ?? 0)
  const gain = Number(result.gain ?? result.gain_vs_best_fixed ?? 0)
  const fixedAcc = Number(result.best_fixed_acc ?? 0)
  const buildSecs = Number(buildInfo.build_seconds ?? 0)

  if (bestLoss != null) stats.push(['Best Loss', bestLoss.toFixed(4)])
  if (bestAcc) stats.push(['Best Accuracy', `${(bestAcc * 100).toFixed(2)}%`])
  if (fixedAcc) stats.push(['Best Fixed Baseline', `${(fixedAcc * 100).toFixed(2)}%`])
  if (gain) stats.push(['Gain vs Baseline', `${gain >= 0 ? '+' : ''}${(gain * 100).toFixed(2)}%`])
  if (buildSecs) stats.push(['Build Time', `${buildSecs.toFixed(1)}s`])

  // Query source counts
  const querySources = buildInfo.query_sources || {}
  for (const [src, count] of Object.entries(querySources)) {
    stats.push([`Queries: ${src}`, Number(count).toLocaleString()])
  }

  // Negative mining stats
  const negKinds = buildInfo.negative_kinds || {}
  for (const [kind, count] of Object.entries(negKinds)) {
    stats.push([`Negatives: ${kind}`, Number(count).toLocaleString()])
  }

  // Timing
  if (a.started_at && a.finished_at) {
    const dur = (new Date(a.finished_at).getTime() - new Date(a.started_at).getTime()) / 1000
    const mins = Math.floor(dur / 60)
    const secs = Math.round(dur % 60)
    stats.push(['Total Duration', `${mins}m ${secs}s`])
  }

  if (stats.length === 0) return null

  return (
    <div className="p-4 space-y-3">
      <div className="flex items-center gap-2 text-xs font-medium text-soft">
        <Percent className="w-3.5 h-3.5 text-dim" />
        Full Run Statistics
        <span className="text-ghost">({stats.length} data points)</span>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2">
        {stats.map(([label, value], i) => (
          <div key={`${label}-${i}`} className="rounded-lg bg-surface/50 border border-border/20 px-2.5 py-2">
            <p className="text-[10px] text-ghost truncate">{label}</p>
            <p className="text-xs font-mono text-bright font-medium mt-0.5 truncate">{value}</p>
          </div>
        ))}
      </div>
    </div>
  )
}


// ═══════════════════════════════════════════════════════════════════════════════
// SHARED COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

function MetricTile({ icon: Icon, label, value, tone }: { icon: any; label: string; value: string; tone: string }) {
  return (
    <div className="rounded-xl bg-surface/70 border border-border/30 p-3 min-w-0">
      <div className="flex items-center gap-1.5 text-[11px] text-ghost">
        <Icon className={cn('w-3.5 h-3.5', tone)} />
        {label}
      </div>
      <p className={cn('mt-1 text-sm font-mono font-semibold truncate', tone)}>{value}</p>
    </div>
  )
}

function InlineMetricChart({ metrics }: { metrics: Metric[] }) {
  const width = 620
  const height = 220
  const pad = 32
  const epochs = metrics.map((m, i) => Number(m.epoch ?? i + 1))
  const losses = metrics.map(m => Number(m.loss ?? 0))
  const accs = metrics.map(m => Number(m.val_acc ?? 0))
  const minEpoch = Math.min(...epochs)
  const maxEpoch = Math.max(...epochs)
  const minLoss = Math.min(...losses)
  const maxLoss = Math.max(...losses)
  const scaleX = (x: number) => pad + ((x - minEpoch) / Math.max(maxEpoch - minEpoch, 1)) * (width - pad * 2)
  const scaleLoss = (y: number) => height - pad - ((y - minLoss) / Math.max(maxLoss - minLoss, 0.0001)) * (height - pad * 2)
  const scaleAcc = (y: number) => height - pad - Math.max(0, Math.min(1, y)) * (height - pad * 2)

  const lossPoints = metrics.map((m, i) => `${scaleX(epochs[i])},${scaleLoss(Number(m.loss ?? 0))}`).join(' ')
  const accPoints = metrics.map((m, i) => `${scaleX(epochs[i])},${scaleAcc(Number(m.val_acc ?? 0))}`).join(' ')

  // Grid lines
  const yGridCount = 4
  const yGridLines = Array.from({ length: yGridCount + 1 }, (_, i) => {
    const frac = i / yGridCount
    const y = pad + frac * (height - pad * 2)
    const lossVal = maxLoss - frac * (maxLoss - minLoss)
    return { y, label: lossVal.toFixed(3) }
  })

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-56" role="img" aria-label="Training metric chart">
      <rect x="0" y="0" width={width} height={height} rx="10" fill="#090912" />
      {/* Grid lines */}
      {yGridLines.map((g, i) => (
        <g key={i}>
          <line x1={pad} y1={g.y} x2={width - pad} y2={g.y} stroke="#1a1a2e" strokeDasharray="3,3" />
          <text x={pad - 4} y={g.y + 3} fill="#4a4a6a" fontSize="9" textAnchor="end" fontFamily="JetBrains Mono">{g.label}</text>
        </g>
      ))}
      {/* Axes */}
      <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="#252542" />
      <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="#252542" />
      {/* Loss area fill */}
      <polygon
        points={`${scaleX(epochs[0])},${height - pad} ${lossPoints} ${scaleX(epochs[epochs.length - 1])},${height - pad}`}
        fill="url(#lossGrad)" opacity="0.15"
      />
      {/* Loss line */}
      <polyline points={lossPoints} fill="none" stroke="#8b5cf6" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
      {/* Accuracy line */}
      <polyline points={accPoints} fill="none" stroke="#22c55e" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
      {/* Data point dots */}
      {metrics.map((m, i) => (
        <g key={i}>
          <circle cx={scaleX(epochs[i])} cy={scaleLoss(Number(m.loss ?? 0))} r="2.5" fill="#8b5cf6" />
          <circle cx={scaleX(epochs[i])} cy={scaleAcc(Number(m.val_acc ?? 0))} r="2.5" fill="#22c55e" />
        </g>
      ))}
      {/* Gradient definition */}
      <defs>
        <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#8b5cf6" />
          <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0" />
        </linearGradient>
      </defs>
      {/* Legend */}
      <rect x={pad + 8} y={pad - 6} width="8" height="8" rx="2" fill="#8b5cf6" />
      <text x={pad + 20} y={pad + 1} fill="#a78bfa" fontSize="10" fontFamily="JetBrains Mono">loss</text>
      <rect x={width - pad - 72} y={pad - 6} width="8" height="8" rx="2" fill="#22c55e" />
      <text x={width - pad - 60} y={pad + 1} fill="#22c55e" fontSize="10" fontFamily="JetBrains Mono">val_acc</text>
      {/* Epoch labels */}
      <text x={scaleX(minEpoch)} y={height - pad + 14} fill="#4a4a6a" fontSize="9" textAnchor="middle" fontFamily="JetBrains Mono">{minEpoch}</text>
      <text x={scaleX(maxEpoch)} y={height - pad + 14} fill="#4a4a6a" fontSize="9" textAnchor="middle" fontFamily="JetBrains Mono">{maxEpoch}</text>
    </svg>
  )
}
