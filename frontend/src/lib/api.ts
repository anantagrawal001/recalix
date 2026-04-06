/**
 * RECALIX V9 — API Client
 * All backend communication in one place.
 */

const BASE = 'http://localhost:8000'

// ── Types ─────────────────────────────────────────────────────────────────────

export interface MediaResult {
  id: number
  filename: string
  file_path: string
  folder: string
  file_type: 'image' | 'video'
  width: number | null
  height: number | null
  file_size: number | null
  modified_at: string | null
  created_at: string | null
  embedding_index: number | null
  thumbnail_url: string | null
  score: number
  _search_backend?: string
  _signals?: { visual: number; text: number; filename: number; structural: number }
  _weights?: { visual: number; text: number; filename: number; structural: number }
}

export interface SearchResponse {
  query: string
  results: MediaResult[]
  total: number
  page: number
  page_size: number
  total_pages: number
  backend?: string
  message?: string
}

export interface IndexStatus {
  status: 'idle' | 'scanning' | 'indexing' | 'done' | 'error'
  total: number
  processed: number
  skipped: number
  errors: number
  new_indexed: number
  current_file: string
  started_at: string | null
  finished_at: string | null
  message: string
  db_stats: { total: number; images: number; videos: number; last_indexed: string | null }
  embedding_count: number
}

export interface FolderItem { folder: string; count: number }

export interface TrainingStatus {
  status: 'idle' | 'running' | 'done' | 'error'
  mode: string | null
  epoch: number
  total_epochs: number
  loss: number | null
  recall_at5: number | null
  message: string
  started_at: string | null
  finished_at: string | null
}

export interface FusionStatus {
  fusion_active: boolean
  model_path: string
  model_exists: boolean
  device: string | null
  fallback_weights: { visual: number; text: number; filename: number; structural: number }
}

export interface HealthResponse {
  status: string
  version: string
  indexed: number
  search_backend: string
  fusion: boolean
}

// ── Helpers ───────────────────────────────────────────────────────────────────

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { ...init, headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) } })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail ?? `Request failed: ${res.status}`)
  }
  return res.json()
}

// ── Health ────────────────────────────────────────────────────────────────────

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE}/api/health`, { signal: AbortSignal.timeout(3000) })
    return res.ok
  } catch { return false }
}

export async function getHealth(): Promise<HealthResponse> {
  return apiFetch('/api/health')
}

// ── Search ────────────────────────────────────────────────────────────────────

export async function search(params: {
  q: string; type?: string | null; folder?: string | null
  date_from?: string | null; date_to?: string | null
  top_k?: number; page?: number; signal?: AbortSignal
}): Promise<SearchResponse> {
  const { signal, ...rest } = params
  const url = new URL('/api/search', BASE)
  url.searchParams.set('q', rest.q)
  if (rest.type)      url.searchParams.set('type', rest.type)
  if (rest.folder)    url.searchParams.set('folder', rest.folder)
  if (rest.date_from) url.searchParams.set('date_from', rest.date_from)
  if (rest.date_to)   url.searchParams.set('date_to', rest.date_to)
  url.searchParams.set('top_k', String(rest.top_k ?? 50))
  url.searchParams.set('page',  String(rest.page  ?? 1))
  const res = await fetch(url.toString(), { signal })
  if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.detail ?? `Search failed: ${res.status}`) }
  return res.json()
}

// ── Indexing ──────────────────────────────────────────────────────────────────

export async function startIndexing(directory: string, reindex = false) {
  return apiFetch('/api/index', { method: 'POST', body: JSON.stringify({ directory, reindex }) })
}
export async function getIndexStatus(): Promise<IndexStatus> { return apiFetch('/api/index/status') }
export async function clearIndex() { return apiFetch('/api/index', { method: 'DELETE' }) }
export async function selectDirectory(): Promise<string | null> {
  const r = await apiFetch<{ directory: string | null }>('/api/select-directory', { method: 'POST' })
  return r.directory
}
export async function getFolders(): Promise<FolderItem[]> {
  const r = await apiFetch<{ folders: FolderItem[] }>('/api/folders')
  return r.folders
}
export async function getStats() { return apiFetch<any>('/api/stats') }

// ── Files ─────────────────────────────────────────────────────────────────────

export async function openFile(mediaId: number, showInFolder = false) {
  return apiFetch('/api/open-file', { method: 'POST', body: JSON.stringify({ media_id: mediaId, show_in_folder: showInFolder }) })
}

// ── Training ──────────────────────────────────────────────────────────────────

export async function startTraining(mode: string, epochs: number, lr: number, batchSize: number) {
  return apiFetch('/api/training/start', {
    method: 'POST',
    body: JSON.stringify({ mode, epochs, lr, batch_size: batchSize }),
  })
}
export async function getTrainingStatus(): Promise<TrainingStatus> { return apiFetch('/api/training/status') }
export async function getFusionStatus(): Promise<FusionStatus> { return apiFetch('/api/training/fusion-status') }
export async function buildFaissIndex() { return apiFetch('/api/training/build-faiss', { method: 'POST' }) }
export async function deleteModel() { return apiFetch('/api/training/model', { method: 'DELETE' }) }

/** Create EventSource for SSE log streaming */
export function createTrainingStream(): EventSource {
  return new EventSource(`${BASE}/api/training/stream`)
}
