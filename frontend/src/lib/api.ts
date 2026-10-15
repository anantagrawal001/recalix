const BASE = 'http://localhost:8000'

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  })
  if (!res.ok) {
    const e = await res.json().catch(() => ({}))
    throw new Error(e.detail ?? `${res.status}`)
  }
  return res.json()
}

// ── Types ─────────────────────────────────────────────────────────────────────
export interface MediaResult {
  id: number; filename: string; file_path: string; folder: string
  file_type: 'image'|'video'; width: number|null; height: number|null
  file_size: number|null; modified_at: string|null; embedding_index: number|null
  thumbnail_url: string|null; score: number
  _search_backend?: string
  _signals?: {visual:number; text:number; filename:number; structural:number}
  _weights?:  {visual:number; text:number; filename:number; structural:number}
  ocr_text?: string; filename_tokens?: string
}
export interface SearchResponse {
  query:string; results:MediaResult[]; total:number; page:number
  page_size:number; total_pages:number; backend?:string; message?:string
}
export interface GalleryResponse {
  folder:string|null; results:MediaResult[]; total:number; page:number
  page_size:number; total_pages:number
}
export interface IndexStatus {
  status:'idle'|'scanning'|'indexing'|'done'|'error'
  total:number; processed:number; skipped:number; errors:number; new_indexed:number
  current_file:string; started_at:string|null; finished_at:string|null; message:string
  detail?: {
    stage?: string; file_type?: string; batch_size?: number; embedding_index?: number
    width?: number|null; height?: number|null; ocr_chars?: number; filename_tokens?: string
    is_document?: boolean; is_screenshot?: boolean; device?: string
  }
  db_stats:{total:number; images:number; videos:number; last_indexed:string|null}
  embedding_count:number
}
export interface TrainingStatus {
  status:'idle'|'running'|'done'|'error'; mode:string|null
  epoch:number; total_epochs:number; loss:number|null; recall_at5:number|null
  message:string; started_at:string|null; finished_at:string|null
}
export interface FusionStatus {
  fusion_active:boolean; model_path:string; model_exists:boolean
  device:string|null; fallback_weights:{visual:number;text:number;filename:number;structural:number}
}

// ── Health ────────────────────────────────────────────────────────────────────
export const checkHealth = async () => {
  try { const r = await fetch(`${BASE}/api/health`,{signal:AbortSignal.timeout(3000)}); return r.ok }
  catch { return false }
}
export const getHealth = () => apiFetch<any>('/api/health')

// ── Search ────────────────────────────────────────────────────────────────────
export async function search(params: {
  q:string; type?:string|null; folder?:string|null
  date_from?:string|null; date_to?:string|null; ext?:string|null; top_k?:number; page?:number
  signal?:AbortSignal
}): Promise<SearchResponse> {
  const { signal, ...rest } = params
  const url = new URL('/api/search', BASE)
  url.searchParams.set('q', rest.q)
  if (rest.type)      url.searchParams.set('type',      rest.type)
  if (rest.folder)    url.searchParams.set('folder',    rest.folder)
  if (rest.date_from) url.searchParams.set('date_from', rest.date_from)
  if (rest.date_to)   url.searchParams.set('date_to',   rest.date_to)
  if (rest.ext)       url.searchParams.set('ext',       rest.ext)
  url.searchParams.set('top_k', String(rest.top_k ?? 100))
  url.searchParams.set('page',  String(rest.page  ?? 1))
  const res = await fetch(url.toString(), { signal })
  if (!res.ok) { const e = await res.json().catch(()=>({})); throw new Error(e.detail??`${res.status}`) }
  return res.json()
}

// ── Indexing ──────────────────────────────────────────────────────────────────
export const startIndexing = (directory:string, reindex=false) =>
  apiFetch<{message:string; started:boolean; directory?:string}>('/api/index', {method:'POST', body:JSON.stringify({directory,reindex})})
export const getIndexStatus = () => apiFetch<IndexStatus>('/api/index/status')
export const clearIndex     = () => apiFetch('/api/index', {method:'DELETE'})
export const clearIndexFolder = (folder: string, includeSubfolders = true) =>
  apiFetch<{message:string; deleted:number; folder:string; include_subfolders:boolean}>(`/api/index/folder?folder=${encodeURIComponent(folder)}&include_subfolders=${includeSubfolders}`, {method:'DELETE'})
export const getFolders     = async () => {
  const r = await apiFetch<{folders:{folder:string;count:number}[]}>('/api/folders')
  return r.folders
}
export async function getGallery(params: {
  folder?:string|null; type?:string|null; ext?:string|null; page?:number; page_size?:number; signal?:AbortSignal
}): Promise<GalleryResponse> {
  const { signal, ...rest } = params
  const url = new URL('/api/gallery', BASE)
  if (rest.folder) url.searchParams.set('folder', rest.folder)
  if (rest.type) url.searchParams.set('type', rest.type)
  if (rest.ext) url.searchParams.set('ext', rest.ext)
  url.searchParams.set('page', String(rest.page ?? 1))
  url.searchParams.set('page_size', String(rest.page_size ?? 48))
  const res = await fetch(url.toString(), { signal })
  if (!res.ok) { const e = await res.json().catch(()=>({})); throw new Error(e.detail??`${res.status}`) }
  return res.json()
}
export const getStats       = () => apiFetch<any>('/api/stats')
export const selectDirectory= async () => {
  const r = await apiFetch<{directory:string|null}>('/api/select-directory',{method:'POST'})
  return r.directory
}

// ── Files ─────────────────────────────────────────────────────────────────────
export const openFile = (media_id:number, show_in_folder=false) =>
  apiFetch('/api/open-file', {method:'POST', body:JSON.stringify({media_id,show_in_folder})})

// ── Training ──────────────────────────────────────────────────────────────────
export const startTraining  = (mode:string,epochs:number,lr:number,batch_size:number) =>
  apiFetch('/api/training/start',{method:'POST',body:JSON.stringify({mode,epochs,lr,batch_size})})
export const getTrainingStatus = () => apiFetch<TrainingStatus>('/api/training/status')
export const getFusionStatus   = () => apiFetch<FusionStatus>('/api/training/fusion-status')
export const buildFaissIndex   = () => apiFetch('/api/training/build-faiss',{method:'POST'})
export const deleteModel       = () => apiFetch('/api/training/model',{method:'DELETE'})
export const createTrainingStream = () => new EventSource(`${BASE}/api/training/stream`)

// Fix #7: training artifacts
export interface TrainingArtifact {
  session_id: string; mode: string; epochs: number; lr: number
  batch_size: number; started_at: string|null; finished_at: string|null
  result: Record<string,any>; status: string
  metrics?: { epoch:number; loss:number; val_acc:number }[]
  baselines?: Record<string, number>
  process?: { stage:string; detail:string; [key:string]:any }[]
  learning_curve_url?: string
  terminal_log_url?: string
  summary?: Record<string, any>
}
export const getArtifacts = () => apiFetch<{artifacts: TrainingArtifact[]}>('/api/training/artifacts')
export const assetUrl = (path: string) => `${BASE}${path}`

// ── Search History ──────────────────────────────────────────────────────────
export interface SearchHistoryItem {
  id: number; query: string; folder: string|null; date_from: string|null; date_to: string|null
  extensions: string|null; created_at: string
}
export const getSearchHistory = (limit = 3) =>
  apiFetch<{items: SearchHistoryItem[]}>(`/api/search/history?limit=${limit}`)
export const clearSearchHistory = () => apiFetch('/api/search/history', {method:'DELETE'})
export const deleteSearchHistoryItem = (itemId: number) =>
  apiFetch(`/api/search/history/${itemId}`, {method:'DELETE'})
