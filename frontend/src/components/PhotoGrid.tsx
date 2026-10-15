import { useState, useCallback } from 'react'
import { Image, Video, Calendar, Folder, ExternalLink, FolderOpen, Cpu, Zap } from 'lucide-react'
import { cn, fmtDate, fmtSize } from '@/lib/utils'
import { type MediaResult, openFile } from '@/lib/api'

export function PhotoGrid({
  photos,
  isLoading,
  variant = 'search',
}: {
  photos: MediaResult[]
  isLoading?: boolean
  variant?: 'search' | 'gallery'
}) {
  if (isLoading) return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6 gap-4">
      {Array.from({length:18}).map((_,i) => (
        <div key={i} className="aspect-square rounded-2xl shimmer bg-card"
          style={{animationDelay:`${i*0.04}s`}} />
      ))}
    </div>
  )
  if (!photos.length) return (
    <div className="flex flex-col items-center justify-center py-24 text-center animate-fade-in">
      <div className="w-20 h-20 rounded-3xl glass flex items-center justify-center mb-4 border border-border/40">
        <Image className="w-9 h-9 text-ghost" />
      </div>
      <h3 className="text-lg font-display font-semibold text-soft mb-1">No results found</h3>
      <p className="text-sm text-ghost max-w-xs">
        {variant === 'gallery' ? 'This folder has no indexed media for the current filter' : 'Try a different query or adjust your filters'}
      </p>
    </div>
  )
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6 gap-4 animate-fade-in">
      {photos.map((p, i) => (
        <div key={p.id} style={{animationDelay:`${Math.min(i,20)*0.025}s`}} className="animate-slide-up">
          <PhotoCard photo={p} variant={variant} />
        </div>
      ))}
    </div>
  )
}

function PhotoCard({ photo, variant }: { photo: MediaResult; variant: 'search' | 'gallery' }) {
  const [hovered,   setHovered]   = useState(false)
  const [imgError,  setImgError]  = useState(false)
  const [showInfo,  setShowInfo]  = useState(false)

  const thumb = photo.thumbnail_url ? `http://localhost:8000${photo.thumbnail_url}` : null
  const scorePct = Math.round(photo.score * 100)
  const isFaiss  = photo._search_backend === 'faiss'

  const handleOpen   = useCallback(async () => { try { await openFile(photo.id,false) } catch(e){alert(e)} }, [photo.id])
  const handleFolder = useCallback(async () => { try { await openFile(photo.id,true)  } catch(e){alert(e)} }, [photo.id])

  return (
    <div
      className={cn(
        'group relative rounded-2xl overflow-hidden cursor-default',
        'glass border border-border/40 transition-all duration-300',
        hovered && 'border-accent/30 shadow-glow scale-[1.02] -translate-y-0.5',
      )}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => { setHovered(false); setShowInfo(false) }}
    >
      {/* Thumbnail */}
      <div className="aspect-square relative overflow-hidden bg-surface">
        {thumb && !imgError
          ? <img src={thumb} alt={photo.filename} loading="lazy"
              className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110"
              onError={() => setImgError(true)} />
          : <div className="w-full h-full flex items-center justify-center">
              {photo.file_type==='video' ? <Video className="w-10 h-10 text-ghost"/> : <Image className="w-10 h-10 text-ghost"/>}
            </div>
        }

        {/* Top badges */}
        <div className="absolute top-2 left-2 flex gap-1.5">
          <div className="glass px-1.5 py-0.5 rounded-md border border-white/5">
            {photo.file_type==='video' ? <Video className="w-3 h-3 text-accent"/> : <Image className="w-3 h-3 text-dim"/>}
          </div>
          {isFaiss && <div className="glass px-1.5 py-0.5 rounded-md border border-accent/20" title="FAISS ANN search">
            <Zap className="w-3 h-3 text-accent"/>
          </div>}
        </div>

        {/* Score */}
        {variant === 'search' && (
          <div className="absolute top-2 right-2">
            <div className={cn('glass px-2 py-0.5 rounded-md text-xs font-mono border border-white/5',
              scorePct>=40 ? 'text-success' : scorePct>=28 ? 'text-warning' : 'text-dim')}>
              {scorePct}%
            </div>
          </div>
        )}

        {/* Hover overlay */}
        <div className={cn(
          'absolute inset-0 flex flex-col justify-end',
          'bg-gradient-to-t from-void/90 via-void/40 to-transparent',
          'transition-opacity duration-200',
          hovered ? 'opacity-100' : 'opacity-0',
        )}>
          <div className="p-3 space-y-2">
            <p className="text-xs font-medium text-white truncate">{photo.filename}</p>
            <div className="flex items-center gap-1.5 text-xs text-white/60">
              <Calendar className="w-3 h-3"/><span>{fmtDate(photo.modified_at)}</span>
            </div>
            <div className="flex items-center gap-1.5 text-xs text-white/60">
              <Folder className="w-3 h-3 shrink-0"/>
              <span className="truncate">{(photo.folder||'').split(/[/\\]/).pop()||photo.folder}</span>
            </div>

            {/* Signal breakdown */}
            {photo._signals && showInfo && (
              <div className="grid grid-cols-2 gap-1 text-xs font-mono">
                {Object.entries(photo._signals).map(([k,v]) => (
                  <div key={k} className="flex justify-between bg-void/70 rounded px-1.5 py-0.5">
                    <span className="text-white/40">{k[0].toUpperCase()}</span>
                    <span className="text-white/80">{Math.round((v as number)*100)}%</span>
                  </div>
                ))}
              </div>
            )}

            <div className="flex gap-1.5">
              <button onClick={handleOpen}
                className="flex-1 py-1.5 rounded-lg glass border border-white/10 text-xs font-medium
                           text-white flex items-center justify-center gap-1 hover:bg-white/10 transition-colors">
                <ExternalLink className="w-3 h-3"/>Open
              </button>
              <button onClick={handleFolder}
                className="flex-1 py-1.5 rounded-lg glass border border-white/10 text-xs font-medium
                           text-white flex items-center justify-center gap-1 hover:bg-white/10 transition-colors">
                <FolderOpen className="w-3 h-3"/>Folder
              </button>
              {photo._signals && (
                <button onClick={() => setShowInfo(s=>!s)} title="Score breakdown"
                  className="p-1.5 rounded-lg glass border border-white/10 text-white hover:bg-white/10 transition-colors">
                  <Cpu className="w-3 h-3"/>
                </button>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Caption */}
      <div className="px-3 py-2.5">
        <p className="text-xs font-medium text-soft truncate">{photo.filename}</p>
        <p className="text-xs text-ghost mt-0.5 truncate">
          {fmtSize(photo.file_size)}{photo.width&&photo.height ? ` · ${photo.width}×${photo.height}`:''}</p>
      </div>
    </div>
  )
}
