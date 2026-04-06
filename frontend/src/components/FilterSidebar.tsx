import { useState } from 'react'
import { Filter, Image, Video, Folder, ChevronDown, ChevronRight, X } from 'lucide-react'
import { cn } from '@/lib/utils'

export interface FilterState {
  fileTypes: ('image' | 'video')[]
  folders:   string[]
  dateRange: { min: number; max: number } | null
}

interface Props {
  folders:         string[]
  filters:         FilterState
  onFiltersChange: (f: FilterState) => void
  resultCount:     number
  isOpen:          boolean
  onClose:         () => void
}

export function FilterSidebar({ folders, filters, onFiltersChange, resultCount, isOpen, onClose }: Props) {
  const [open, setOpen] = useState({ type: true, folders: true })
  const toggle = (k: 'type' | 'folders') => setOpen(p => ({ ...p, [k]: !p[k] }))

  const toggleType = (t: 'image' | 'video') => {
    const types = filters.fileTypes.includes(t)
      ? filters.fileTypes.filter(x => x !== t)
      : [...filters.fileTypes, t]
    onFiltersChange({ ...filters, fileTypes: types })
  }

  const toggleFolder = (f: string) => {
    const fs = filters.folders.includes(f)
      ? filters.folders.filter(x => x !== f)
      : [...filters.folders, f]
    onFiltersChange({ ...filters, folders: fs })
  }

  const hasActive = filters.fileTypes.length !== 2 || filters.folders.length > 0

  const Section = ({ id, label, children }: { id: 'type'|'folders'; label: string; children: React.ReactNode }) => (
    <div className="space-y-2">
      <button onClick={() => toggle(id)}
        className="flex items-center justify-between w-full text-left text-sm font-medium text-soft hover:text-bright transition-colors">
        {label}
        {open[id] ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
      </button>
      {open[id] && <div className="space-y-1">{children}</div>}
    </div>
  )

  return (
    <>
      {isOpen && <div className="fixed inset-0 bg-void/60 z-40 lg:hidden backdrop-blur-sm" onClick={onClose} />}

      <aside className={cn(
        'fixed lg:sticky top-14 left-0 z-50 lg:z-auto',
        'w-72 h-[calc(100vh-3.5rem)]',
        'glass-heavy border-r border-border/40',
        'flex flex-col',
        'transition-transform duration-300 ease-out',
        isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0',
      )}>
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-border/30">
          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-accent" />
            <span className="font-medium text-sm text-bright">Filters</span>
          </div>
          <div className="flex items-center gap-2">
            {hasActive && (
              <button onClick={() => onFiltersChange({ fileTypes: ['image','video'], folders: [], dateRange: null })}
                className="text-xs text-dim hover:text-soft transition-colors">
                Clear all
              </button>
            )}
            <button onClick={onClose} className="lg:hidden p-1 rounded hover:bg-lift text-ghost">
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Result count pill */}
        <div className="px-4 py-2.5 border-b border-border/20">
          <span className="text-xs text-ghost">
            <span className="font-semibold text-soft">{resultCount.toLocaleString()}</span> results
          </span>
        </div>

        {/* Filters */}
        <div className="flex-1 overflow-y-auto p-4 space-y-5">
          <Section id="type" label="File Type">
            {(['image', 'video'] as const).map(t => (
              <label key={t} className="flex items-center gap-3 px-2 py-1.5 rounded-lg cursor-pointer
                                         hover:bg-lift transition-colors group">
                <div className={cn(
                  'w-4 h-4 rounded border flex items-center justify-center transition-all',
                  filters.fileTypes.includes(t) ? 'bg-accent border-accent' : 'border-border group-hover:border-accent/40'
                )}>
                  {filters.fileTypes.includes(t) && (
                    <svg className="w-2.5 h-2.5 text-white" fill="none" viewBox="0 0 12 12">
                      <path d="M2 6l3 3 5-5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  )}
                </div>
                {t === 'image' ? <Image className="w-3.5 h-3.5 text-dim" /> : <Video className="w-3.5 h-3.5 text-dim" />}
                <span className="text-sm text-soft capitalize">{t}s</span>
              </label>
            ))}
          </Section>

          {folders.length > 0 && (
            <Section id="folders" label="Folders">
              <div className="max-h-52 overflow-y-auto space-y-0.5">
                {folders.map(f => (
                  <label key={f} className="flex items-center gap-3 px-2 py-1.5 rounded-lg cursor-pointer
                                             hover:bg-lift transition-colors group">
                    <div className={cn(
                      'w-4 h-4 rounded border flex items-center justify-center flex-shrink-0 transition-all',
                      filters.folders.includes(f) ? 'bg-accent border-accent' : 'border-border group-hover:border-accent/40'
                    )}>
                      {filters.folders.includes(f) && (
                        <svg className="w-2.5 h-2.5 text-white" fill="none" viewBox="0 0 12 12">
                          <path d="M2 6l3 3 5-5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                      )}
                    </div>
                    <Folder className="w-3.5 h-3.5 text-dim flex-shrink-0" />
                    <span className="text-xs text-soft truncate" title={f}>
                      {f === '/' ? 'Root' : f.split(/[/\\]/).pop() || f}
                    </span>
                  </label>
                ))}
              </div>
            </Section>
          )}
        </div>
      </aside>
    </>
  )
}
