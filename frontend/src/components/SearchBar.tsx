import { useCallback, useRef } from 'react'
import { Search, X, Sparkles } from 'lucide-react'
import { cn } from '@/lib/utils'

const EXAMPLES = [
  'sunset on beach', 'people smiling', 'birthday party',
  'cats', 'city at night', 'mountain', 'receipt or invoice', 'whatsapp screenshot',
]

interface Props {
  onSearch:    (q: string) => void
  onClear:     () => void
  isSearching: boolean
  disabled?:   boolean
  value:       string
  onChange:    (q: string) => void
  showExamples?: boolean
  history?: { id: number; query: string; created_at: string }[]
  onHistorySelect?: (q: string) => void
  onClearHistory?: () => void
  onDeleteHistoryItem?: (id: number) => void
}

export function SearchBar({
  onSearch,
  onClear,
  isSearching,
  disabled,
  value,
  onChange,
  showExamples = true,
  history = [],
  onHistorySelect,
  onClearHistory,
  onDeleteHistoryItem,
}: Props) {
  // Fix #5: input is NEVER disabled while searching.
  // The parent holds the query string so the input doesn't re-mount on re-render.
  const debRef   = useRef<ReturnType<typeof setTimeout> | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const [focused, setFocused] = [useRef(false), { current: false }]

  const trigger = useCallback((q: string) => {
    q.trim() ? onSearch(q.trim()) : onClear()
  }, [onSearch, onClear])

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const q = e.target.value
    onChange(q)
    if (debRef.current) clearTimeout(debRef.current)
    debRef.current = setTimeout(() => trigger(q), 500)
  }, [onChange, trigger])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (debRef.current) clearTimeout(debRef.current)
    trigger(value)
  }

  const handleClear = () => {
    onChange('')
    onClear()
    inputRef.current?.focus()   // keep focus after clearing
  }

  const handleExample = (ex: string) => {
    onChange(ex)
    if (debRef.current) clearTimeout(debRef.current)
    onSearch(ex)
    inputRef.current?.focus()
  }

  const handleHistoryPick = (q: string) => {
    onChange(q)
    if (debRef.current) clearTimeout(debRef.current)
    onSearch(q)
    inputRef.current?.focus()
  }

  return (
    <div className="w-full space-y-3">
      <form onSubmit={handleSubmit}>
        <div className="relative flex items-center rounded-2xl transition-all duration-200 glass border border-border/60 focus-within:border-accent/50 focus-within:shadow-glow">
          {/* Search icon / spinner */}
          <div className="absolute left-4 text-dim pointer-events-none">
            {isSearching
              ? <div className="w-5 h-5 border-2 border-accent border-t-transparent rounded-full animate-spin" />
              : <Search className="w-5 h-5" />}
          </div>

          {/* Input — never disabled while searching so cursor stays */}
          <input
            ref={inputRef}
            type="text"
            value={value}
            onChange={handleChange}
            disabled={disabled}   // only disabled during indexing, NOT during searching
            placeholder="Search your photos… 'sunset', 'birthday party', 'invoice'"
            className="w-full py-4 pl-12 pr-16 bg-transparent outline-none text-base
                       text-bright placeholder:text-ghost disabled:opacity-50"
            autoComplete="off"
          />

          {/* Clear button */}
          {value && (
            <button type="button" onClick={handleClear}
              className="absolute right-12 p-1.5 rounded-full text-ghost hover:text-soft hover:bg-lift transition-colors">
              <X className="w-4 h-4" />
            </button>
          )}

          {/* Submit button */}
          <button type="submit" disabled={!value.trim() || !!disabled}
            className="absolute right-3 p-2.5 rounded-xl bg-gradient-to-br from-accent to-purple-500
                       text-white shadow-glow-sm hover:opacity-90 active:scale-95
                       disabled:opacity-30 disabled:active:scale-100 transition-all">
            <Sparkles className="w-4 h-4" />
          </button>
        </div>
      </form>

      {history.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs text-ghost">Recent</span>
            {onClearHistory && (
              <button
                type="button"
                onClick={onClearHistory}
                className="text-xs text-dim hover:text-soft transition-colors">
                Clear history
              </button>
            )}
          </div>
          <div className="flex flex-wrap gap-2">
            {history.map(item => (
              <div key={item.id} className="relative group">
                <button
                  type="button"
                  onClick={() => (onHistorySelect ? onHistorySelect(item.query) : handleHistoryPick(item.query))}
                  className="px-3 py-1 text-xs rounded-full bg-surface border border-border/50 text-soft
                             hover:border-accent/40 hover:text-bright hover:bg-accent/10 transition-all duration-150">
                  {item.query}
                </button>
                {onDeleteHistoryItem && (
                  <button
                    type="button"
                    onClick={() => onDeleteHistoryItem(item.id)}
                    className="absolute -top-1.5 -right-1.5 p-0.5 rounded-full bg-void/70 border border-border/40
                               text-ghost hover:text-soft hover:border-accent/40 transition-colors">
                    <X className="w-3 h-3" />
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Example chips */}
      {showExamples && <div className="flex flex-wrap gap-2 justify-center">
        <span className="text-xs text-ghost self-center">Try:</span>
        {EXAMPLES.map(ex => (
          <button key={ex} onClick={() => handleExample(ex)}
            disabled={!!disabled}
            className="px-3 py-1 text-xs rounded-full bg-lift border border-border/50 text-soft
                       hover:border-accent/40 hover:text-bright hover:bg-accent/10
                       disabled:opacity-40 transition-all duration-150">
            {ex}
          </button>
        ))}
      </div>}
    </div>
  )
}
