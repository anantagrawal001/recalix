import { useState, useCallback, useRef } from 'react'
import { Search, Sparkles, X } from 'lucide-react'
import { cn } from '@/lib/utils'

interface Props {
  onSearch: (q: string) => void
  onClear:  () => void
  isSearching: boolean
  disabled?: boolean
}

const EXAMPLES = [
  'sunset on beach', 'people smiling', 'birthday party',
  'cats and dogs', 'city at night', 'mountain landscape',
  'receipt or invoice', 'whatsapp screenshot',
]

export function SearchBar({ onSearch, onClear, isSearching, disabled }: Props) {
  const [query,   setQuery]   = useState('')
  const [focused, setFocused] = useState(false)
  const debRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const trigger = useCallback((q: string) => {
    q.trim() ? onSearch(q.trim()) : onClear()
  }, [onSearch, onClear])

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const q = e.target.value
    setQuery(q)
    if (debRef.current) clearTimeout(debRef.current)
    debRef.current = setTimeout(() => trigger(q), 450)
  }, [trigger])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (debRef.current) clearTimeout(debRef.current)
    trigger(query)
  }

  const handleClear = () => { setQuery(''); onClear() }

  const handleExample = (ex: string) => {
    setQuery(ex)
    if (debRef.current) clearTimeout(debRef.current)
    onSearch(ex)
  }

  return (
    <div className="w-full max-w-3xl mx-auto">
      <form onSubmit={handleSubmit}>
        <div className={cn(
          'relative flex items-center rounded-2xl transition-all duration-300',
          'glass border border-border/60',
          focused && 'border-accent/50 shadow-glow',
        )}>
          {/* Left icon */}
          <div className="absolute left-5 text-dim pointer-events-none">
            {isSearching
              ? <div className="w-5 h-5 border-2 border-accent border-t-transparent rounded-full animate-spin" />
              : <Search className="w-5 h-5" />}
          </div>

          <input
            type="text"
            value={query}
            onChange={handleChange}
            onFocus={() => setFocused(true)}
            onBlur={() => setFocused(false)}
            disabled={disabled || isSearching}
            placeholder="Search your photos… 'sunset on beach', 'birthday party', 'invoice'"
            className={cn(
              'w-full py-4 pl-14 pr-16 text-base bg-transparent outline-none',
              'text-bright placeholder:text-ghost',
              'disabled:opacity-50 disabled:cursor-not-allowed',
            )}
          />

          {query && (
            <button type="button" onClick={handleClear}
              className="absolute right-14 p-1.5 rounded-full text-ghost hover:text-soft hover:bg-lift transition-colors">
              <X className="w-4 h-4" />
            </button>
          )}

          <button type="submit"
            disabled={!query.trim() || isSearching || disabled}
            className={cn(
              'absolute right-3 p-2.5 rounded-xl transition-all duration-150',
              'bg-accent-grad text-white shadow-glow-sm',
              'hover:opacity-90 active:scale-95',
              'disabled:opacity-30 disabled:cursor-not-allowed disabled:active:scale-100',
            )}>
            <Sparkles className="w-4 h-4" />
          </button>
        </div>
      </form>

      {/* Example chips */}
      <div className="mt-3 flex flex-wrap gap-2 justify-center">
        <span className="text-xs text-ghost self-center">Try:</span>
        {EXAMPLES.map(ex => (
          <button key={ex} onClick={() => handleExample(ex)}
            disabled={disabled || isSearching}
            className="px-3 py-1 text-xs rounded-full bg-lift border border-border/50 text-soft
                       hover:border-accent/40 hover:text-bright hover:bg-accent/10
                       disabled:opacity-40 transition-all duration-150">
            {ex}
          </button>
        ))}
      </div>
    </div>
  )
}
