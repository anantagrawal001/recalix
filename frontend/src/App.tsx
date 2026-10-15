import { useState, useEffect } from 'react'
import { Routes, Route, Link, useLocation } from 'react-router-dom'
import { Search, BrainCircuit, Database, Sun, Moon } from 'lucide-react'
import SearchPage   from './pages/SearchPage'
import TrainingPage from './pages/TrainingPage'
import IndexPage from './pages/IndexPage'
import { cn } from './lib/utils'

// Fix #4: dark/light mode toggle - persisted in localStorage
function useTheme() {
  const [dark, setDark] = useState(() => {
    try { return localStorage.getItem('theme') !== 'light' }
    catch { return true }
  })

  useEffect(() => {
    document.documentElement.classList.toggle('light-mode', !dark)
    try { localStorage.setItem('theme', dark ? 'dark' : 'light') } catch {}
  }, [dark])

  return [dark, setDark] as const
}

export default function App() {
  const loc         = useLocation()
  const [dark, setDark] = useTheme()

  const tabs = [
    { to: '/',         label: 'Search',   Icon: Search },
    { to: '/index',    label: 'Index',    Icon: Database },
    { to: '/training', label: 'Training', Icon: BrainCircuit },
  ]

  return (
    <div className="min-h-screen flex flex-col">
      <header className="sticky top-0 z-50 glass-heavy border-b border-border/40">
        <div className="max-w-screen-2xl mx-auto px-5 h-14 flex items-center gap-6">
          {/* Logo */}
          <div className="flex items-center gap-2 shrink-0">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-accent to-purple-500 flex items-center justify-center shadow-glow-sm">
              <Search className="w-3.5 h-3.5 text-white" />
            </div>
            <span className="font-display font-semibold text-bright tracking-tight">
              RECALIX <span className="text-accent font-mono text-xs font-normal ml-0.5">v10</span>
            </span>
          </div>

          {/* Nav */}
          <nav className="flex items-center gap-1">
            {tabs.map(({ to, label, Icon }) => (
              <Link key={to} to={to} className={cn(
                'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all duration-150',
                loc.pathname === to
                  ? 'bg-accent/15 text-accent border border-accent/20'
                  : 'text-dim hover:text-soft hover:bg-lift'
              )}>
                <Icon className="w-3.5 h-3.5" />{label}
              </Link>
            ))}
          </nav>

          {/* Right side */}
          <div className="ml-auto flex items-center gap-3">
            <span className="text-xs text-ghost font-mono hidden md:block">Local AI Photo Search</span>
            {/* Fix #4: dark/light toggle */}
            <button
              onClick={() => setDark(d => !d)}
              title={dark ? 'Switch to light mode' : 'Switch to dark mode'}
              className="p-2 rounded-lg glass border border-border/40 text-dim hover:text-soft hover:border-accent/30 transition-all">
              {dark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>
          </div>
        </div>
      </header>

      <main className="flex-1">
        <Routes>
          <Route path="/"         element={<SearchPage />} />
          <Route path="/index"    element={<IndexPage />} />
          <Route path="/training" element={<TrainingPage />} />
        </Routes>
      </main>
    </div>
  )
}
