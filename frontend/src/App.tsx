import { useState } from 'react'
import { Routes, Route, Link, useLocation } from 'react-router-dom'
import { Search, BrainCircuit, Github } from 'lucide-react'
import SearchPage   from './pages/SearchPage'
import TrainingPage from './pages/TrainingPage'
import { cn } from './lib/utils'

export default function App() {
  const loc = useLocation()

  return (
    <div className="min-h-screen flex flex-col">
      {/* ── Top nav ──────────────────────────────────────────────────────── */}
      <header className="sticky top-0 z-50 glass-heavy border-b border-border/50">
        <div className="max-w-screen-2xl mx-auto px-6 h-14 flex items-center gap-8">
          {/* Logo */}
          <div className="flex items-center gap-2.5 shrink-0">
            <div className="w-7 h-7 rounded-lg bg-accent-grad flex items-center justify-center shadow-glow-sm">
              <Search className="w-3.5 h-3.5 text-white" />
            </div>
            <span className="font-display font-semibold text-bright tracking-tight">
              RECALIX <span className="text-accent text-xs font-mono font-normal ml-1">v9</span>
            </span>
          </div>

          {/* Nav tabs */}
          <nav className="flex items-center gap-1">
            {[
              { to: '/',         label: 'Search',   Icon: Search },
              { to: '/training', label: 'Training', Icon: BrainCircuit },
            ].map(({ to, label, Icon }) => (
              <Link
                key={to}
                to={to}
                className={cn(
                  'flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-sm font-medium transition-all duration-150',
                  loc.pathname === to
                    ? 'bg-accent/15 text-accent border border-accent/20'
                    : 'text-dim hover:text-soft hover:bg-lift'
                )}
              >
                <Icon className="w-4 h-4" />
                {label}
              </Link>
            ))}
          </nav>

          {/* Right side */}
          <div className="ml-auto flex items-center gap-3">
            <span className="text-xs text-ghost font-mono hidden md:block">
              Local AI Photo Search
            </span>
          </div>
        </div>
      </header>

      {/* ── Page content ─────────────────────────────────────────────────── */}
      <main className="flex-1">
        <Routes>
          <Route path="/"         element={<SearchPage />} />
          <Route path="/training" element={<TrainingPage />} />
        </Routes>
      </main>
    </div>
  )
}
