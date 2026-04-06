import { TrainingPanel } from '@/components/TrainingPanel'
import { BrainCircuit, BookOpen } from 'lucide-react'

export default function TrainingPage() {
  return (
    <div className="max-w-3xl mx-auto p-6 lg:p-8 space-y-6">
      {/* Header */}
      <div className="space-y-1">
        <h1 className="font-display text-2xl font-semibold text-bright flex items-center gap-3">
          <BrainCircuit className="w-6 h-6 text-accent" />
          Model Training
        </h1>
        <p className="text-sm text-ghost">
          Train the Query-Conditioned Fusion model to improve search relevance.
          The model learns query-adaptive weights across 4 signals: visual, OCR, filename, structural.
        </p>
      </div>

      {/* Research context box */}
      <div className="glass border border-accent/20 rounded-2xl p-4 space-y-2">
        <div className="flex items-center gap-2 text-sm font-medium text-accent">
          <BookOpen className="w-4 h-4" />
          Research Context
        </div>
        <p className="text-xs text-ghost leading-relaxed">
          The fusion model is a 148K-parameter MLP trained on your own indexed images.
          It predicts per-query weights — shifting toward visual signals for aesthetic queries,
          and toward OCR for document/receipt queries. This adaptive behavior is the core
          research finding: structural metadata is often <em className="text-soft">more discriminative</em> than
          raw visual similarity for personal photo collections.
          Use <strong className="text-soft">Baseline</strong> mode to generate the comparison table for your paper.
        </p>
      </div>

      <TrainingPanel />
    </div>
  )
}
