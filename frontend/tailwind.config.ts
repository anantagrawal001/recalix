import type { Config } from 'tailwindcss'
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans:  ['DM Sans', 'system-ui', 'sans-serif'],
        mono:  ['JetBrains Mono', 'Fira Code', 'monospace'],
        display: ['Outfit', 'DM Sans', 'sans-serif'],
      },
      colors: {
        void:   '#050508', ink: '#0a0a12', slate: '#0f0f1a',
        surface:'#14141f', card: '#1a1a28', lift: '#1f1f30',
        border: '#2a2a3d', muted: '#3d3d55', ghost: '#4a4a66',
        dim:    '#6b6b88', soft: '#9090aa', base: '#c8c8dc',
        bright: '#e8e8f0',
        accent: { DEFAULT:'#6c63ff', dim:'#4a43cc', glow:'#8b83ff',
                  muted:'#6c63ff22', soft:'#6c63ff44' },
        success:{ DEFAULT:'#22d47a', muted:'#22d47a22' },
        warning:{ DEFAULT:'#f5a623', muted:'#f5a62322' },
        error:  { DEFAULT:'#ff4d6a', muted:'#ff4d6a22' },
        info:   { DEFAULT:'#38bdf8', muted:'#38bdf822' },
      },
      boxShadow: {
        glass:    '0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05)',
        glow:     '0 0 20px rgba(108,99,255,0.25), 0 0 60px rgba(108,99,255,0.08)',
        'glow-sm':'0 0 10px rgba(108,99,255,0.3)',
        card:     '0 2px 16px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.04)',
      },
      animation: {
        'fade-in':  'fadeIn 0.3s ease-out',
        'slide-up': 'slideUp 0.4s cubic-bezier(0.16,1,0.3,1)',
        'shimmer':  'shimmer 1.8s infinite linear',
        'log-in':   'logIn 0.12s ease-out',
      },
      keyframes: {
        fadeIn:  { from:{opacity:'0'},           to:{opacity:'1'} },
        slideUp: { from:{opacity:'0',transform:'translateY(10px)'}, to:{opacity:'1',transform:'translateY(0)'} },
        shimmer: { '0%':{backgroundPosition:'-200% 0'}, '100%':{backgroundPosition:'200% 0'} },
        logIn:   { from:{opacity:'0',transform:'translateX(-4px)'}, to:{opacity:'1',transform:'translateX(0)'} },
      },
    },
  },
  plugins: [],
} satisfies Config
