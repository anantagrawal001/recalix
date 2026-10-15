import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'
export const cn = (...i: ClassValue[]) => twMerge(clsx(i))
export const fmtDate = (s: string|null) => s ? new Date(s).toLocaleDateString('en-US',{year:'numeric',month:'short',day:'numeric'}) : '—'
export const fmtSize = (b: number|null) => !b ? '' : b < 1048576 ? `${(b/1024).toFixed(1)} KB` : `${(b/1048576).toFixed(1)} MB`
