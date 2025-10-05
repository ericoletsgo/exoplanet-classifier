import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatNumber(num: number, decimals: number = 2): string {
  return num.toFixed(decimals)
}

export function formatPercentage(num: number, decimals: number = 1): string {
  return `${(num * 100).toFixed(decimals)}%`
}

export function getDispositionColor(disposition: string): string {
  switch (disposition.toUpperCase()) {
    case 'CONFIRMED':
      return 'text-green-400 bg-green-900/30 border-green-700'
    case 'CANDIDATE':
      return 'text-yellow-400 bg-yellow-900/30 border-yellow-700'
    case 'FALSE POSITIVE':
      return 'text-red-400 bg-red-900/30 border-red-700'
    default:
      return 'text-slate-400 bg-slate-900/30 border-slate-700'
  }
}

export function getConfidenceColor(confidence: number): string {
  if (confidence >= 0.8) return 'text-green-400'
  if (confidence >= 0.6) return 'text-yellow-400'
  return 'text-red-400'
}
