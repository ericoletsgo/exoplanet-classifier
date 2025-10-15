import { Loader2, Target, Brain, Database } from 'lucide-react'

interface LoadingScreenProps {
  message?: string
  subMessage?: string
  type?: 'default' | 'training' | 'prediction' | 'dataset'
  progress?: number // 0-100 for progress bar
  showProgress?: boolean
}

export default function LoadingScreen({ 
  message = "Loading...", 
  subMessage,
  type = 'default',
  progress = 0,
  showProgress = false
}: LoadingScreenProps) {
  const getIcon = () => {
    switch (type) {
      case 'training':
        return <Brain className="w-16 h-16 text-purple-400 animate-pulse" />
      case 'prediction':
        return <Target className="w-16 h-16 text-blue-400 animate-pulse" />
      case 'dataset':
        return <Database className="w-16 h-16 text-green-400 animate-pulse" />
      default:
        return <Loader2 className="w-16 h-16 text-primary-500 animate-spin" />
    }
  }

  const getAnimationClass = () => {
    switch (type) {
      case 'training':
        return 'animate-bounce'
      case 'prediction':
        return 'animate-pulse'
      case 'dataset':
        return 'animate-pulse'
      default:
        return 'animate-spin'
    }
  }

  return (
    <div className="fixed inset-0 bg-slate-900/95 backdrop-blur-sm z-50 flex items-center justify-center">
      <div className="bg-slate-800 rounded-2xl p-8 shadow-2xl border border-slate-700 max-w-md w-full mx-4">
        {/* Animated Icon */}
        <div className="flex justify-center mb-6">
          <div className={`${getAnimationClass()} transform transition-all duration-300`}>
            {getIcon()}
          </div>
        </div>

        {/* Main Message */}
        <div className="text-center mb-4">
          <h3 className="text-xl font-semibold text-white mb-2">
            {message}
          </h3>
          {subMessage && (
            <p className="text-slate-400 text-sm">
              {subMessage}
            </p>
          )}
        </div>

        {/* Progress Bar */}
        {showProgress && (
          <div className="mb-6">
            <div className="flex justify-between text-xs text-slate-400 mb-2">
              <span>Progress</span>
              <span>{Math.round(progress)}%</span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-2 overflow-hidden">
              <div 
                className="bg-gradient-to-r from-primary-500 to-primary-400 h-2 rounded-full transition-all duration-500 ease-out"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}

        {/* Animated Dots */}
        <div className="flex justify-center space-x-1">
          <div className="w-2 h-2 bg-primary-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
          <div className="w-2 h-2 bg-primary-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
          <div className="w-2 h-2 bg-primary-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
        </div>

        {/* Type-specific tips */}
        {type === 'training' && (
          <div className="mt-6 text-xs text-slate-500 text-center">
            <p>🤖 Training ensemble models...</p>
            <p className="mt-1">This may take 1-2 minutes depending on dataset size</p>
          </div>
        )}
        
        {type === 'prediction' && (
          <div className="mt-6 text-xs text-slate-500 text-center">
            <p>🔮 Analyzing features...</p>
            <p className="mt-1">Running classification algorithms</p>
          </div>
        )}

        {type === 'dataset' && (
          <div className="mt-6 text-xs text-slate-500 text-center">
            <p>📊 Loading dataset...</p>
            <p className="mt-1">Processing large CSV files</p>
          </div>
        )}
      </div>
    </div>
  )
}

// Inline loading component for smaller areas
export function LoadingSpinner({ size = 'md', message }: { size?: 'sm' | 'md' | 'lg', message?: string }) {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6', 
    lg: 'w-8 h-8'
  }

  return (
    <div className="flex items-center gap-2">
      <Loader2 className={`${sizeClasses[size]} animate-spin text-primary-500`} />
      {message && <span className="text-sm text-slate-400">{message}</span>}
    </div>
  )
}

// Full page loading component
export function FullPageLoading({ message = "Loading application..." }: { message?: string }) {
  return (
    <div className="min-h-screen bg-slate-900 flex items-center justify-center">
      <div className="text-center">
        <div className="mb-6">
          <Target className="w-20 h-20 text-primary-500 mx-auto animate-pulse" />
        </div>
        <h2 className="text-2xl font-bold text-white mb-2">Exoplanet Classifier</h2>
        <p className="text-slate-400 mb-6">{message}</p>
        <div className="flex justify-center space-x-1">
          <div className="w-3 h-3 bg-primary-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
          <div className="w-3 h-3 bg-primary-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
          <div className="w-3 h-3 bg-primary-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
        </div>
      </div>
    </div>
  )
}
