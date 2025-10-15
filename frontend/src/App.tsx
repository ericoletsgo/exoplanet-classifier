import { Home, Target, Database, Upload, Brain } from 'lucide-react'
import { useState, Suspense, lazy } from 'react'
import { Loader2 } from 'lucide-react'
import HomePage from './pages/HomePage'

// Lazy load heavy components to improve initial load time
const PredictPage = lazy(() => import('./pages/PredictPage'))
const DatasetsPage = lazy(() => import('./pages/DatasetsPage'))
const BatchPredictPage = lazy(() => import('./pages/BatchPredictPage'))
const ModelRetrainingPage = lazy(() => import('./pages/ModelRetrainingPage'))

interface NavigationProps {
  activeTab: string
  onTabChange: (tab: string) => void
}

function Navigation({ activeTab, onTabChange }: NavigationProps) {
  const navItems = [
    { id: 'home', icon: Home, label: 'Home' },
    { id: 'predict', icon: Target, label: 'Predict' },
    { id: 'batch', icon: Upload, label: 'Batch Upload' },
    { id: 'retrain', icon: Brain, label: 'Model Retraining' },
    { id: 'datasets', icon: Database, label: 'Datasets' },
  ]
  
  return (
    <nav className="bg-slate-900 border-b border-slate-700">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <h1 className="text-xl font-bold text-white flex items-center gap-2">
                <Target className="w-6 h-6 text-primary-500" />
                Exoplanet Classifier
              </h1>
            </div>
            <div className="ml-10 flex items-baseline space-x-4">
              {navItems.map((item) => {
                const Icon = item.icon
                const isActive = activeTab === item.id
                return (
                  <button
                    key={item.id}
                    onClick={() => onTabChange(item.id)}
                    className={`flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                      isActive
                        ? 'bg-primary-600 text-white'
                        : 'text-slate-300 hover:bg-slate-700 hover:text-white'
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    {item.label}
                  </button>
                )
              })}
            </div>
          </div>
        </div>
      </div>
    </nav>
  )
}

function App() {
  const [activeTab, setActiveTab] = useState('home')

  const renderActiveComponent = () => {
    switch (activeTab) {
      case 'home':
        return <HomePage onNavigate={setActiveTab} />
      case 'predict':
        return (
          <Suspense fallback={
            <div className="flex items-center justify-center h-64">
              <Loader2 className="w-8 h-8 animate-spin text-primary-500" />
            </div>
          }>
            <PredictPage />
          </Suspense>
        )
      case 'batch':
        return (
          <Suspense fallback={
            <div className="flex items-center justify-center h-64">
              <Loader2 className="w-8 h-8 animate-spin text-primary-500" />
            </div>
          }>
            <BatchPredictPage />
          </Suspense>
        )
      case 'retrain':
        return (
          <Suspense fallback={
            <div className="flex items-center justify-center h-64">
              <Loader2 className="w-8 h-8 animate-spin text-primary-500" />
            </div>
          }>
            <ModelRetrainingPage />
          </Suspense>
        )
      case 'datasets':
        return (
          <Suspense fallback={
            <div className="flex items-center justify-center h-64">
              <Loader2 className="w-8 h-8 animate-spin text-primary-500" />
            </div>
          }>
            <DatasetsPage />
          </Suspense>
        )
      default:
        return <HomePage onNavigate={setActiveTab} />
    }
  }

  return (
    <div className="min-h-screen bg-slate-900 text-white">
      <Navigation activeTab={activeTab} onTabChange={setActiveTab} />
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {renderActiveComponent()}
      </main>
    </div>
  )
}

export default App
