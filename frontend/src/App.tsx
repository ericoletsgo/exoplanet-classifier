import { Home, Target, Database, Upload, Brain } from 'lucide-react'
import { Suspense, lazy, useEffect } from 'react'
import { Routes, Route, NavLink } from 'react-router-dom'
import HomePage from './pages/HomePage'
import LoadingScreen from './components/LoadingScreen'
import { api } from './lib/api'

// Lazy load heavy components to improve initial load time
const PredictPage = lazy(() => import('./pages/PredictPage'))
const DatasetsPage = lazy(() => import('./pages/DatasetsPage'))
const BatchPredictPage = lazy(() => import('./pages/BatchPredictPage'))
const ModelRetrainingPage = lazy(() => import('./pages/ModelRetrainingPage'))

function Navigation() {
  const navItems = [
    { path: '/', icon: Home, label: 'Home' },
    { path: '/predict', icon: Target, label: 'Predict' },
    { path: '/batch', icon: Upload, label: 'Batch Upload' },
    { path: '/retrain', icon: Brain, label: 'Model Retraining' },
    { path: '/datasets', icon: Database, label: 'Datasets' },
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
                return (
                  <NavLink
                    key={item.path}
                    to={item.path}
                    end={item.path === '/'}
                    className={({ isActive }) =>
                      `flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                        isActive
                          ? 'bg-primary-600 text-white'
                          : 'text-slate-300 hover:bg-slate-700 hover:text-white'
                      }`
                    }
                  >
                    <Icon className="w-4 h-4" />
                    {item.label}
                  </NavLink>
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
  // Fire a lightweight warm-up call immediately on mount to trigger the
  // serverless cold-start and model pre-load while the user is still
  // reading the page. This is fire-and-forget; errors are silently ignored.
  useEffect(() => {
    api.warmUp()
  }, [])

  return (
    <div className="min-h-screen bg-slate-900 text-white">
      <Navigation />
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route
            path="/predict"
            element={
              <Suspense fallback={
                <LoadingScreen
                  message="Loading Prediction Interface"
                  subMessage="Preparing ML models and features..."
                  type="prediction"
                />
              }>
                <PredictPage />
              </Suspense>
            }
          />
          <Route
            path="/batch"
            element={
              <Suspense fallback={
                <LoadingScreen
                  message="Loading Batch Upload"
                  subMessage="Preparing CSV processing tools..."
                  type="dataset"
                />
              }>
                <BatchPredictPage />
              </Suspense>
            }
          />
          <Route
            path="/retrain"
            element={
              <Suspense fallback={
                <LoadingScreen
                  message="Loading Model Training"
                  subMessage="Initializing training algorithms..."
                  type="training"
                />
              }>
                <ModelRetrainingPage />
              </Suspense>
            }
          />
          <Route
            path="/datasets"
            element={
              <Suspense fallback={
                <LoadingScreen
                  message="Loading Datasets"
                  subMessage="Connecting to database..."
                  type="dataset"
                />
              }>
                <DatasetsPage />
              </Suspense>
            }
          />
          {/* Fallback route */}
          <Route path="*" element={<HomePage />} />
        </Routes>
      </main>
    </div>
  )
}

export default App
