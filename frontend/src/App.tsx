import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom'
import { Home, Target, Database, Upload, Brain } from 'lucide-react'
import HomePage from './pages/HomePage'
import PredictPage from './pages/PredictPage'
import DatasetsPage from './pages/DatasetsPage'
import BatchPredictPage from './pages/BatchPredictPage'
import ModelRetrainingPage from './pages/ModelRetrainingPage'

function Navigation() {
  const location = useLocation()
  
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
                const isActive = location.pathname === item.path
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={`flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                      isActive
                        ? 'bg-primary-600 text-white'
                        : 'text-slate-300 hover:bg-slate-700 hover:text-white'
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    {item.label}
                  </Link>
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
  return (
    <Router>
      <div className="min-h-screen bg-slate-900 text-white">
        <Navigation />
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/predict" element={<PredictPage />} />
            <Route path="/batch" element={<BatchPredictPage />} />
            <Route path="/retrain" element={<ModelRetrainingPage />} />
            <Route path="/datasets" element={<DatasetsPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}

export default App
