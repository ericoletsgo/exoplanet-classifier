import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Target, BarChart3, Database, TrendingUp, Upload, Brain } from 'lucide-react'
import { api } from '../lib/api'

export default function HomePage() {
  const [modelInfo, setModelInfo] = useState<any>(null)

  useEffect(() => {
    loadModelInfo()
  }, [])

  const loadModelInfo = async () => {
    try {
      const metrics = await api.getMetrics()
      setModelInfo(metrics.model_info)
    } catch (error) {
      console.error('Failed to load model info:', error)
      // Set default model info if loading fails
      setModelInfo({
        n_features: 19,
        n_samples: 0,
        classes: ['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED']
      })
    }
  }

  const features = [
    {
      icon: Target,
      title: 'Make Predictions',
      description: 'Classify exoplanet candidates using our trained ML model',
      link: '/predict',
      color: 'text-blue-400',
    },
    {
      icon: Upload,
      title: 'Batch Upload',
      description: 'Upload CSV files to classify multiple candidates at once',
      link: '/batch',
      color: 'text-cyan-400',
    },
    {
      icon: BarChart3,
      title: 'View Metrics',
      description: 'Explore model performance, accuracy, and feature importance',
      link: '/metrics',
      color: 'text-green-400',
    },
    {
      icon: Brain,
      title: 'Model Retraining',
      description: 'Train new models and manage existing ones',
      link: '/retrain',
      color: 'text-purple-400',
    },
    {
      icon: Database,
      title: 'Browse Datasets',
      description: 'Explore KOI, K2, and TOI exoplanet datasets',
      link: '/datasets',
      color: 'text-indigo-400',
    },
  ]

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold text-white">Exoplanet Classifier</h1>
        <p className="text-xl text-slate-400 max-w-2xl mx-auto">
          Machine learning powered classification system for identifying exoplanet candidates
          from Kepler mission data
        </p>
      </div>

      {/* Feature Cards */}
      <div className="grid md:grid-cols-2 lg:grid-cols-5 gap-6">
        {features.map((feature) => {
          const Icon = feature.icon
          return (
            <Link
              key={feature.link}
              to={feature.link}
              className="card hover:border-primary-500 transition-all duration-200 hover:scale-105"
            >
              <Icon className={`w-12 h-12 ${feature.color} mb-4`} />
              <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
              <p className="text-slate-400">{feature.description}</p>
            </Link>
          )
        })}
      </div>

      {/* Quick Stats */}
      {modelInfo && (
        <div className="card">
          <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-primary-500" />
            Model Information
          </h3>
          <div className="grid md:grid-cols-3 gap-4">
            <div>
              <p className="text-sm text-slate-400">Features</p>
              <p className="text-2xl font-bold text-white">{modelInfo.n_features}</p>
            </div>
            <div>
              <p className="text-sm text-slate-400">Training Samples</p>
              <p className="text-2xl font-bold text-white">
                {modelInfo.n_samples?.toLocaleString() || 'N/A'}
              </p>
            </div>
            <div>
              <p className="text-sm text-slate-400">Classes</p>
              <p className="text-2xl font-bold text-white">
                {modelInfo.classes?.length || 3}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* About */}
      <div className="card bg-slate-800/50">
        <h3 className="text-xl font-semibold mb-3">About This System</h3>
        <p className="text-slate-300 leading-relaxed">
          This application uses an ensemble of machine learning models (Gradient Boosting,
          Random Forest, XGBoost, and LightGBM) to classify exoplanet candidates from NASA's
          Kepler mission. The system analyzes stellar parameters, orbital characteristics, and
          signal quality metrics to determine whether a candidate is a confirmed exoplanet,
          a candidate requiring further study, or a false positive.
        </p>
      </div>
    </div>
  )
}
