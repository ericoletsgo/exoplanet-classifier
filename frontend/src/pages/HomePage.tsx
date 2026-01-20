import { Target, BarChart3, Database, Upload, Brain } from 'lucide-react'
import { useNavigate } from 'react-router-dom'

export default function HomePage() {
  const navigate = useNavigate()

  const features = [
    {
      icon: Target,
      title: 'Make Predictions',
      description: 'Classify exoplanet candidates using our trained ML model',
      path: '/predict',
      color: 'text-blue-400',
    },
    {
      icon: Upload,
      title: 'Batch Upload',
      description: 'Upload CSV files to classify multiple candidates at once',
      path: '/batch',
      color: 'text-cyan-400',
    },
    {
      icon: BarChart3,
      title: 'View Metrics',
      description: 'Explore model performance, accuracy, and feature importance',
      path: '/retrain',
      color: 'text-green-400',
    },
    {
      icon: Brain,
      title: 'Model Retraining',
      description: 'Train new models and manage existing ones',
      path: '/retrain',
      color: 'text-purple-400',
    },
    {
      icon: Database,
      title: 'Browse Datasets',
      description: 'Explore KOI, K2, and TOI exoplanet datasets',
      path: '/datasets',
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
        {features.map((feature, index) => {
          const Icon = feature.icon
          return (
            <button
              key={`${feature.path}-${index}`}
              onClick={() => navigate(feature.path)}
              className="card hover:border-primary-500 transition-all duration-200 hover:scale-105 text-left w-full"
            >
              <Icon className={`w-12 h-12 ${feature.color} mb-4`} />
              <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
              <p className="text-slate-400">{feature.description}</p>
            </button>
          )
        })}
      </div>

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
