import { useState, useEffect } from 'react'
import { Target, Loader2, AlertCircle } from 'lucide-react'
import { api, type PredictionResponse, type FeaturesResponse } from '../lib/api'
import { formatPercentage, getDispositionColor, getConfidenceColor } from '../lib/utils'

export default function PredictPage() {
  const [features, setFeatures] = useState<FeaturesResponse | null>(null)
  const [formData, setFormData] = useState<Record<string, number>>({})
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeCategory, setActiveCategory] = useState<string>('signal_quality')

  useEffect(() => {
    loadFeatures()
  }, [])

  const loadFeatures = async () => {
    try {
      const data = await api.getFeatures()
      setFeatures(data)
      
      // Initialize form with zeros
      const initialData: Record<string, number> = {}
      Object.values(data.features).flat().forEach(feature => {
        initialData[feature] = 0
      })
      setFormData(initialData)
    } catch (err) {
      setError('Failed to load features')
    }
  }

  const handleInputChange = (feature: string, value: string) => {
    setFormData(prev => ({
      ...prev,
      [feature]: parseFloat(value) || 0
    }))
  }

  const handlePredict = async () => {
    setLoading(true)
    setError(null)
    setPrediction(null)

    try {
      const result = await api.predict({ features: formData })
      setPrediction(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  const handleLoadExample = async () => {
    try {
      // Load a random example from the dataset
      const randomPage = Math.floor(Math.random() * 100) + 1 // Random page between 1-100
      const dataset = await api.getDataset('koi', randomPage, 1)
      if (dataset.data.length > 0) {
        const example = dataset.data[0]
        const newFormData: Record<string, number> = {}
        
        Object.keys(formData).forEach(feature => {
          newFormData[feature] = example[feature] !== null && example[feature] !== undefined 
            ? parseFloat(example[feature]) 
            : 0
        })
        
        setFormData(newFormData)
        setPrediction(null) // Clear previous prediction
      }
    } catch (err) {
      setError('Failed to load example')
    }
  }

  const handleReset = () => {
    if (features) {
      const resetData: Record<string, number> = {}
      Object.values(features.features).flat().forEach(feature => {
        resetData[feature] = 0
      })
      setFormData(resetData)
      setPrediction(null)
      setError(null)
    }
  }

  if (!features) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-primary-500" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <Target className="w-8 h-8 text-primary-500" />
            Make Prediction
          </h1>
          <p className="text-slate-400 mt-2">
            Enter feature values to classify an exoplanet candidate
          </p>
        </div>
        <div className="flex gap-3">
          <button onClick={handleLoadExample} className="btn-secondary">
            Load Example
          </button>
          <button onClick={handleReset} className="btn-secondary">
            Reset
          </button>
        </div>
      </div>

      {error && (
        <div className="card bg-red-900/20 border-red-700">
          <div className="flex items-center gap-2 text-red-400">
            <AlertCircle className="w-5 h-5" />
            <p>{error}</p>
          </div>
        </div>
      )}

      {/* Prediction Result */}
      {prediction && (
        <div className="card bg-gradient-to-br from-primary-900/30 to-slate-800 border-primary-700">
          <h3 className="text-xl font-semibold mb-4">Prediction Result</h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <p className="text-sm text-slate-400 mb-2">Classification</p>
              <div className={`inline-block px-4 py-2 rounded-lg border ${getDispositionColor(prediction.prediction)}`}>
                <p className="text-2xl font-bold">{prediction.prediction}</p>
              </div>
            </div>
            <div>
              <p className="text-sm text-slate-400 mb-2">Confidence</p>
              <p className={`text-3xl font-bold ${getConfidenceColor(prediction.confidence)}`}>
                {formatPercentage(prediction.confidence)}
              </p>
            </div>
          </div>
          
          <div className="mt-6">
            <p className="text-sm text-slate-400 mb-3">Class Probabilities</p>
            <div className="space-y-2">
              {Object.entries(prediction.probabilities).map(([label, prob]) => (
                <div key={label}>
                  <div className="flex justify-between text-sm mb-1">
                    <span>{label}</span>
                    <span className="font-semibold">{formatPercentage(prob)}</span>
                  </div>
                  <div className="w-full bg-slate-700 rounded-full h-2">
                    <div
                      className="bg-primary-500 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${prob * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Feature Input Form */}
      <div className="card">
        <div className="flex gap-2 mb-6 border-b border-slate-700 overflow-x-auto">
          {Object.keys(features.features).map((category) => (
            <button
              key={category}
              onClick={() => setActiveCategory(category)}
              className={`px-4 py-2 font-medium transition-colors whitespace-nowrap ${
                activeCategory === category
                  ? 'text-primary-400 border-b-2 border-primary-400'
                  : 'text-slate-400 hover:text-slate-300'
              }`}
            >
              {category.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
            </button>
          ))}
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4 max-h-96 overflow-y-auto pr-2">
          {features.features[activeCategory].map((feature) => (
            <div key={feature} className="flex flex-col">
              <label 
                className="block text-sm font-medium text-slate-300 mb-1 cursor-help" 
                title={features.descriptions?.[feature] || ''}
              >
                {features.labels?.[feature] || feature}
              </label>
              <div className="flex-grow">
                <input
                  type="number"
                  step="any"
                  value={formData[feature] || 0}
                  onChange={(e) => handleInputChange(feature, e.target.value)}
                  className="input-field w-full"
                  placeholder="0.0"
                />
              </div>
              {features.descriptions?.[feature] && (
                <p className="text-xs text-slate-500 mt-1 min-h-[2.5rem]">
                  {features.descriptions[feature]}
                </p>
              )}
            </div>
          ))}
        </div>

        <div className="mt-6 pt-6 border-t border-slate-700">
          <button
            onClick={handlePredict}
            disabled={loading}
            className="btn-primary w-full flex items-center justify-center gap-2"
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Predicting...
              </>
            ) : (
              <>
                <Target className="w-5 h-5" />
                Predict
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  )
}
