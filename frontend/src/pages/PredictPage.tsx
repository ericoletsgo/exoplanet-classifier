import { useState, useEffect } from 'react'
import { Target, Loader2, AlertCircle, Dice1, Dice2, Dice3, CheckCircle, XCircle } from 'lucide-react'
import { api, type PredictionResponse, type FeaturesResponse } from '../lib/api'
import { formatPercentage, getDispositionColor, getConfidenceColor } from '../lib/utils'

interface RandomExampleData {
  features: Record<string, number>
  metadata: {
    row_index: number
    koi_name: string
    expected_disposition: string
    dataset: string
  }
  raw_row: Record<string, any>
}

export default function PredictPage() {
  const [features, setFeatures] = useState<FeaturesResponse | null>(null)
  const [formData, setFormData] = useState<Record<string, number>>({})
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeCategory, setActiveCategory] = useState<string>('signal_quality')
  const [randomExampleData, setRandomExampleData] = useState<RandomExampleData | null>(null)
  const [models, setModels] = useState<any[]>([])
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null)
  // const [modelInfo, setModelInfo] = useState<any>(null) // Not used to prevent slow loading

  useEffect(() => {
    loadFeatures()
    loadModels()
    // Model info loading is optional - will load when needed
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

  const loadModels = async () => {
    try {
      const response = await api.listModels()
      setModels(response.models || [])
    } catch (err) {
      console.error('Failed to load models:', err)
    }
  }

  // Optional model info loading - not used by default to prevent slow page loads
  // const loadModelInfo = async () => {
  //   try {
  //     const metrics = await api.getMetrics()
  //     setModelInfo(metrics.model_info)
  //   } catch (err) {
  //     console.error('Failed to load model info:', err)
  //     // Set default model info if loading fails
  //     setModelInfo({
  //       n_features: 19,
  //       n_samples: 0,
  //       classes: ['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED']
  //     })
  //   }
  // }

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
      let result
      
      // If we have random example data, use raw prediction for accuracy
      if (randomExampleData) {
        result = await api.predictRaw(randomExampleData.raw_row)
      } else {
        // For manual input, use regular prediction with selected model
        result = await api.predict({ 
          features: formData,
          model_id: selectedModelId || undefined
        })
      }
      
      setPrediction(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  const handleLoadRandomExample = async (disposition?: string) => {
    try {
      const exampleData = await api.getRandomExample('koi', disposition)
      setRandomExampleData(exampleData)
      setFormData(exampleData.features)
      setPrediction(null) // Clear previous prediction
      setError(null)
    } catch (err) {
      setError('Failed to load random example')
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
      setRandomExampleData(null)
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
          <button onClick={handleReset} className="btn-secondary">
            Reset
          </button>
        </div>
      </div>

      {/* Model Selection */}
      {models.length > 0 && (
        <div className="card">
          <h3 className="text-lg font-semibold mb-3">Select Model</h3>
          <div className="flex items-center gap-4">
            <select
              value={selectedModelId || ''}
              onChange={(e) => setSelectedModelId(e.target.value || null)}
              className="input-field flex-1"
            >
              <option value="">Default Model (Latest)</option>
              {models.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name} - {(model.test_accuracy * 100).toFixed(1)}% accuracy
                  {model.created_at && ` (${new Date(model.created_at).toLocaleDateString()})`}
                </option>
              ))}
            </select>
            {selectedModelId && (
              <button
                onClick={() => setSelectedModelId(null)}
                className="text-sm text-slate-400 hover:text-slate-300"
              >
                Clear
              </button>
            )}
          </div>
          <p className="text-xs text-slate-500 mt-2">
            {selectedModelId 
              ? `Using model: ${models.find(m => m.id === selectedModelId)?.name || 'Unknown'}`
              : 'Using the default model for predictions'}
          </p>
        </div>
      )}

      {/* Random Example Buttons */}
      <div className="card">
        <h3 className="text-lg font-semibold mb-4">Load Random Examples</h3>
        <div className="grid md:grid-cols-3 gap-3">
          <button
            onClick={() => handleLoadRandomExample('CONFIRMED')}
            className="btn-secondary flex items-center justify-center gap-2"
          >
            <Dice1 className="w-4 h-4" />
            Random Confirmed Planet
          </button>
          <button
            onClick={() => handleLoadRandomExample('CANDIDATE')}
            className="btn-secondary flex items-center justify-center gap-2"
          >
            <Dice2 className="w-4 h-4" />
            Random Candidate
          </button>
          <button
            onClick={() => handleLoadRandomExample('FALSE POSITIVE')}
            className="btn-secondary flex items-center justify-center gap-2"
          >
            <Dice3 className="w-4 h-4" />
            Random False Positive
          </button>
        </div>
        <p className="text-sm text-slate-500 mt-3">
          These buttons sample random examples from the KOI dataset.
        </p>
      </div>

      {/* Random Example Information */}
      {randomExampleData && (
        <div className="card bg-slate-800/50 border-slate-600">
          <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
            <CheckCircle className="w-5 h-5 text-green-400" />
            Random Example Loaded
          </h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-slate-400">Data Source</p>
              <p className="font-semibold">
                Row {randomExampleData.metadata.row_index} from {randomExampleData.metadata.dataset}.csv
              </p>
              <p className="text-sm text-slate-500">KOI: {randomExampleData.metadata.koi_name}</p>
            </div>
            <div>
              <p className="text-sm text-slate-400">Expected Result (NASA)</p>
              <div className={`inline-block px-3 py-1 rounded-lg border ${getDispositionColor(randomExampleData.metadata.expected_disposition)}`}>
                <p className="font-semibold">{randomExampleData.metadata.expected_disposition}</p>
              </div>
            </div>
          </div>
        </div>
      )}

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

          {/* Prediction vs Expected Comparison */}
          {randomExampleData && (
            <div className="mt-6 pt-6 border-t border-slate-600">
              <h4 className="text-lg font-semibold mb-4">📊 Prediction vs Expected</h4>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-slate-400 mb-2">Expected (NASA):</p>
                  <div className={`inline-block px-3 py-2 rounded-lg border ${getDispositionColor(randomExampleData.metadata.expected_disposition)}`}>
                    <p className="font-semibold">{randomExampleData.metadata.expected_disposition}</p>
                  </div>
                </div>
                <div>
                  <p className="text-sm text-slate-400 mb-2">Model Prediction:</p>
                  <div className={`inline-block px-3 py-2 rounded-lg border ${getDispositionColor(prediction.prediction)}`}>
                    <p className="font-semibold">{prediction.prediction}</p>
                  </div>
                </div>
              </div>
              
              {/* Match/Mismatch Result */}
              <div className="mt-4">
                {prediction.prediction === randomExampleData.metadata.expected_disposition ? (
                  <div className="flex items-center gap-2 text-green-400">
                    <CheckCircle className="w-5 h-5" />
                    <span className="font-semibold">🎉 CORRECT! Model prediction matches NASA's classification!</span>
                  </div>
                ) : (
                  <div className="flex items-center gap-2 text-red-400">
                    <XCircle className="w-5 h-5" />
                    <span className="font-semibold">
                      ❌ MISMATCH! Expected {randomExampleData.metadata.expected_disposition}, but model predicted {prediction.prediction}
                    </span>
                  </div>
                )}
              </div>
            </div>
          )}
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
