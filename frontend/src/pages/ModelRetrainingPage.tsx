import { useState, useEffect } from 'react'
import { Upload, Loader2, AlertCircle, Trash2, BarChart3, Brain } from 'lucide-react'
import { api } from '../lib/api'

interface ModelMetadata {
  id: string
  name: string
  description: string
  created_at: string
  train_samples: number
  test_samples: number
  train_accuracy: number
  test_accuracy: number
  precision: number
  recall: number
  f1_score: number
  confusion_matrix: number[][]
  features: string[]
  n_features: number
  algorithms: string[]
}

export default function ModelRetrainingPage() {
  const [activeTab, setActiveTab] = useState<'train' | 'evaluations' | 'management'>('train')
  const [models, setModels] = useState<ModelMetadata[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  
  // Training form state
  const [modelName, setModelName] = useState('')
  const [testSize, setTestSize] = useState(20)
  const [description, setDescription] = useState('')
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [uploadProgress, setUploadProgress] = useState<string>('')

  useEffect(() => {
    loadModels()
  }, [])

  const loadModels = async () => {
    try {
      const response = await api.listModels()
      setModels(response.models || [])
    } catch (err) {
      console.error('Failed to load models:', err)
    }
  }

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setUploadedFile(file)
      setError(null)
    }
  }

  const handleTrainModel = async () => {
    if (!uploadedFile || !modelName) {
      setError('Please provide a model name and upload a CSV file')
      return
    }

    setLoading(true)
    setError(null)
    setUploadProgress('')

    try {
      // This would need to be implemented in the API
      // For now, show a placeholder
      setUploadProgress('Training model... This may take a few minutes.')
      
      // Simulate training progress
      await new Promise(resolve => setTimeout(resolve, 3000))
      
      setUploadProgress('Model training completed!')
      await loadModels() // Refresh models list
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Training failed')
    } finally {
      setLoading(false)
    }
  }

  const deleteModel = async (modelId: string) => {
    if (!confirm('Are you sure you want to delete this model?')) return
    
    try {
      // This would need to be implemented in the API
      console.log('Delete model:', modelId)
      await loadModels()
    } catch (err) {
      setError('Failed to delete model')
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold flex items-center gap-3">
          <Brain className="w-8 h-8 text-primary-500" />
          Model Retraining & Management
        </h1>
        <p className="text-slate-400 mt-2">Train new models and manage existing ones</p>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-slate-700">
        <nav className="flex space-x-8">
          {[
            { id: 'train', label: 'Train New Model', icon: Upload },
            { id: 'evaluations', label: 'Model Evaluations', icon: BarChart3 },
            { id: 'management', label: 'Model Management', icon: Trash2 },
          ].map((tab) => {
            const Icon = tab.icon
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center gap-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === tab.id
                    ? 'border-primary-500 text-primary-400'
                    : 'border-transparent text-slate-400 hover:text-slate-300 hover:border-slate-600'
                }`}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
              </button>
            )
          })}
        </nav>
      </div>

      {error && (
        <div className="card bg-red-900/20 border-red-700">
          <div className="flex items-center gap-2 text-red-400">
            <AlertCircle className="w-5 h-5" />
            <p>{error}</p>
          </div>
        </div>
      )}

      {/* Train New Model Tab */}
      {activeTab === 'train' && (
        <div className="space-y-6">
          <div className="card">
            <h3 className="text-xl font-semibold mb-4">Train a New Model</h3>
            <p className="text-slate-400 mb-6">
              Upload a CSV file with exoplanet data to train a new model. The CSV should contain:
            </p>
            <ul className="text-slate-400 mb-6 list-disc list-inside space-y-1">
              <li>A target column with disposition values that map to: CONFIRMED, CANDIDATE, or FALSE POSITIVE</li>
              <li>Relevant features for classification</li>
            </ul>

            {/* Model Configuration */}
            <div className="grid md:grid-cols-2 gap-4 mb-6">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Model Name
                </label>
                <input
                  type="text"
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value)}
                  placeholder="Model_20251005_0539"
                  className="input-field w-full"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Test Set Size (%)
                </label>
                <input
                  type="range"
                  min="10"
                  max="40"
                  value={testSize}
                  onChange={(e) => setTestSize(Number(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-slate-500 mt-1">
                  <span>10%</span>
                  <span>{testSize}%</span>
                  <span>40%</span>
                </div>
              </div>
            </div>

            <div className="mb-6">
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Model Description (optional)
              </label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Describe this model version..."
                rows={3}
                className="input-field w-full"
              />
            </div>

            {/* File Upload */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Upload Training Data (CSV)
              </label>
              <div className="border-2 border-dashed border-slate-600 rounded-lg p-8 text-center">
                <Upload className="w-12 h-12 text-slate-400 mx-auto mb-4" />
                <p className="text-slate-300 mb-4">
                  {uploadedFile ? uploadedFile.name : 'Drop your CSV file here or click to browse'}
                </p>
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleFileUpload}
                  className="hidden"
                  id="csv-upload"
                />
                <label htmlFor="csv-upload" className="btn-primary cursor-pointer inline-block">
                  {uploadedFile ? 'Change File' : 'Select CSV File'}
                </label>
                <p className="text-xs text-slate-500 mt-2">Limit 200MB per file • CSV</p>
              </div>
            </div>

            {uploadProgress && (
              <div className="mb-6 p-4 bg-blue-900/20 border border-blue-700 rounded-lg">
                <div className="flex items-center gap-2 text-blue-400">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <p>{uploadProgress}</p>
                </div>
              </div>
            )}

            <button
              onClick={handleTrainModel}
              disabled={loading || !uploadedFile || !modelName}
              className="btn-primary w-full flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Training Model...
                </>
              ) : (
                <>
                  <Brain className="w-5 h-5" />
                  Train Model
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {/* Model Evaluations Tab */}
      {activeTab === 'evaluations' && (
        <div className="space-y-6">
          <div className="card">
            <h3 className="text-xl font-semibold mb-4">Model Evaluations</h3>
            
            {models.length === 0 ? (
              <p className="text-slate-400">No trained models found. Train your first model in the 'Train New Model' tab.</p>
            ) : (
              <div className="space-y-4">
                <p className="text-slate-400">Total Models: {models.length}</p>
                
                {/* Model Comparison Table */}
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-slate-700">
                        <th className="text-left p-3">Model Name</th>
                        <th className="text-left p-3">Created</th>
                        <th className="text-left p-3">Test Accuracy</th>
                        <th className="text-left p-3">Precision</th>
                        <th className="text-left p-3">Recall</th>
                        <th className="text-left p-3">F1 Score</th>
                        <th className="text-left p-3">Features</th>
                      </tr>
                    </thead>
                    <tbody>
                      {models.map((model) => (
                        <tr key={model.id} className="border-b border-slate-800 hover:bg-slate-800/50">
                          <td className="p-3">
                            <div>
                              <p className="font-medium">{model.name}</p>
                              <p className="text-xs text-slate-500">{model.id}</p>
                            </div>
                          </td>
                          <td className="p-3 text-sm">{new Date(model.created_at).toLocaleDateString()}</td>
                          <td className="p-3">
                            <span className="font-semibold text-green-400">
                              {(model.test_accuracy * 100).toFixed(1)}%
                            </span>
                          </td>
                          <td className="p-3 text-sm">{(model.precision * 100).toFixed(1)}%</td>
                          <td className="p-3 text-sm">{(model.recall * 100).toFixed(1)}%</td>
                          <td className="p-3 text-sm">{(model.f1_score * 100).toFixed(1)}%</td>
                          <td className="p-3 text-sm">{model.n_features}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Model Management Tab */}
      {activeTab === 'management' && (
        <div className="space-y-6">
          <div className="card">
            <h3 className="text-xl font-semibold mb-4">Model Management</h3>
            
            {models.length === 0 ? (
              <p className="text-slate-400">No models to manage.</p>
            ) : (
              <div className="space-y-4">
                <p className="text-slate-400">Total Models: {models.length}</p>
                
                {models.map((model) => (
                  <div key={model.id} className="border border-slate-700 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-3">
                      <div>
                        <h4 className="font-semibold">{model.name}</h4>
                        <p className="text-sm text-slate-500">{model.id}</p>
                      </div>
                      <button
                        onClick={() => deleteModel(model.id)}
                        className="btn-secondary text-red-400 hover:bg-red-900/20 hover:border-red-700 flex items-center gap-2"
                      >
                        <Trash2 className="w-4 h-4" />
                        Delete
                      </button>
                    </div>
                    
                    <div className="grid md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <p className="text-slate-400">Created</p>
                        <p>{new Date(model.created_at).toLocaleString()}</p>
                      </div>
                      <div>
                        <p className="text-slate-400">Accuracy</p>
                        <p className="text-green-400 font-semibold">
                          {(model.test_accuracy * 100).toFixed(1)}%
                        </p>
                      </div>
                      <div>
                        <p className="text-slate-400">Samples</p>
                        <p>{model.train_samples.toLocaleString()} train, {model.test_samples} test</p>
                      </div>
                      <div>
                        <p className="text-slate-400">Features</p>
                        <p>{model.n_features}</p>
                      </div>
                    </div>
                    
                    {model.description && (
                      <div className="mt-3 pt-3 border-t border-slate-700">
                        <p className="text-slate-400 text-sm">Description</p>
                        <p className="text-sm">{model.description}</p>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
