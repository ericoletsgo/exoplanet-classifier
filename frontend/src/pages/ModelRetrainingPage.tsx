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
  const [availableAlgorithms, setAvailableAlgorithms] = useState<Record<string, boolean>>({})
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  
  // Training form state
  const [modelName, setModelName] = useState('')
  const [testSize, setTestSize] = useState(20)
  const [description, setDescription] = useState('')
  const [selectedDataset, setSelectedDataset] = useState('koi.csv')
  const [selectedAlgorithms, setSelectedAlgorithms] = useState<string[]>(['gradient_boosting', 'random_forest', 'xgboost', 'lightgbm'])
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [uploadProgress, setUploadProgress] = useState<string>('')
  
  // Hyperparameter states
  const [useHyperparameterTuning, setUseHyperparameterTuning] = useState(false)
  const [hyperparameters, setHyperparameters] = useState({
    // Gradient Boosting
    gb_n_estimators: 100,
    gb_learning_rate: 0.1,
    gb_max_depth: 3,
    gb_min_samples_split: 2,
    // Random Forest
    rf_n_estimators: 100,
    rf_max_depth: 10,
    rf_min_samples_split: 2,
    rf_max_features: 'sqrt',
    // XGBoost
    xgb_n_estimators: 100,
    xgb_learning_rate: 0.05,
    xgb_max_depth: 6,
    xgb_subsample: 1.0,
    // LightGBM
    lgb_n_estimators: 100,
    lgb_learning_rate: 0.05,
    lgb_max_depth: -1,
    lgb_num_leaves: 31
  })

  useEffect(() => {
    loadModels()
    loadAvailableAlgorithms()
  }, [])

  const loadModels = async () => {
    try {
      const response = await api.listModels()
      setModels(response.models || [])
    } catch (err) {
      console.error('Failed to load models:', err)
    }
  }

  const loadAvailableAlgorithms = async () => {
    try {
      const response = await api.getAvailableAlgorithms()
      setAvailableAlgorithms(response.algorithms)
    } catch (err) {
      console.error('Failed to load algorithms:', err)
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
    if (!modelName) {
      setError('Please provide a model name')
      return
    }

    if (selectedAlgorithms.length === 0) {
      setError('Please select at least one algorithm')
      return
    }

    setLoading(true)
    setError(null)
    setUploadProgress('')

    try {
      setUploadProgress('Training advanced ensemble... This may take a few minutes.')
      
      // Call the enhanced training API
      const result = await api.trainModel({
        dataset: selectedDataset,
        model_name: modelName,
        description: description,
        test_size: testSize / 100, // Convert percentage to decimal
        algorithms: selectedAlgorithms,
        hyperparameters: hyperparameters,
        use_hyperparameter_tuning: useHyperparameterTuning
      })
      
      if (result.cv_accuracy && result.algorithms_used) {
        setUploadProgress(`🎯 Ensemble trained successfully! CV Accuracy: ${(result.cv_accuracy * 100).toFixed(1)}%, Test Accuracy: ${(result.metrics?.accuracy * 100).toFixed(1)}%, Used ${result.algorithms_used.length} algorithms`)
      } else {
        setUploadProgress(`Model training completed! Accuracy: ${(result.metrics?.accuracy * 100).toFixed(1)}%`)
      }
      
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

            {/* Dataset Selection */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Dataset
              </label>
              <select
                value={selectedDataset}
                onChange={(e) => setSelectedDataset(e.target.value)}
                className="input-field w-full"
              >
                <option value="koi.csv">KOI Dataset (Kepler Objects of Interest)</option>
                <option value="k2.csv">K2 Dataset</option>
                <option value="toi.csv">TOI Dataset (TESS Objects of Interest)</option>
              </select>
            </div>
            
            {/* Algorithm Selection */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Algorithms (Select algorithms to include in ensemble)
              </label>
              <div className="grid grid-cols-2 gap-2">
                {Object.entries(availableAlgorithms).map(([algo, available]) => (
                  <label key={algo} className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={selectedAlgorithms.includes(algo)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedAlgorithms([...selectedAlgorithms, algo])
                        } else {
                          setSelectedAlgorithms(selectedAlgorithms.filter(a => a !== algo))
                        }
                      }}
                      disabled={!available}
                      className="rounded border-slate-600 bg-slate-700 text-primary-500 focus:ring-primary-500"
                    />
                    <span className={`text-sm ${available ? 'text-slate-300' : 'text-slate-500'}`}>
                      {algo === 'gradient_boosting' ? 'Gradient Boosting' :
                       algo === 'random_forest' ? 'Random Forest' :
                       algo === 'xgboost' ? 'XGBoost' :
                       algo === 'lightgbm' ? 'LightGBM' : algo}
                      {available ? ' ✓' : ' ⚠ Not Available'}
                    </span>
                  </label>
                ))}
              </div>
              <p className="text-xs text-slate-400 mt-2">
                {selectedAlgorithms.length} algorithm(s) selected
              </p>
            </div>

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

            {/* Hyperparameter Tuning Toggle */}
            <div className="mb-6">
              <div className="flex items-center space-x-3">
                <input
                  type="checkbox"
                  id="hyperparameter-tuning"
                  checked={useHyperparameterTuning}
                  onChange={(e) => setUseHyperparameterTuning(e.target.checked)}
                  className="rounded border-slate-600 bg-slate-700 text-primary-500 focus:ring-primary-500"
                />
                <label htmlFor="hyperparameter-tuning" className="text-sm font-medium text-slate-300">
                  Enable Hyperparameter Tuning (Grid Search)
                </label>
              </div>
              <p className="text-xs text-slate-500 mt-1">
                Automatically optimize algorithm parameters using grid search. Takes longer but may improve accuracy.
              </p>
            </div>

            {/* Hyperparameter Controls */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-slate-300 mb-4">
                Algorithm Hyperparameters
              </label>
              
              {/* Gradient Boosting Parameters */}
              {selectedAlgorithms.includes('gradient_boosting') && (
                <div className="mb-6 p-4 bg-slate-800/50 rounded-lg border border-slate-700">
                  <h4 className="text-sm font-semibold text-green-400 mb-3">Gradient Boosting</h4>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs text-slate-400 mb-1">N Estimators</label>
                      <input
                        type="number"
                        min="50"
                        max="500"
                        value={hyperparameters.gb_n_estimators}
                        onChange={(e) => setHyperparameters(prev => ({ ...prev, gb_n_estimators: Number(e.target.value) }))}
                        className="input-field w-full text-sm"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-slate-400 mb-1">Learning Rate</label>
                      <input
                        type="number"
                        step="0.01"
                        min="0.01"
                        max="0.3"
                        value={hyperparameters.gb_learning_rate}
                        onChange={(e) => setHyperparameters(prev => ({ ...prev, gb_learning_rate: Number(e.target.value) }))}
                        className="input-field w-full text-sm"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-slate-400 mb-1">Max Depth</label>
                      <input
                        type="number"
                        min="1"
                        max="10"
                        value={hyperparameters.gb_max_depth}
                        onChange={(e) => setHyperparameters(prev => ({ ...prev, gb_max_depth: Number(e.target.value) }))}
                        className="input-field w-full text-sm"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-slate-400 mb-1">Min Samples Split</label>
                      <input
                        type="number"
                        min="2"
                        max="20"
                        value={hyperparameters.gb_min_samples_split}
                        onChange={(e) => setHyperparameters(prev => ({ ...prev, gb_min_samples_split: Number(e.target.value) }))}
                        className="input-field w-full text-sm"
                      />
                    </div>
                  </div>
                </div>
              )}

              {/* Random Forest Parameters */}
              {selectedAlgorithms.includes('random_forest') && (
                <div className="mb-6 p-4 bg-slate-800/50 rounded-lg border border-slate-700">
                  <h4 className="text-sm font-semibold text-blue-400 mb-3">Random Forest</h4>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs text-slate-400 mb-1">N Estimators</label>
                      <input
                        type="number"
                        min="50"
                        max="500"
                        value={hyperparameters.rf_n_estimators}
                        onChange={(e) => setHyperparameters(prev => ({ ...prev, rf_n_estimators: Number(e.target.value) }))}
                        className="input-field w-full text-sm"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-slate-400 mb-1">Max Depth</label>
                      <input
                        type="number"
                        min="5"
                        max="20"
                        value={hyperparameters.rf_max_depth}
                        onChange={(e) => setHyperparameters(prev => ({ ...prev, rf_max_depth: Number(e.target.value) }))}
                        className="input-field w-full text-sm"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-slate-400 mb-1">Min Samples Split</label>
                      <input
                        type="number"
                        min="2"
                        max="20"
                        value={hyperparameters.rf_min_samples_split}
                        onChange={(e) => setHyperparameters(prev => ({ ...prev, rf_min_samples_split: Number(e.target.value) }))}
                        className="input-field w-full text-sm"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-slate-400 mb-1">Max Features</label>
                      <select
                        value={hyperparameters.rf_max_features}
                        onChange={(e) => setHyperparameters(prev => ({ ...prev, rf_max_features: e.target.value }))}
                        className="input-field w-full text-sm"
                      >
                        <option value="sqrt">sqrt</option>
                        <option value="log2">log2</option>
                        <option value="auto">auto</option>
                      </select>
                    </div>
                  </div>
                </div>
              )}

              {/* XGBoost Parameters */}
              {selectedAlgorithms.includes('xgboost') && (
                <div className="mb-6 p-4 bg-slate-800/50 rounded-lg border border-slate-700">
                  <h4 className="text-sm font-semibold text-purple-400 mb-3">XGBoost</h4>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs text-slate-400 mb-1">N Estimators</label>
                      <input
                        type="number"
                        min="50"
                        max="500"
                        value={hyperparameters.xgb_n_estimators}
                        onChange={(e) => setHyperparameters(prev => ({ ...prev, xgb_n_estimators: Number(e.target.value) }))}
                        className="input-field w-full text-sm"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-slate-400 mb-1">Learning Rate</label>
                      <input
                        type="number"
                        step="0.01"
                        min="0.01"
                        max="0.3"
                        value={hyperparameters.xgb_learning_rate}
                        onChange={(e) => setHyperparameters(prev => ({ ...prev, xgb_learning_rate: Number(e.target.value) }))}
                        className="input-field w-full text-sm"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-slate-400 mb-1">Max Depth</label>
                      <input
                        type="number"
                        min="3"
                        max="10"
                        value={hyperparameters.xgb_max_depth}
                        onChange={(e) => setHyperparameters(prev => ({ ...prev, xgb_max_depth: Number(e.target.value) }))}
                        className="input-field w-full text-sm"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-slate-400 mb-1">Subsample</label>
                      <input
                        type="number"
                        step="0.1"
                        min="0.6"
                        max="1.0"
                        value={hyperparameters.xgb_subsample}
                        onChange={(e) => setHyperparameters(prev => ({ ...prev, xgb_subsample: Number(e.target.value) }))}
                        className="input-field w-full text-sm"
                      />
                    </div>
                  </div>
                </div>
              )}

              {/* LightGBM Parameters */}
              {selectedAlgorithms.includes('lightgbm') && (
                <div className="mb-6 p-4 bg-slate-800/50 rounded-lg border border-slate-700">
                  <h4 className="text-sm font-semibold text-yellow-400 mb-3">LightGBM</h4>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs text-slate-400 mb-1">N Estimators</label>
                      <input
                        type="number"
                        min="50"
                        max="500"
                        value={hyperparameters.lgb_n_estimators}
                        onChange={(e) => setHyperparameters(prev => ({ ...prev, lgb_n_estimators: Number(e.target.value) }))}
                        className="input-field w-full text-sm"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-slate-400 mb-1">Learning Rate</label>
                      <input
                        type="number"
                        step="0.01"
                        min="0.01"
                        max="0.3"
                        value={hyperparameters.lgb_learning_rate}
                        onChange={(e) => setHyperparameters(prev => ({ ...prev, lgb_learning_rate: Number(e.target.value) }))}
                        className="input-field w-full text-sm"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-slate-400 mb-1">Max Depth</label>
                      <input
                        type="number"
                        min="-1"
                        max="10"
                        value={hyperparameters.lgb_max_depth}
                        onChange={(e) => setHyperparameters(prev => ({ ...prev, lgb_max_depth: Number(e.target.value) }))}
                        className="input-field w-full text-sm"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-slate-400 mb-1">Num Leaves</label>
                      <input
                        type="number"
                        min="10"
                        max="100"
                        value={hyperparameters.lgb_num_leaves}
                        onChange={(e) => setHyperparameters(prev => ({ ...prev, lgb_num_leaves: Number(e.target.value) }))}
                        className="input-field w-full text-sm"
                      />
                    </div>
                  </div>
                </div>
              )}
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
