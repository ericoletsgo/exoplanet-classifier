import { useState, useEffect } from 'react'
import { Upload, AlertCircle, Trash2, BarChart3, Brain, Loader2 } from 'lucide-react'
import { api } from '../lib/api'
import LoadingScreen, { LoadingSpinner } from '../components/LoadingScreen'

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
  const [includeK2, setIncludeK2] = useState(false)
  const [selectedAlgorithms, setSelectedAlgorithms] = useState<string[]>(['gradient_boosting', 'random_forest', 'xgboost', 'lightgbm'])
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [uploadProgress, setUploadProgress] = useState<string>('')
  const [csvContent, setCsvContent] = useState<string>('')
  const [trainingProgress, setTrainingProgress] = useState<number>(0)
  const [showTrainingScreen, setShowTrainingScreen] = useState(false)
  
  // Target column selection
  const [datasetColumns, setDatasetColumns] = useState<any[]>([])
  const [loadingColumns, setLoadingColumns] = useState(false)
  const [targetColumn, setTargetColumn] = useState<string>('')
  const [targetMapping, setTargetMapping] = useState<Record<string, number>>({})
  const [availableTargetValues, setAvailableTargetValues] = useState<string[]>([])
  
  // Model evaluation after training
  const [trainedModelId, setTrainedModelId] = useState<string | null>(null)
  const [modelEvaluation, setModelEvaluation] = useState<any>(null)
  
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
    // Defer non-essential API calls to improve initial load
    setTimeout(() => {
      loadModels()
      loadAvailableAlgorithms()
    }, 200)
  }, [])

  useEffect(() => {
    if (selectedDataset && selectedDataset !== 'combined') {
      loadDatasetColumns()
    } else {
      setDatasetColumns([])
      setTargetColumn('')
    }
  }, [selectedDataset])

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

  const loadDatasetColumns = async () => {
    setLoadingColumns(true)
    try {
      const datasetName = selectedDataset.replace('.csv', '')
      const response = await api.getDatasetColumns(datasetName)
      setDatasetColumns(response.columns)
      
      // Find potential target columns
      const dispositionCols = response.columns.filter(col => 
        col.name.toLowerCase().includes('disp') || 
        col.name.toLowerCase().includes('status')
      )
      
      if (dispositionCols.length > 0 && dispositionCols[0].sample_values) {
        setTargetColumn(dispositionCols[0].name)
        setAvailableTargetValues(dispositionCols[0].sample_values || [])
        
        // Auto-set default mapping
        const defaultMapping: Record<string, number> = {}
        dispositionCols[0].sample_values?.forEach((val: string) => {
          if (val.toUpperCase().includes('CONFIRMED')) defaultMapping[val] = 2
          else if (val.toUpperCase().includes('CANDIDATE')) defaultMapping[val] = 1
          else if (val.toUpperCase().includes('FALSE') || val.toUpperCase().includes('POSITIVE')) defaultMapping[val] = 0
        })
        setTargetMapping(defaultMapping)
      }
    } catch (err) {
      console.error('Failed to load dataset columns:', err)
    } finally {
      setLoadingColumns(false)
    }
  }

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setUploadedFile(file)
      setError(null)
      
      // Parse CSV to get columns
      try {
        const text = await file.text()
        setCsvContent(text) // Store CSV content for training
        const lines = text.split('\n').filter(line => !line.startsWith('#'))
        if (lines.length > 0) {
          const headers = lines[0].split(',')
          
          // Create column info from CSV headers
          const csvColumns = headers.map(header => ({
            name: header.trim(),
            type: 'unknown',
            non_null_count: 0,
            null_count: 0,
            sample_values: [],
            unique_count: 0
          }))
          
          setDatasetColumns(csvColumns)
          
          // Try to find disposition column
          const dispositionCol = headers.find(h => 
            h.toLowerCase().includes('disp') || 
            h.toLowerCase().includes('status')
          )
          
          if (dispositionCol) {
            setTargetColumn(dispositionCol.trim())
            
            // Parse sample values from first 100 rows
            const colIndex = headers.indexOf(dispositionCol)
            const values = new Set<string>()
            for (let i = 1; i < Math.min(100, lines.length); i++) {
              const cols = lines[i].split(',')
              if (cols[colIndex]) {
                values.add(cols[colIndex].trim())
              }
            }
            
            const uniqueValues = Array.from(values).filter(v => v && v !== '')
            setAvailableTargetValues(uniqueValues)
            
            // Auto-set default mapping
            const defaultMapping: Record<string, number> = {}
            uniqueValues.forEach((val: string) => {
              if (val.toUpperCase().includes('CONFIRMED')) defaultMapping[val] = 2
              else if (val.toUpperCase().includes('CANDIDATE')) defaultMapping[val] = 1
              else if (val.toUpperCase().includes('FALSE') || val.toUpperCase().includes('POSITIVE')) defaultMapping[val] = 0
            })
            setTargetMapping(defaultMapping)
          }
        }
      } catch (err) {
        console.error('Failed to parse CSV:', err)
      }
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

    if (!selectedDataset && !uploadedFile) {
      setError('Please select a dataset or upload a CSV file')
      return
    }

    setLoading(true)
    setError(null)
    setUploadProgress('')
    setShowTrainingScreen(true)
    setTrainingProgress(0)

    // Simulate progress updates during training
    const progressInterval = setInterval(() => {
      setTrainingProgress(prev => {
        if (prev >= 90) return prev // Stop at 90% until actual completion
        return prev + Math.random() * 10
      })
    }, 2000)

    try {
      setUploadProgress('Training advanced ensemble... This may take a few minutes.')
      
      // Call the enhanced training API
      const result = await api.trainModel({
        dataset: selectedDataset || 'uploaded.csv',
        model_name: modelName,
        description: description,
        test_size: testSize / 100, // Convert percentage to decimal
        algorithms: selectedAlgorithms,
        hyperparameters: hyperparameters,
        use_hyperparameter_tuning: useHyperparameterTuning,
        include_k2: includeK2,
        target_column: targetColumn || undefined,
        target_mapping: Object.keys(targetMapping).length > 0 ? targetMapping : undefined,
        csv_data: uploadedFile ? csvContent : undefined
      })
      
      // Complete progress
      setTrainingProgress(100)
      clearInterval(progressInterval)
      
      if (result.cv_accuracy && result.algorithms_used) {
        setUploadProgress(`🎯 Ensemble trained successfully! CV Accuracy: ${(result.cv_accuracy * 100).toFixed(1)}%, Test Accuracy: ${(result.metrics?.accuracy * 100).toFixed(1)}%, Used ${result.algorithms_used.length} algorithms`)
      } else {
        setUploadProgress(`Model training completed! Accuracy: ${(result.metrics?.accuracy * 100).toFixed(1)}%`)
      }
      
      // Load model evaluation
      if (result.model_id) {
        setTrainedModelId(result.model_id)
        try {
          const evaluation = await api.evaluateModel(result.model_id)
          setModelEvaluation(evaluation)
        } catch (evalErr) {
          console.error('Failed to load model evaluation:', evalErr)
        }
      }
      
      await loadModels() // Refresh models list
      
      // Hide training screen after a delay
      setTimeout(() => {
        setShowTrainingScreen(false)
      }, 2000)
      
    } catch (err) {
      clearInterval(progressInterval)
      setError(err instanceof Error ? err.message : 'Training failed')
      setShowTrainingScreen(false)
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
      {/* Training Loading Screen */}
      {showTrainingScreen && (
        <LoadingScreen 
          message="Training Model" 
          subMessage="Training ensemble with multiple algorithms..."
          type="training"
          progress={trainingProgress}
          showProgress={true}
        />
      )}

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

            {/* Dataset Selection - Optional when CSV uploaded */}
            {!uploadedFile && (
              <div className="mb-6">
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Select Built-in Dataset
                </label>
                <select
                  value={selectedDataset}
                  onChange={(e) => setSelectedDataset(e.target.value)}
                  className="input-field w-full"
                >
                  <option value="koi.csv">KOI Dataset (Kepler Objects ofInterest)</option>
                  <option value="k2.csv">K2 Dataset</option>
                  <option value="combined">Combined Dataset (Multi-Mission)</option>
                </select>
                <p className="text-xs text-slate-500 mt-1">
                  Choose which NASA dataset to train on, or upload your own CSV below.
                </p>
              </div>
            )}

            {/* OR Divider */}
            {!uploadedFile && (
              <div className="mb-6 flex items-center gap-4">
                <div className="flex-1 border-t border-slate-700"></div>
                <span className="text-slate-500 text-sm">OR</span>
                <div className="flex-1 border-t border-slate-700"></div>
              </div>
            )}

            {/* Upload Your Own CSV */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Upload Your Own CSV
              </label>
              <div className="border-2 border-dashed border-slate-600 rounded-lg p-6 text-center">
                <Upload className="w-10 h-10 text-slate-400 mx-auto mb-3" />
                <p className="text-slate-300 mb-2">
                  {uploadedFile ? `✓ ${uploadedFile.name}` : 'Drop your CSV file here or click to browse'}
                </p>
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleFileUpload}
                  className="hidden"
                  id="csv-upload"
                />
                <label htmlFor="csv-upload" className="btn-primary cursor-pointer inline-block text-sm px-4 py-2">
                  {uploadedFile ? 'Change File' : 'Select CSV File'}
                </label>
                {uploadedFile && (
                  <button
                    onClick={() => {
                      setUploadedFile(null)
                      setCsvContent('')
                      setDatasetColumns([])
                      setTargetColumn('')
                      setTargetMapping({})
                    }}
                    className="ml-2 text-sm text-red-400 hover:text-red-300"
                  >
                    Remove
                  </button>
                )}
                <p className="text-xs text-slate-500 mt-2">
                  {uploadedFile ? 'Training will use this uploaded CSV file' : 'Upload a CSV to train on your own data'}
                </p>
              </div>
            </div>

            {/* Target Column Selection - for uploaded files or datasets */}
            {((uploadedFile && datasetColumns.length > 0) || (selectedDataset !== 'combined' && datasetColumns.length > 0)) && (
              <div className="mb-6 p-4 bg-slate-800/50 rounded-lg border border-slate-700">
                <h4 className="text-sm font-semibold text-primary-400 mb-3">
                  Target Column Configuration
                  {uploadedFile && <span className="text-xs text-slate-400 ml-2">(from uploaded CSV)</span>}
                </h4>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      Select Target Column
                    </label>
                    <select
                      value={targetColumn}
                      onChange={(e) => {
                        setTargetColumn(e.target.value)
                        const col = datasetColumns.find(c => c.name === e.target.value)
                        if (col?.sample_values) {
                          setAvailableTargetValues(col.sample_values)
                        }
                      }}
                      className="input-field w-full"
                      disabled={loadingColumns}
                    >
                      <option value="">Auto-detect (recommended)</option>
                      {datasetColumns.map(col => (
                        <option key={col.name} value={col.name}>
                          {col.name}
                          {col.unique_count ? ` (${col.unique_count} unique values)` : ''}
                        </option>
                      ))}
                    </select>
                    <p className="text-xs text-slate-500 mt-1">
                      {loadingColumns ? 'Loading columns...' : 'Select the column containing classification labels'}
                    </p>
                  </div>
                  
                  {targetColumn && availableTargetValues.length > 0 && (
                    <div className="space-y-2">
                      <label className="block text-sm font-medium text-slate-300">
                        Map Values to Classes
                      </label>
                      <div className="grid grid-cols-3 gap-2 text-xs">
                        <div className="font-semibold text-slate-400">Value</div>
                        <div className="font-semibold text-slate-400">→</div>
                        <div className="font-semibold text-slate-400">Class</div>
                      </div>
                      {availableTargetValues.map(val => (
                        <div key={val} className="grid grid-cols-3 gap-2 items-center">
                          <div className="text-sm text-slate-300 truncate">{val}</div>
                          <div className="text-slate-500">→</div>
                          <select
                            value={targetMapping[val] ?? ''}
                            onChange={(e) => {
                              const newMapping = { ...targetMapping }
                              if (e.target.value === '') {
                                delete newMapping[val]
                              } else {
                                newMapping[val] = parseInt(e.target.value)
                              }
                              setTargetMapping(newMapping)
                            }}
                            className="input-field text-sm py-1"
                          >
                            <option value="">Skip</option>
                            <option value="2">Confirmed (2)</option>
                            <option value="1">Candidate (1)</option>
                            <option value="0">False Positive (0)</option>
                          </select>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Multi-Dataset Options */}
            {selectedDataset === 'combined' && (
              <div className="mb-6 p-4 bg-slate-800/50 rounded-lg border border-slate-700">
                <h4 className="text-sm font-semibold text-primary-400 mb-3">Multi-Mission Training Options</h4>
                <div className="space-y-3">
                  <div className="flex items-center space-x-3">
                    <input
                      type="checkbox"
                      id="include-k2"
                      checked={includeK2}
                      onChange={(e) => setIncludeK2(e.target.checked)}
                      className="rounded border-slate-600 bg-slate-700 text-primary-500 focus:ring-primary-500"
                    />
                    <label htmlFor="include-k2" className="text-sm text-slate-300">
                      Include K2 Mission Data (4,004 samples)
                    </label>
                  </div>
                </div>
                <p className="text-xs text-slate-500 mt-2">
                  KOI dataset is always included. Check to add K2 mission data.
                </p>
              </div>
            )}
            
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

            {uploadProgress && (
              <div className="mb-6 p-4 bg-blue-900/20 border border-blue-700 rounded-lg">
                <div className="flex items-center gap-2 text-blue-400">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <p>{uploadProgress}</p>
                </div>
              </div>
            )}

            {/* Model Evaluation After Training */}
            {modelEvaluation && trainedModelId && (
              <div className="mb-6 p-6 bg-green-900/10 border border-green-700/50 rounded-lg">
                <h4 className="text-lg font-semibold text-green-400 mb-4 flex items-center gap-2">
                  <BarChart3 className="w-5 h-5" />
                  Model Evaluation Results
                </h4>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                  <div className="bg-slate-800/50 p-3 rounded">
                    <div className="text-xs text-slate-400 mb-1">Test Accuracy</div>
                    <div className="text-2xl font-bold text-primary-400">
                      {(modelEvaluation.metrics.test_accuracy * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="bg-slate-800/50 p-3 rounded">
                    <div className="text-xs text-slate-400 mb-1">Precision</div>
                    <div className="text-2xl font-bold text-blue-400">
                      {(modelEvaluation.metrics.precision * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="bg-slate-800/50 p-3 rounded">
                    <div className="text-xs text-slate-400 mb-1">Recall</div>
                    <div className="text-2xl font-bold text-purple-400">
                      {(modelEvaluation.metrics.recall * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="bg-slate-800/50 p-3 rounded">
                    <div className="text-xs text-slate-400 mb-1">F1 Score</div>
                    <div className="text-2xl font-bold text-green-400">
                      {modelEvaluation.metrics.f1_score.toFixed(3)}
                    </div>
                  </div>
                </div>

                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <h5 className="text-sm font-semibold text-slate-300 mb-2">Model Info</h5>
                    <div className="text-xs space-y-1 text-slate-400">
                      <div><span className="font-medium">Algorithms:</span> {modelEvaluation.model_info.algorithms.join(', ')}</div>
                      <div><span className="font-medium">Features:</span> {modelEvaluation.model_info.n_features}</div>
                      <div><span className="font-medium">Train Samples:</span> {modelEvaluation.dataset_info.train_samples}</div>
                      <div><span className="font-medium">Test Samples:</span> {modelEvaluation.dataset_info.test_samples}</div>
                    </div>
                  </div>
                  
                  <div>
                    <h5 className="text-sm font-semibold text-slate-300 mb-2">Confusion Matrix</h5>
                    <div className="grid grid-cols-3 gap-1 text-xs">
                      {modelEvaluation.confusion_matrix.map((row: number[], i: number) => 
                        row.map((val: number, j: number) => (
                          <div 
                            key={`${i}-${j}`}
                            className={`p-2 text-center rounded ${
                              i === j ? 'bg-green-900/30 text-green-300' : 'bg-slate-800/50 text-slate-400'
                            }`}
                          >
                            {val}
                          </div>
                        ))
                      )}
                    </div>
                    <div className="text-xs text-slate-500 mt-1">
                      Rows: Actual, Columns: Predicted
                    </div>
                  </div>
                </div>
              </div>
            )}

            <button
              onClick={handleTrainModel}
              disabled={loading || !modelName}
              className="btn-primary w-full flex items-center justify-center gap-2"
            >
              {loading ? (
                <LoadingSpinner size="sm" message="Starting Training..." />
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
          {/* Selected Model Evaluation Display */}
          {modelEvaluation && trainedModelId && (
            <div className="card">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-xl font-semibold">Model Details: {modelEvaluation.model_info.name}</h3>
                <button
                  onClick={() => {
                    setModelEvaluation(null)
                    setTrainedModelId(null)
                  }}
                  className="text-sm text-slate-400 hover:text-slate-300"
                >
                  Close
                </button>
              </div>
              
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                <div className="bg-slate-800/50 p-3 rounded">
                  <div className="text-xs text-slate-400 mb-1">Test Accuracy</div>
                  <div className="text-2xl font-bold text-primary-400">
                    {(modelEvaluation.metrics.test_accuracy * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="bg-slate-800/50 p-3 rounded">
                  <div className="text-xs text-slate-400 mb-1">Precision</div>
                  <div className="text-2xl font-bold text-blue-400">
                    {(modelEvaluation.metrics.precision * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="bg-slate-800/50 p-3 rounded">
                  <div className="text-xs text-slate-400 mb-1">Recall</div>
                  <div className="text-2xl font-bold text-purple-400">
                    {(modelEvaluation.metrics.recall * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="bg-slate-800/50 p-3 rounded">
                  <div className="text-xs text-slate-400 mb-1">F1 Score</div>
                  <div className="text-2xl font-bold text-green-400">
                    {modelEvaluation.metrics.f1_score.toFixed(3)}
                  </div>
                </div>
              </div>

              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <h5 className="text-sm font-semibold text-slate-300 mb-2">Model Info</h5>
                  <div className="text-xs space-y-1 text-slate-400">
                    <div><span className="font-medium">ID:</span> {modelEvaluation.model_id}</div>
                    <div><span className="font-medium">Algorithms:</span> {modelEvaluation.model_info.algorithms.join(', ')}</div>
                    <div><span className="font-medium">Features:</span> {modelEvaluation.model_info.n_features}</div>
                    <div><span className="font-medium">Train Samples:</span> {modelEvaluation.dataset_info.train_samples}</div>
                    <div><span className="font-medium">Test Samples:</span> {modelEvaluation.dataset_info.test_samples}</div>
                    <div><span className="font-medium">Created:</span> {new Date(modelEvaluation.model_info.created_at).toLocaleString()}</div>
                  </div>
                </div>
                
                <div>
                  <h5 className="text-sm font-semibold text-slate-300 mb-2">Confusion Matrix</h5>
                  <div className="grid grid-cols-3 gap-1 text-xs">
                    {modelEvaluation.confusion_matrix.map((row: number[], i: number) => 
                      row.map((val: number, j: number) => (
                        <div 
                          key={`${i}-${j}`}
                          className={`p-2 text-center rounded ${
                            i === j ? 'bg-green-900/30 text-green-300' : 'bg-slate-800/50 text-slate-400'
                          }`}
                        >
                          {val}
                        </div>
                      ))
                    )}
                  </div>
                  <div className="text-xs text-slate-500 mt-1">
                    Rows: Actual, Columns: Predicted
                  </div>
                </div>
              </div>
            </div>
          )}

          <div className="card">
            <h3 className="text-xl font-semibold mb-4">All Models</h3>
            
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
                        <th className="text-left p-3">Actions</th>
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
                          <td className="p-3 text-sm">{model.f1_score.toFixed(3)}</td>
                          <td className="p-3 text-sm">{model.n_features}</td>
                          <td className="p-3">
                            <button
                              onClick={async () => {
                                try {
                                  const evaluation = await api.evaluateModel(model.id)
                                  setModelEvaluation(evaluation)
                                  setTrainedModelId(model.id)
                                  window.scrollTo({ top: 0, behavior: 'smooth' })
                                } catch (err) {
                                  setError(err instanceof Error ? err.message : 'Failed to load evaluation')
                                }
                              }}
                              className="text-sm text-primary-400 hover:text-primary-300"
                            >
                              View Details
                            </button>
                          </td>
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
