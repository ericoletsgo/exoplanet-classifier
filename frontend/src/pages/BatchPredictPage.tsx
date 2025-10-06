import { useState, useEffect } from 'react'
import { Upload, Loader2, AlertCircle, CheckCircle, Download } from 'lucide-react'
import { api } from '../lib/api'

interface ColumnMapping {
  csvColumn: string
  featureName: string
  mapped: boolean
}

interface PredictionResult {
  row: number
  prediction: string
  confidence: number
  [key: string]: any
}

export default function BatchPredictPage() {
  const [csvData, setCsvData] = useState<any[]>([])
  const [columnMappings, setColumnMappings] = useState<ColumnMapping[]>([])
  const [predictions, setPredictions] = useState<PredictionResult[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [step, setStep] = useState<'upload' | 'mapping' | 'results'>('upload')
  const [availableFeatures, setAvailableFeatures] = useState<string[]>([])
  const [models, setModels] = useState<any[]>([])
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null)

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

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFile = e.target.files?.[0]
    if (!uploadedFile) return

    setError(null)
    setLoading(true)

    try {
      // Read CSV file
      const text = await uploadedFile.text()
      const lines = text.split('\n').filter(line => line.trim())
      
      // Parse CSV (simple parsing - assumes comma-separated)
      const headers = lines[0].split(',').map(h => h.trim())
      const data = lines.slice(1).map(line => {
        const values = line.split(',')
        const row: any = {}
        headers.forEach((header, idx) => {
          row[header] = values[idx]?.trim()
        })
        return row
      })

      setCsvData(data)

      // Get available features from API
      const featuresResponse = await api.getFeatures()
      const allFeatures = Object.values(featuresResponse.features).flat()
      setAvailableFeatures(allFeatures)

      // Auto-map columns that match feature names
      const mappings: ColumnMapping[] = headers.map(col => {
        const matchedFeature = allFeatures.find(f => 
          f.toLowerCase() === col.toLowerCase() || 
          col.toLowerCase().includes(f.toLowerCase())
        )
        return {
          csvColumn: col,
          featureName: matchedFeature || '',
          mapped: !!matchedFeature
        }
      })

      setColumnMappings(mappings)
      setStep('mapping')
    } catch (err) {
      setError('Failed to parse CSV file')
    } finally {
      setLoading(false)
    }
  }

  const updateMapping = (csvColumn: string, featureName: string) => {
    setColumnMappings(prev => prev.map(m => 
      m.csvColumn === csvColumn 
        ? { ...m, featureName, mapped: !!featureName }
        : m
    ))
  }

  const handlePredict = async () => {
    setLoading(true)
    setError(null)

    try {
      const results: PredictionResult[] = []

      // Process each row
      for (let i = 0; i < csvData.length; i++) {
        const row = csvData[i]
        
        // Map CSV columns to features
        const features: Record<string, number> = {}
        columnMappings.forEach(mapping => {
          if (mapping.mapped && mapping.featureName) {
            const value = parseFloat(row[mapping.csvColumn])
            features[mapping.featureName] = isNaN(value) ? 0 : value
          }
        })

        // Fill missing features with 0
        availableFeatures.forEach(feature => {
          if (!(feature in features)) {
            features[feature] = 0
          }
        })

        // Make prediction with selected model
        const prediction = await api.predict({ 
          features,
          model_id: selectedModelId || undefined
        })
        
        results.push({
          row: i + 1,
          ...row,
          prediction: prediction.prediction,
          confidence: prediction.confidence,
          probabilities: prediction.probabilities
        })
      }

      setPredictions(results)
      setStep('results')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  const downloadResults = () => {
    const csv = [
      // Headers
      ['Row', 'Prediction', 'Confidence', 'False Positive %', 'Candidate %', 'Confirmed %'].join(','),
      // Data
      ...predictions.map(p => [
        p.row,
        p.prediction,
        (p.confidence * 100).toFixed(1) + '%',
        (p.probabilities['FALSE POSITIVE'] * 100).toFixed(1) + '%',
        (p.probabilities['CANDIDATE'] * 100).toFixed(1) + '%',
        (p.probabilities['CONFIRMED'] * 100).toFixed(1) + '%'
      ].join(','))
    ].join('\n')

    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'predictions.csv'
    a.click()
  }

  const getSummary = () => {
    const total = predictions.length
    const confirmed = predictions.filter(p => p.prediction === 'CONFIRMED').length
    const candidate = predictions.filter(p => p.prediction === 'CANDIDATE').length
    const falsePositive = predictions.filter(p => p.prediction === 'FALSE POSITIVE').length
    
    return { total, confirmed, candidate, falsePositive }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold flex items-center gap-3">
          <Upload className="w-8 h-8 text-primary-500" />
          Batch Predictions
        </h1>
        <p className="text-slate-400 mt-2">Upload a CSV file to classify multiple candidates</p>
      </div>

      {/* Model Selection */}
      {models.length > 0 && (
        <div className="card">
          <h3 className="text-lg font-semibold mb-3">Select Model</h3>
          <select
            value={selectedModelId || ''}
            onChange={(e) => setSelectedModelId(e.target.value || null)}
            className="input-field w-full"
          >
            <option value="">Default Model (Latest)</option>
            {models.map((model) => (
              <option key={model.id} value={model.id}>
                {model.name} - {(model.test_accuracy * 100).toFixed(1)}% accuracy
              </option>
            ))}
          </select>
          <p className="text-xs text-slate-500 mt-2">
            {selectedModelId 
              ? `Using model: ${models.find(m => m.id === selectedModelId)?.name || 'Unknown'}`
              : 'Using the default model for batch predictions'}
          </p>
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

      {/* Step 1: Upload */}
      {step === 'upload' && (
        <div className="card">
          <h3 className="text-xl font-semibold mb-4">Upload CSV File</h3>
          <div className="border-2 border-dashed border-slate-600 rounded-lg p-8 text-center">
            <Upload className="w-12 h-12 text-slate-400 mx-auto mb-4" />
            <p className="text-slate-300 mb-4">
              Drop your CSV file here or click to browse
            </p>
            <input
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
              className="hidden"
              id="csv-upload"
            />
            <label htmlFor="csv-upload" className="btn-primary cursor-pointer inline-block">
              Select CSV File
            </label>
          </div>
        </div>
      )}

      {/* Step 2: Column Mapping */}
      {step === 'mapping' && (
        <div className="space-y-4">
          <div className="card">
            <h3 className="text-xl font-semibold mb-4">Map CSV Columns to Features</h3>
            <p className="text-sm text-slate-400 mb-4">
              {columnMappings.filter(m => m.mapped).length} of {columnMappings.length} columns mapped
            </p>
            
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {columnMappings.map(mapping => (
                <div key={mapping.csvColumn} className="flex items-center gap-4 p-3 bg-slate-800 rounded">
                  <div className="flex-1">
                    <p className="font-medium">{mapping.csvColumn}</p>
                    <p className="text-xs text-slate-500">CSV Column</p>
                  </div>
                  <div className="flex-1">
                    <select
                      value={mapping.featureName}
                      onChange={(e) => updateMapping(mapping.csvColumn, e.target.value)}
                      className="input-field w-full"
                    >
                      <option value="">-- Skip this column --</option>
                      {availableFeatures.map(feature => (
                        <option key={feature} value={feature}>{feature}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    {mapping.mapped ? (
                      <CheckCircle className="w-5 h-5 text-green-400" />
                    ) : (
                      <div className="w-5 h-5 border-2 border-slate-600 rounded-full" />
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="flex gap-3">
            <button onClick={() => setStep('upload')} className="btn-secondary">
              Back
            </button>
            <button
              onClick={handlePredict}
              disabled={loading || columnMappings.filter(m => m.mapped).length === 0}
              className="btn-primary flex items-center gap-2"
            >
              {loading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Processing {csvData.length} rows...
                </>
              ) : (
                <>
                  Run Predictions ({csvData.length} rows)
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {/* Step 3: Results */}
      {step === 'results' && predictions.length > 0 && (
        <div className="space-y-4">
          {/* Summary */}
          <div className="grid md:grid-cols-4 gap-4">
            <div className="card">
              <p className="text-sm text-slate-400">Total Processed</p>
              <p className="text-3xl font-bold">{getSummary().total}</p>
            </div>
            <div className="card bg-green-900/20 border-green-700">
              <p className="text-sm text-slate-400">Confirmed Planets</p>
              <p className="text-3xl font-bold text-green-400">{getSummary().confirmed}</p>
            </div>
            <div className="card bg-yellow-900/20 border-yellow-700">
              <p className="text-sm text-slate-400">Candidates</p>
              <p className="text-3xl font-bold text-yellow-400">{getSummary().candidate}</p>
            </div>
            <div className="card bg-red-900/20 border-red-700">
              <p className="text-sm text-slate-400">False Positives</p>
              <p className="text-3xl font-bold text-red-400">{getSummary().falsePositive}</p>
            </div>
          </div>

          {/* Results Table */}
          <div className="card">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-semibold">Prediction Results</h3>
              <button onClick={downloadResults} className="btn-secondary flex items-center gap-2">
                <Download className="w-4 h-4" />
                Download CSV
              </button>
            </div>
            
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-slate-700">
                    <th className="text-left p-2">Row</th>
                    <th className="text-left p-2">Prediction</th>
                    <th className="text-left p-2">Confidence</th>
                    <th className="text-left p-2">Probabilities</th>
                  </tr>
                </thead>
                <tbody>
                  {predictions.slice(0, 50).map(pred => (
                    <tr key={pred.row} className="border-b border-slate-800">
                      <td className="p-2">{pred.row}</td>
                      <td className="p-2">
                        <span className={`px-2 py-1 rounded text-sm ${
                          pred.prediction === 'CONFIRMED' ? 'bg-green-900/50 text-green-400' :
                          pred.prediction === 'CANDIDATE' ? 'bg-yellow-900/50 text-yellow-400' :
                          'bg-red-900/50 text-red-400'
                        }`}>
                          {pred.prediction}
                        </span>
                      </td>
                      <td className="p-2">{(pred.confidence * 100).toFixed(1)}%</td>
                      <td className="p-2 text-sm">
                        <div className="space-y-1">
                          <div>FP: {(pred.probabilities['FALSE POSITIVE'] * 100).toFixed(1)}%</div>
                          <div>Cand: {(pred.probabilities['CANDIDATE'] * 100).toFixed(1)}%</div>
                          <div>Conf: {(pred.probabilities['CONFIRMED'] * 100).toFixed(1)}%</div>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {predictions.length > 50 && (
                <p className="text-sm text-slate-400 mt-4 text-center">
                  Showing first 50 of {predictions.length} results. Download CSV for full results.
                </p>
              )}
            </div>
          </div>

          <button onClick={() => { setStep('upload'); setPredictions([]) }} className="btn-secondary">
            Upload Another File
          </button>
        </div>
      )}
    </div>
  )
}
