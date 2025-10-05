import { useState } from 'react'
import { Upload, Loader2, AlertCircle, Download } from 'lucide-react'
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
  const [predictions, setPredictions] = useState<PredictionResult[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [step, setStep] = useState<'upload' | 'results'>('upload')

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFile = e.target.files?.[0]
    if (!uploadedFile) return

    setError(null)
    setLoading(true)

    try {
      // Read CSV file
      const text = await uploadedFile.text()
      
      // Find the first non-comment line for headers
      const lines = text.split('\n').filter(line => line.trim())
      const headerIndex = lines.findIndex(line => !line.trim().startsWith('#'))
      
      if (headerIndex === -1) {
        throw new Error('Could not find a valid header row in the CSV file.')
      }
      
      // Remove BOM and parse headers
      const headerLine = lines[headerIndex].replace(/^\uFEFF/, '')
      const headers = headerLine.split(',').map(h => h.trim())
      console.log('[Batch Upload] CSV Headers:', headers)
      
      // Get data rows (all lines after the header)
      const dataLines = lines.slice(headerIndex + 1)
      const data = dataLines.map(line => {
        const values = line.split(',')
        const row: any = {}
        headers.forEach((header, idx) => {
          row[header] = values[idx]?.trim()
        })
        return row
      })

      // Get available features from API
      const featuresResponse = await api.getFeatures()
      const allFeatures = Object.values(featuresResponse.features).flat()
      console.log('[Batch Upload] Expected Features:', allFeatures)

      // Auto-map columns that match feature names (improved matching algorithm)
      const mappings: ColumnMapping[] = headers.map(col => {
        const colLower = col.toLowerCase().trim()
        
        // Skip error/uncertainty columns (they contain _err, _unc, etc.)
        if (colLower.includes('_err') || colLower.includes('_unc') || colLower.includes('error')) {
          return {
            csvColumn: col,
            featureName: '',
            mapped: false
          }
        }
        
        // Try exact match first (highest priority)
        let matchedFeature = allFeatures.find(f => f.toLowerCase() === colLower)
        
        // Only try fuzzy matching if no exact match found
        if (!matchedFeature) {
          // Try if column contains feature name
          matchedFeature = allFeatures.find(f => colLower.includes(f.toLowerCase()))
          
          // Try if feature name contains column (reversed)
          if (!matchedFeature) {
            matchedFeature = allFeatures.find(f => f.toLowerCase().includes(colLower))
          }
          
          // Try common variations (remove underscores, spaces, etc.)
          if (!matchedFeature) {
            const colNormalized = colLower.replace(/[_\s-]/g, '')
            matchedFeature = allFeatures.find(f => {
              const featureNormalized = f.toLowerCase().replace(/[_\s-]/g, '')
              return featureNormalized === colNormalized || 
                     colNormalized.includes(featureNormalized) ||
                     featureNormalized.includes(colNormalized)
            })
          }
        }
        
        return {
          csvColumn: col,
          featureName: matchedFeature || '',
          mapped: !!matchedFeature
        }
      })
      
      // Count how many features were successfully mapped
      const mappedCount = mappings.filter(m => m.mapped).length
      const totalFeaturesNeeded = allFeatures.length
      const mappedFeatures = new Set(mappings.filter(m => m.mapped).map(m => m.featureName))
      
      console.log(`[Batch Upload] Mapped ${mappedCount} of ${headers.length} CSV columns`)
      console.log(`[Batch Upload] Covered ${mappedFeatures.size} of ${totalFeaturesNeeded} required model features`)
      console.log('[Batch Upload] Mappings:', mappings.filter(m => m.mapped))
      
      // Warn if too few features were mapped (less than 25% of required features)
      const coveragePercent = (mappedFeatures.size / totalFeaturesNeeded) * 100
      if (mappedFeatures.size < 5) {
        throw new Error(
          `Only ${mappedFeatures.size} of ${totalFeaturesNeeded} model features found in CSV (${coveragePercent.toFixed(0)}% coverage). ` +
          `Please ensure your CSV has columns matching: ${allFeatures.slice(0, 10).join(', ')}...`
        )
      }
      
      if (coveragePercent < 50) {
        console.warn(`[Batch Upload] Warning: Only ${coveragePercent.toFixed(0)}% of features mapped. Predictions may be less accurate.`)
      }
      
      // Automatically run predictions without requiring user interaction
      await runPredictions(data, mappings, allFeatures)
    } catch (err) {
      console.error('[Batch Upload] Error:', err)
      setError(err instanceof Error ? err.message : 'Failed to parse CSV file or run predictions')
    } finally {
      setLoading(false)
    }
  }

  const runPredictions = async (data: any[], mappings: ColumnMapping[], features: string[]) => {
    try {
      // Prepare all records for batch prediction
      const records: Record<string, number>[] = []
      
      for (let i = 0; i < data.length; i++) {
        const row = data[i]
        
        // Map CSV columns to features
        const featureValues: Record<string, number> = {}
        let nonZeroCount = 0
        
        mappings.forEach(mapping => {
          if (mapping.mapped && mapping.featureName) {
            const value = parseFloat(row[mapping.csvColumn])
            const finalValue = isNaN(value) ? 0 : value
            featureValues[mapping.featureName] = finalValue
            if (finalValue !== 0) nonZeroCount++
          }
        })

        // Fill missing features with 0
        features.forEach(feature => {
          if (!(feature in featureValues)) {
            featureValues[feature] = 0
          }
        })
        
        // Log first few rows for debugging
        if (i < 3) {
          console.log(`[Batch Upload] Row ${i + 1} feature values:`, featureValues)
          console.log(`[Batch Upload] Row ${i + 1} non-zero features: ${nonZeroCount}`)
          console.log(`[Batch Upload] Row ${i + 1} original data:`, row)
        }
        
        records.push(featureValues)
      }

      console.log(`[Batch Upload] Sending ${records.length} records for batch prediction`)
      
      // Make batch prediction (single API call for all rows!)
      const batchResponse = await api.batchPredict({ records })
      
      console.log(`[Batch Upload] Received ${batchResponse.predictions.length} predictions`)
      
      // Combine predictions with original data
      const results: PredictionResult[] = batchResponse.predictions.map((pred, i) => ({
        row: i + 1,
        ...data[i],
        prediction: pred.prediction,
        confidence: pred.confidence,
        probabilities: pred.probabilities
      }))
      
      // Compare with actual dispositions if available
      if (data[0] && 'koi_disposition' in data[0]) {
        const actualCounts = {
          'CONFIRMED': data.filter(row => row.koi_disposition === 'CONFIRMED').length,
          'CANDIDATE': data.filter(row => row.koi_disposition === 'CANDIDATE').length,
          'FALSE POSITIVE': data.filter(row => row.koi_disposition === 'FALSE POSITIVE').length
        }
        const predictedCounts = {
          'CONFIRMED': results.filter(r => r.prediction === 'CONFIRMED').length,
          'CANDIDATE': results.filter(r => r.prediction === 'CANDIDATE').length,
          'FALSE POSITIVE': results.filter(r => r.prediction === 'FALSE POSITIVE').length
        }
        console.log('[Batch Upload] Actual dispositions in dataset:', actualCounts)
        console.log('[Batch Upload] Model predictions:', predictedCounts)
      }

      setPredictions(results)
      setStep('results')
    } catch (err) {
      throw err
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
            <p className="text-sm text-slate-400 mb-4">
              Your file will be automatically processed and predictions will be generated
            </p>
            <input
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
              className="hidden"
              id="csv-upload"
              disabled={loading}
            />
            <label htmlFor="csv-upload" className={`btn-primary cursor-pointer inline-block ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}>
              {loading ? (
                <span className="flex items-center gap-2">
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Processing...
                </span>
              ) : (
                'Select CSV File'
              )}
            </label>
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
