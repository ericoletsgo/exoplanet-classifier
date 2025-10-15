import { useState, useEffect } from 'react'
import { Database, AlertCircle, ChevronLeft, ChevronRight, Filter } from 'lucide-react'
import { api, type DatasetResponse } from '../lib/api'
import { getDispositionColor } from '../lib/utils'
import LoadingScreen from '../components/LoadingScreen'

export default function DatasetsPage() {
  const [dataset, setDataset] = useState<DatasetResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedDataset, setSelectedDataset] = useState<'koi' | 'k2' | 'toi'>('koi')
  const [currentPage, setCurrentPage] = useState(1)
  const [filterDisposition, setFilterDisposition] = useState<string>('')
  const [pageSize] = useState(50)

  useEffect(() => {
    loadDataset()
  }, [selectedDataset, currentPage, filterDisposition])

  const loadDataset = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.getDataset(
        selectedDataset,
        currentPage,
        pageSize,
        filterDisposition || undefined
      )
      setDataset(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load dataset')
    } finally {
      setLoading(false)
    }
  }

  const handleDatasetChange = (newDataset: 'koi' | 'k2' | 'toi') => {
    setSelectedDataset(newDataset)
    setCurrentPage(1)
  }

  const handleFilterChange = (disposition: string) => {
    setFilterDisposition(disposition)
    setCurrentPage(1)
  }

  const goToPage = (page: number) => {
    if (page >= 1 && dataset && page <= dataset.total_pages) {
      setCurrentPage(page)
    }
  }

  const displayColumns = dataset?.columns.slice(0, 10) || []

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold flex items-center gap-3">
          <Database className="w-8 h-8 text-primary-500" />
          Datasets
        </h1>
        <p className="text-slate-400 mt-2">Browse exoplanet candidate datasets</p>
      </div>

      {/* Dataset Selector */}
      <div className="card">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex gap-2">
            {(['koi', 'k2', 'toi'] as const).map((ds) => (
              <button
                key={ds}
                onClick={() => handleDatasetChange(ds)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  selectedDataset === ds
                    ? 'bg-primary-600 text-white'
                    : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                }`}
              >
                {ds.toUpperCase()}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-2 ml-auto">
            <Filter className="w-4 h-4 text-slate-400" />
            <select
              value={filterDisposition}
              onChange={(e) => handleFilterChange(e.target.value)}
              className="input-field"
            >
              <option value="">All Dispositions</option>
              <option value="CONFIRMED">Confirmed</option>
              <option value="CANDIDATE">Candidate</option>
              <option value="FALSE POSITIVE">False Positive</option>
            </select>
          </div>
        </div>

        {dataset && (
          <div className="mt-4 text-sm text-slate-400">
            Showing {((currentPage - 1) * pageSize) + 1} - {Math.min(currentPage * pageSize, dataset.total_rows)} of {dataset.total_rows.toLocaleString()} rows
          </div>
        )}
      </div>

      {error && (
        <div className="card bg-red-900/20 border-red-700">
          <div className="flex items-center gap-2 text-red-400">
            <AlertCircle className="w-5 h-5" />
            <p>{error}</p>
          </div>
        </div>
      )}

      {/* Data Table */}
      {loading ? (
        <LoadingScreen 
          message="Loading Dataset" 
          subMessage="Fetching data from database..."
          type="dataset"
        />
      ) : dataset ? (
        <div className="card overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-slate-700">
                  {displayColumns.map((col) => (
                    <th
                      key={col}
                      className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider"
                    >
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-700">
                {dataset.data.map((row, idx) => (
                  <tr key={idx} className="hover:bg-slate-700/50 transition-colors">
                    {displayColumns.map((col) => {
                      const value = row[col]
                      const isDisposition = col === 'koi_disposition'
                      
                      return (
                        <td key={col} className="px-4 py-3 text-sm">
                          {isDisposition && value ? (
                            <span className={`inline-block px-2 py-1 rounded text-xs font-medium border ${getDispositionColor(value)}`}>
                              {value}
                            </span>
                          ) : value !== null && value !== undefined ? (
                            typeof value === 'number' ? (
                              value.toFixed(4)
                            ) : (
                              String(value)
                            )
                          ) : (
                            <span className="text-slate-500">—</span>
                          )}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="flex items-center justify-between px-4 py-3 border-t border-slate-700">
            <div className="text-sm text-slate-400">
              Page {currentPage} of {dataset.total_pages}
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => goToPage(currentPage - 1)}
                disabled={currentPage === 1}
                className="btn-secondary disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1"
              >
                <ChevronLeft className="w-4 h-4" />
                Previous
              </button>
              <button
                onClick={() => goToPage(currentPage + 1)}
                disabled={currentPage === dataset.total_pages}
                className="btn-secondary disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1"
              >
                Next
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  )
}
