import { useState, useEffect } from 'react'
import { BarChart3, Loader2, AlertCircle, TrendingUp, Grid3X3, GitCompare } from 'lucide-react'
import { api, type MetricsResponse } from '../lib/api'
import { formatPercentage } from '../lib/utils'
import HeatmapGrid from 'react-heatmap-grid'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  PointElement,
  LineElement
} from 'chart.js'
import { Bar } from 'react-chartjs-2'

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  PointElement,
  LineElement
)

export default function MetricsPage() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null)
  const [correlations, setCorrelations] = useState<any>(null)
  const [models, setModels] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'overview' | 'correlations' | 'comparison'>('overview')

  useEffect(() => {
    // Only load lightweight data on mount
    loadModels()
  }, [])

  // Load metrics when overview tab is accessed
  useEffect(() => {
    if (activeTab === 'overview' && !metrics && !loading) {
      loadMetrics()
    }
  }, [activeTab, metrics, loading])

  // Load correlations when correlations tab is accessed
  useEffect(() => {
    if (activeTab === 'correlations' && !correlations) {
      loadCorrelations()
    }
  }, [activeTab, correlations])

  const loadMetrics = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.getMetrics()
      setMetrics(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load metrics')
    } finally {
      setLoading(false)
    }
  }

  const loadCorrelations = async () => {
    try {
      const data = await api.getFeatureCorrelations()
      setCorrelations(data)
    } catch (err) {
      console.error('Failed to load correlations:', err)
    }
  }

  const loadModels = async () => {
    try {
      const data = await api.listModels()
      setModels(data.models || [])
    } catch (err) {
      console.error('Failed to load models:', err)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold flex items-center gap-3">
          <BarChart3 className="w-8 h-8 text-primary-500" />
          Model Metrics & Analytics
        </h1>
        <p className="text-slate-400 mt-2">Performance analysis, feature correlations, and model comparison</p>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-slate-700">
        <nav className="flex space-x-8">
          {[
            { id: 'overview', label: 'Overview', icon: BarChart3 },
            { id: 'correlations', label: 'Feature Correlations', icon: Grid3X3 },
            { id: 'comparison', label: 'Model Comparison', icon: GitCompare },
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

      {/* Tab Content */}
      {activeTab === 'overview' && (
        <div className="space-y-6">
          {loading ? (
            <div className="card">
              <div className="text-center py-8">
                <Loader2 className="w-8 h-8 animate-spin text-primary-500 mx-auto mb-4" />
                <p className="text-slate-400">Loading model performance metrics...</p>
                <p className="text-sm text-slate-500 mt-2">This may take a few moments as we calculate accuracy, precision, and other metrics.</p>
              </div>
            </div>
          ) : error ? (
            <div className="card bg-red-900/20 border-red-700">
              <div className="flex items-center gap-2 text-red-400">
                <AlertCircle className="w-5 h-5" />
                <p>{error}</p>
              </div>
            </div>
          ) : metrics ? (
            <>
              {/* Key Metrics */}
              <div className="grid md:grid-cols-4 gap-4">
                {[
                  { label: 'Accuracy', value: metrics.accuracy, icon: TrendingUp },
                  { label: 'Precision', value: metrics.precision, icon: TrendingUp },
                  { label: 'Recall', value: metrics.recall, icon: TrendingUp },
                  { label: 'F1 Score', value: metrics.f1_score, icon: TrendingUp },
                ].map((metric) => {
                  const Icon = metric.icon
                  return (
                    <div key={metric.label} className="card">
                      <div className="flex items-center justify-between mb-2">
                        <p className="text-sm text-slate-400">{metric.label}</p>
                        <Icon className="w-4 h-4 text-primary-500" />
                      </div>
                      <p className="text-3xl font-bold text-white">
                        {formatPercentage(metric.value)}
                      </p>
                    </div>
                  )
                })}
              </div>

              {/* Model Info */}
              <div className="card">
                <h3 className="text-xl font-semibold mb-4">Model Information</h3>
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <p className="text-sm text-slate-400 mb-1">Model Type</p>
                    <p className="font-semibold">{metrics.model_info.model_type}</p>
                  </div>
                  <div>
                    <p className="text-sm text-slate-400 mb-1">Features Used</p>
                    <p className="font-semibold">{metrics.model_info.n_features}</p>
                  </div>
                  <div>
                    <p className="text-sm text-slate-400 mb-1">Training Samples</p>
                    <p className="font-semibold">{metrics.model_info.n_samples.toLocaleString()}</p>
                  </div>
                  <div>
                    <p className="text-sm text-slate-400 mb-1">Classes</p>
                    <p className="font-semibold">{metrics.model_info.classes.join(', ')}</p>
                  </div>
                </div>
              </div>

              {/* Confusion Matrix */}
              <div className="card">
                <h3 className="text-xl font-semibold mb-4">Confusion Matrix</h3>
                <div className="h-80">
                  <Bar
                    data={{
                      labels: ['False Positive', 'Candidate', 'Confirmed Planet'],
                      datasets: [
                        {
                          label: 'Predicted as False Positive',
                          data: [metrics.confusion_matrix[0][0], metrics.confusion_matrix[1][0], metrics.confusion_matrix[2][0]],
                          backgroundColor: 'rgba(239, 68, 68, 0.8)',
                        },
                        {
                          label: 'Predicted as Candidate',
                          data: [metrics.confusion_matrix[0][1], metrics.confusion_matrix[1][1], metrics.confusion_matrix[2][1]],
                          backgroundColor: 'rgba(245, 158, 11, 0.8)',
                        },
                        {
                          label: 'Predicted as Confirmed',
                          data: [metrics.confusion_matrix[0][2], metrics.confusion_matrix[1][2], metrics.confusion_matrix[2][2]],
                          backgroundColor: 'rgba(34, 197, 94, 0.8)',
                        },
                      ],
                    }}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: {
                          labels: {
                            color: 'rgb(203, 213, 225)',
                          },
                        },
                      },
                      scales: {
                        x: {
                          title: {
                            display: true,
                            text: 'Actual Class',
                            color: 'rgb(203, 213, 225)',
                          },
                          ticks: { color: 'rgb(148, 163, 184)' },
                          grid: { color: 'rgba(148, 163, 184, 0.1)' },
                        },
                        y: {
                          title: {
                            display: true,
                            text: 'Count',
                            color: 'rgb(203, 213, 225)',
                          },
                          ticks: { color: 'rgb(148, 163, 184)' },
                          grid: { color: 'rgba(148, 163, 184, 0.1)' },
                        },
                      },
                    }}
                  />
                </div>
              </div>

              {/* Feature Importance */}
              {metrics.feature_importances && metrics.feature_importances.length > 0 && (
                <div className="card">
                  <h3 className="text-xl font-semibold mb-4">Top Feature Importance</h3>
                  <div className="h-80">
                    <Bar
                      data={{
                        labels: metrics.feature_importances.slice(0, 10).map(f => f.feature.replace('koi_', '').replace('_', ' ')),
                        datasets: [
                          {
                            label: 'Importance',
                            data: metrics.feature_importances.slice(0, 10).map(f => f.importance),
                            backgroundColor: 'rgba(59, 130, 246, 0.8)',
                          },
                        ],
                      }}
                      options={{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                          legend: {
                            labels: {
                              color: 'rgb(203, 213, 225)',
                            },
                          },
                        },
                        scales: {
                          x: {
                            ticks: { color: 'rgb(148, 163, 184)' },
                            grid: { color: 'rgba(148, 163, 184, 0.1)' },
                          },
                          y: {
                            ticks: { color: 'rgb(148, 163, 184)' },
                            grid: { color: 'rgba(148, 163, 184, 0.1)' },
                          },
                        },
                        indexAxis: 'y' as const,
                      }}
                    />
                  </div>
                </div>
              )}
            </>
          ) : null}
        </div>
      )}

      {/* Feature Correlations Tab */}
      {activeTab === 'correlations' && (
        <div className="space-y-6">
          <div className="card">
            <h3 className="text-xl font-semibold mb-4">Feature Correlation Matrix</h3>
            {!correlations ? (
              <div className="flex items-center justify-center h-64">
                <div className="text-center">
                  <Loader2 className="w-8 h-8 animate-spin text-primary-500 mx-auto mb-4" />
                  <p className="text-slate-400">Loading correlation matrix...</p>
                  <p className="text-sm text-slate-500 mt-2">Analyzing feature relationships in the dataset.</p>
                </div>
              </div>
            ) : (
              <>
                <p className="text-sm text-slate-400 mb-4">
                  Correlation analysis based on {correlations.sample_size.toLocaleString()} samples from the dataset.
                  Darker colors indicate stronger correlations.
                </p>
                <div className="overflow-x-auto">
                  <div className="min-w-[800px]">
                    <HeatmapGrid
                      data={correlations.matrix}
                      xLabels={correlations.features.map((f: string) => f.replace('koi_', '').replace('_', ' '))}
                      yLabels={correlations.features.map((f: string) => f.replace('koi_', '').replace('_', ' '))}
                      cellStyle={(_, value) => ({
                        background: `rgba(59, 130, 246, ${Math.abs(value)})`,
                        fontSize: '10px',
                        color: '#fff',
                      })}
                      cellHeight="20px"
                      xLabelWidth="120px"
                      yLabelWidth="120px"
                    />
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* Model Comparison Tab */}
      {activeTab === 'comparison' && (
        <div className="space-y-6">
          <div className="card">
            <h3 className="text-xl font-semibold mb-4">Model Comparison</h3>
            {models.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-700">
                      <th className="text-left py-3 px-4">Model</th>
                      <th className="text-left py-3 px-4">Accuracy</th>
                      <th className="text-left py-3 px-4">Precision</th>
                      <th className="text-left py-3 px-4">Recall</th>
                      <th className="text-left py-3 px-4">F1 Score</th>
                      <th className="text-left py-3 px-4">Features</th>
                      <th className="text-left py-3 px-4">Created</th>
                    </tr>
                  </thead>
                  <tbody>
                    {models.map((model, idx) => (
                      <tr key={idx} className="border-b border-slate-800 hover:bg-slate-800/50">
                        <td className="py-3 px-4 font-medium">{model.name || `Model ${idx + 1}`}</td>
                        <td className="py-3 px-4">{formatPercentage(model.test_accuracy || model.train_accuracy || 0)}</td>
                        <td className="py-3 px-4">{formatPercentage(model.precision || 0)}</td>
                        <td className="py-3 px-4">{formatPercentage(model.recall || 0)}</td>
                        <td className="py-3 px-4">{formatPercentage(model.f1_score || 0)}</td>
                        <td className="py-3 px-4">{model.n_features || 'N/A'}</td>
                        <td className="py-3 px-4 text-slate-400">
                          {model.created_at ? new Date(model.created_at).toLocaleDateString() : 'N/A'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="text-center py-8">
                <p className="text-slate-400">No additional models found. Only the main production model is available.</p>
                <p className="text-sm text-slate-500 mt-2">
                  Train new models using the Model Retraining page to see comparisons here.
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}