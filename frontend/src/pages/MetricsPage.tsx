import { useState, useEffect } from 'react'
import { BarChart3, Loader2, AlertCircle, TrendingUp } from 'lucide-react'
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
  LineElement,
} from 'chart.js'
import { Bar, Line } from 'react-chartjs-2'
import { api, type MetricsResponse } from '../lib/api'
import { formatPercentage, formatNumber } from '../lib/utils'

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
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadMetrics()
  }, [])

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

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-primary-500" />
      </div>
    )
  }

  if (error || !metrics) {
    return (
      <div className="card bg-red-900/20 border-red-700">
        <div className="flex items-center gap-2 text-red-400">
          <AlertCircle className="w-5 h-5" />
          <p>{error || 'Failed to load metrics'}</p>
        </div>
      </div>
    )
  }

  // Confusion Matrix Chart Data
  // Calculate totals for each actual class (row sums)
  const actualTotals = metrics.confusion_matrix.map(row => row.reduce((a, b) => a + b, 0))
  
  const confusionMatrixData = {
    labels: ['False Positive', 'Candidate', 'Confirmed'],
    datasets: metrics.confusion_matrix.map((row, idx) => ({
      label: ['False Positive', 'Candidate', 'Confirmed'][idx],
      data: row,
      backgroundColor: [
        'rgba(239, 68, 68, 0.5)',
        'rgba(234, 179, 8, 0.5)',
        'rgba(34, 197, 94, 0.5)',
      ][idx],
      borderColor: [
        'rgba(239, 68, 68, 1)',
        'rgba(234, 179, 8, 1)',
        'rgba(34, 197, 94, 1)',
      ][idx],
      borderWidth: 1,
    })),
  }

  // Feature Importance Chart Data
  const topFeatures = metrics.feature_importances?.slice(0, 15) || []
  const featureImportanceData = {
    labels: topFeatures.map((f) => f.feature),
    datasets: [
      {
        label: 'Importance',
        data: topFeatures.map((f) => f.importance),
        backgroundColor: 'rgba(59, 130, 246, 0.5)',
        borderColor: 'rgba(59, 130, 246, 1)',
        borderWidth: 1,
      },
    ],
  }

  // ROC Curve Data
  const rocCurveData = metrics.roc_data
    ? {
        labels: [],
        datasets: Object.entries(metrics.roc_data).map(([label, data], idx) => ({
          label: `${label} (AUC: ${formatNumber(data.auc, 3)})`,
          data: data.fpr.map((fpr, i) => ({ x: fpr, y: data.tpr[i] })),
          borderColor: [
            'rgba(239, 68, 68, 1)',
            'rgba(234, 179, 8, 1)',
            'rgba(34, 197, 94, 1)',
          ][idx],
          backgroundColor: [
            'rgba(239, 68, 68, 0.1)',
            'rgba(234, 179, 8, 0.1)',
            'rgba(34, 197, 94, 0.1)',
          ][idx],
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.1,
        })),
      }
    : null

  const confusionMatrixOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: {
          color: 'rgb(203, 213, 225)',
        },
      },
      tooltip: {
        callbacks: {
          title: (context: any[]) => {
            const actualClass = context[0].label
            return `Actual: ${actualClass}`
          },
          label: (context: any) => {
            const predictedClass = context.dataset.label
            const count = context.parsed.y
            const actualClassIdx = context.dataIndex
            const total = actualTotals[actualClassIdx]
            const percentage = ((count / total) * 100).toFixed(1)
            const isCorrect = context.datasetIndex === actualClassIdx
            const status = isCorrect ? '✓ Correct' : '✗ Wrong'
            
            return `${status} - Predicted as ${predictedClass}: ${count}/${total} (${percentage}%)`
          },
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
  }

  const chartOptions = {
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
  }

  const rocChartOptions = {
    ...chartOptions,
    scales: {
      x: {
        type: 'linear' as const,
        title: { display: true, text: 'False Positive Rate', color: 'rgb(203, 213, 225)' },
        ticks: { color: 'rgb(148, 163, 184)' },
        grid: { color: 'rgba(148, 163, 184, 0.1)' },
      },
      y: {
        type: 'linear' as const,
        title: { display: true, text: 'True Positive Rate', color: 'rgb(203, 213, 225)' },
        ticks: { color: 'rgb(148, 163, 184)' },
        grid: { color: 'rgba(148, 163, 184, 0.1)' },
      },
    },
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold flex items-center gap-3">
          <BarChart3 className="w-8 h-8 text-primary-500" />
          Model Metrics
        </h1>
        <p className="text-slate-400 mt-2">Performance analysis and feature importance</p>
      </div>

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
        <div className="grid md:grid-cols-3 gap-4">
          <div>
            <p className="text-sm text-slate-400">Model Type</p>
            <p className="font-semibold">{metrics.model_info.model_type}</p>
          </div>
          <div>
            <p className="text-sm text-slate-400">Features</p>
            <p className="font-semibold">{metrics.model_info.n_features}</p>
          </div>
          <div>
            <p className="text-sm text-slate-400">Samples</p>
            <p className="font-semibold">{metrics.model_info.n_samples?.toLocaleString()}</p>
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Confusion Matrix */}
        <div className="card">
          <h3 className="text-xl font-semibold mb-4">Confusion Matrix</h3>
          <p className="text-sm text-slate-400 mb-4">Hover over bars to see counts and percentages</p>
          <div className="h-80">
            <Bar data={confusionMatrixData} options={confusionMatrixOptions} />
          </div>
        </div>

        {/* ROC Curve */}
        {rocCurveData && (
          <div className="card">
            <h3 className="text-xl font-semibold mb-4">ROC Curves</h3>
            <div className="h-80">
              <Line data={rocCurveData} options={rocChartOptions} />
            </div>
          </div>
        )}
      </div>

      {/* Feature Importance */}
      {topFeatures.length > 0 && (
        <div className="card">
          <h3 className="text-xl font-semibold mb-4">Top 15 Feature Importances</h3>
          <div className="h-96">
            <Bar
              data={featureImportanceData}
              options={{
                ...chartOptions,
                indexAxis: 'y' as const,
              }}
            />
          </div>
        </div>
      )}
    </div>
  )
}
