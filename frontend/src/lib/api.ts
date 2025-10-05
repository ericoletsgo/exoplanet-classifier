const API_BASE_URL = import.meta.env.VITE_API_URL || '/api'

export interface PredictionRequest {
  features: Record<string, number>
}

export interface PredictionResponse {
  prediction: string
  confidence: number
  probabilities: Record<string, number>
  prediction_class: number
}

export interface MetricsResponse {
  accuracy: number
  precision: number
  recall: number
  f1_score: number
  confusion_matrix: number[][]
  roc_data?: Record<string, { fpr: number[]; tpr: number[]; auc: number }>
  feature_importances?: Array<{ feature: string; importance: number }>
  model_info: Record<string, any>
}

export interface DatasetResponse {
  total_rows: number
  columns: string[]
  data: Record<string, any>[]
  page: number
  page_size: number
  total_pages: number
}

export interface FeatureCategory {
  signal_quality: string[]
  flux_centroid: string[]
  orbital_params: string[]
  stellar_params: string[]
  error_params: string[]
  [key: string]: string[] // Allow dynamic access
}

export interface FeaturesResponse {
  features: FeatureCategory
  labels: Record<string, string>
  descriptions: Record<string, string>
  total_features: number
  categories: string[]
}

class APIClient {
  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
      throw new Error(error.detail || `HTTP ${response.status}`)
    }

    return response.json()
  }

  async healthCheck() {
    return this.request<{ status: string; service: string; version: string }>('/')
  }

  async getFeatures() {
    return this.request<FeaturesResponse>('/features')
  }

  async predict(data: PredictionRequest) {
    return this.request<PredictionResponse>('/predict', {
      method: 'POST',
      body: JSON.stringify(data),
    })
  }

  async predictRaw(rawRowData: Record<string, any>) {
    return this.request<PredictionResponse>('/predict-raw', {
      method: 'POST',
      body: JSON.stringify(rawRowData),
    })
  }

  async getMetrics() {
    return this.request<MetricsResponse>('/metrics')
  }

  async getDataset(
    datasetName: string,
    page: number = 1,
    pageSize: number = 50,
    filterDisposition?: string
  ) {
    const params = new URLSearchParams({
      page: page.toString(),
      page_size: pageSize.toString(),
    })
    if (filterDisposition) {
      params.append('filter_disposition', filterDisposition)
    }
    return this.request<DatasetResponse>(`/datasets/${datasetName}?${params}`)
  }

  async listModels() {
    return this.request<{ models: any[] }>('/models')
  }

  async getRandomExample(datasetName: string, disposition?: string) {
    const params = new URLSearchParams()
    if (disposition) {
      params.append('disposition', disposition)
    }
    const queryString = params.toString()
    const url = `/random-example/${datasetName}${queryString ? `?${queryString}` : ''}`
    return this.request<{
      features: Record<string, number>
      metadata: {
        row_index: number
        koi_name: string
        expected_disposition: string
        dataset: string
      }
      raw_row: Record<string, any>
    }>(url)
  }
}

export const api = new APIClient()
