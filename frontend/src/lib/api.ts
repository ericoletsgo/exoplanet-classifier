const API_BASE_URL = import.meta.env.VITE_API_URL || (import.meta.env.DEV ? 'http://localhost:8000' : '/api')

// Debug logging
if (typeof window !== 'undefined') {
  console.log('API_BASE_URL:', API_BASE_URL)
  console.log('VITE_API_URL env var:', import.meta.env.VITE_API_URL)
}

export interface PredictionRequest {
  features: Record<string, number>
  model_id?: string
}

export interface PredictionResponse {
  prediction: string
  confidence: number
  probabilities: Record<string, number>
  prediction_class: number
}

export interface BatchPredictionRequest {
  records: Record<string, number>[]
}

export interface BatchPredictionResponse {
  predictions: PredictionResponse[]
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

export interface DatasetColumnsResponse {
  dataset: string
  total_rows: number
  total_columns: number
  columns: Array<{
    name: string
    type: string
    non_null_count: number
    null_count: number
    sample_values?: string[]
    unique_count?: number
  }>
}

export interface ModelEvaluationResponse {
  model_id: string
  model_info: {
    name: string
    description: string
    created_at: string
    algorithms: string[]
    n_features: number
  }
  metrics: {
    train_accuracy: number
    test_accuracy: number
    precision: number
    recall: number
    f1_score: number
  }
  confusion_matrix: number[][]
  dataset_info: {
    train_samples: number
    test_samples: number
  }
  features: string[]
}

class APIClient {
  private async request<T>(endpoint: string, options?: RequestInit & { timeout?: number }): Promise<T> {
    const { timeout = 15000, ...fetchOptions } = options || {}
    
    const controller = new AbortController()
    const timeoutId = timeout ? setTimeout(() => controller.abort(), timeout) : null
    
    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        ...fetchOptions,
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          ...fetchOptions?.headers,
        },
      })

      if (timeoutId) clearTimeout(timeoutId)

      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
        throw new Error(error.detail || `HTTP ${response.status}`)
      }

      return response.json()
    } catch (error) {
      if (timeoutId) clearTimeout(timeoutId)
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error('Request timeout')
      }
      throw error
    }
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

  async batchPredict(data: BatchPredictionRequest) {
    return this.request<BatchPredictionResponse>('/batch-predict', {
      method: 'POST',
      body: JSON.stringify(data),
    })
  }

  async getMetrics() {
    return this.request<MetricsResponse>('/metrics', { timeout: 30000 }) // 30 second timeout for heavy operation
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

  async getDatasetColumns(datasetName: string) {
    return this.request<DatasetColumnsResponse>(`/datasets/${datasetName}/columns`, { timeout: 20000 }) // 20 second timeout for columns
  }

  async evaluateModel(modelId: string) {
    return this.request<ModelEvaluationResponse>(`/models/${modelId}/evaluate`)
  }

  async listModels() {
    return this.request<{ models: any[] }>('/models', { timeout: 30000 }) // 30 second timeout for models
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


  async getFeatureCorrelations() {
    return this.request<{
      features: string[]
      matrix: number[][]
      sample_size: number
      total_features: number
    }>('/feature-correlations')
  }

  async trainModel(data: { 
    dataset: string
    model_name: string
    description: string
    test_size?: number
    algorithms?: string[]
    hyperparameters?: {
      // Gradient Boosting
      gb_n_estimators?: number
      gb_learning_rate?: number
      gb_max_depth?: number
      gb_min_samples_split?: number
      // Random Forest
      rf_n_estimators?: number
      rf_max_depth?: number
      rf_min_samples_split?: number
      rf_max_features?: string
      // XGBoost
      xgb_n_estimators?: number
      xgb_learning_rate?: number
      xgb_max_depth?: number
      xgb_subsample?: number
      // LightGBM
      lgb_n_estimators?: number
      lgb_learning_rate?: number
      lgb_max_depth?: number
      lgb_num_leaves?: number
    }
    use_hyperparameter_tuning?: boolean
    include_k2?: boolean
    include_toi?: boolean
    target_column?: string
    target_mapping?: Record<string, number>
    csv_data?: string
  }) {
    return this.request<{
      status: string
      message: string
      model_id?: string
      metrics?: Record<string, any>
      algorithms_used?: string[]
      cv_accuracy?: number
      dataset_summary?: Record<string, any>
    }>('/train', {
      method: 'POST',
      body: JSON.stringify(data),
      timeout: 120000 // 2 minute timeout for training
    })
  }

  async getAvailableAlgorithms() {
    return this.request<{
      algorithms: Record<string, boolean>
      available_count: number
      total_count: number
    }>('/algorithms', { timeout: 20000 }) // 20 second timeout for algorithms
  }
}

export const api = new APIClient()
