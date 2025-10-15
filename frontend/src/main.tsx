import React, { Suspense } from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'
import { FullPageLoading } from './components/LoadingScreen'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <Suspense fallback={<FullPageLoading message="Loading Exoplanet Classifier..." />}>
      <App />
    </Suspense>
  </React.StrictMode>,
)
