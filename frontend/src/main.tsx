import React, { Suspense } from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App.tsx'
import './index.css'
import { FullPageLoading } from './components/LoadingScreen'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Suspense fallback={<FullPageLoading message="Loading Exoplanet Classifier..." />}>
        <App />
      </Suspense>
    </BrowserRouter>
  </React.StrictMode>,
)
