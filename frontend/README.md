# Exoplanet Classifier Frontend

Modern React + TypeScript + Tailwind CSS frontend for the Exoplanet Classifier ML system.

## Features

- **Home Dashboard**: Overview of system status and model information
- **Prediction Interface**: Interactive form to classify exoplanet candidates
- **Metrics Visualization**: Charts showing model performance, ROC curves, and feature importance
- **Dataset Explorer**: Browse and filter KOI, K2, and TOI datasets with pagination

## Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Styling
- **React Router** - Navigation
- **Chart.js** - Data visualization
- **Lucide React** - Icons

## Prerequisites

- Node.js 18+ and npm
- Backend API running on `http://localhost:8000`

## Installation

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install
```

## Development

```bash
# Start dev server (runs on http://localhost:5173)
npm run dev
```

The dev server includes a proxy that forwards `/api/*` requests to `http://localhost:8000`.

## Build

```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

## Project Structure

```
frontend/
├── src/
│   ├── lib/
│   │   ├── api.ts          # API client and types
│   │   └── utils.ts        # Utility functions
│   ├── pages/
│   │   ├── HomePage.tsx    # Dashboard
│   │   ├── PredictPage.tsx # Prediction interface
│   │   ├── MetricsPage.tsx # Model metrics
│   │   └── DatasetsPage.tsx # Dataset browser
│   ├── App.tsx             # Main app with routing
│   ├── main.tsx            # Entry point
│   └── index.css           # Global styles
├── index.html
├── package.json
├── vite.config.ts
├── tailwind.config.js
└── tsconfig.json
```

## API Integration

The frontend communicates with the FastAPI backend through the following endpoints:

- `GET /` - Health check
- `GET /features` - Get feature list
- `POST /predict` - Make predictions
- `GET /metrics` - Get model metrics
- `GET /datasets/{name}` - Get dataset with pagination

## Customization

### Colors

Edit `tailwind.config.js` to customize the color scheme:

```js
theme: {
  extend: {
    colors: {
      primary: { ... }
    }
  }
}
```

### API URL

Update `vite.config.ts` proxy settings if your backend runs on a different port:

```ts
proxy: {
  '/api': {
    target: 'http://localhost:YOUR_PORT',
    ...
  }
}
```

## License

MIT
