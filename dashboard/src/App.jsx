import React, { useState } from 'react'
import DamageMap from './components/DamageMap'
import DamageChart from './components/DamageChart'
import PredictForm from './components/PredictForm'
import StatusBar from './components/StatusBar'
import './App.css'

export default function App() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  return (
    <div className="app">
      <header className="app-header">
        <h1>Conflict Damage Monitor</h1>
        <StatusBar />
      </header>

      <main className="app-body">
        <aside className="sidebar">
          <PredictForm
            onResult={setResult}
            onLoading={setLoading}
            onError={setError}
          />
          {error && <div className="error-box">{error}</div>}
          {result && <DamageChart stats={result.class_stats} />}
        </aside>

        <section className="map-panel">
          <DamageMap result={result} loading={loading} />
        </section>
      </main>
    </div>
  )
}
