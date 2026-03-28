import React, { useEffect, useState } from 'react'
import axios from 'axios'

const API_BASE = '/api'

export default function StatusBar() {
  const [status, setStatus] = useState(null)   // null = loading

  useEffect(() => {
    const check = () =>
      axios.get(`${API_BASE}/health`)
        .then(r => setStatus(r.data))
        .catch(() => setStatus({ status: 'error', model_loaded: false, device: 'unknown' }))

    check()
    const id = setInterval(check, 30_000)
    return () => clearInterval(id)
  }, [])

  if (!status) return <div className="status-bar"><div className="status-dot loading" />Connecting…</div>

  const ok = status.status === 'ok'
  return (
    <div className="status-bar">
      <div className={`status-dot ${ok ? 'ok' : 'err'}`} />
      <span style={{ color: ok ? '#22c55e' : '#ef4444' }}>
        API {ok ? 'online' : 'offline'}
      </span>
      {ok && (
        <>
          <span style={{ color: '#475569' }}>|</span>
          <span style={{ color: '#94a3b8' }}>
            {status.model_loaded ? 'model ready' : 'no checkpoint'}
          </span>
          <span style={{ color: '#475569' }}>|</span>
          <span style={{ color: '#94a3b8' }}>{status.device}</span>
        </>
      )}
    </div>
  )
}
