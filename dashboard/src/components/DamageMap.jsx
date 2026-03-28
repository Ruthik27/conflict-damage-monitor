import React, { useEffect, useRef } from 'react'
import { MapContainer, TileLayer, GeoJSON, useMap } from 'react-leaflet'
import L from 'leaflet'

// Damage class → colour
const CLASS_COLOURS = {
  'background':   '#64748b',
  'no-damage':    '#22c55e',
  'minor-damage': '#facc15',
  'major-damage': '#f97316',
  'destroyed':    '#ef4444',
}

function GeoJSONLayer({ geojson }) {
  const map = useMap()

  const style = (feature) => ({
    color: CLASS_COLOURS[feature.properties.class_name] || '#94a3b8',
    weight: 1,
    fillOpacity: 0.5,
  })

  const onEach = (feature, layer) => {
    const p = feature.properties
    layer.bindTooltip(`${p.class_name} (class ${p.class_id})`, { sticky: true })
  }

  useEffect(() => {
    if (!geojson?.features?.length) return
    // Fit map to GeoJSON bounds
    const layer = L.geoJSON(geojson)
    const bounds = layer.getBounds()
    if (bounds.isValid()) map.fitBounds(bounds, { padding: [20, 20] })
  }, [geojson, map])

  if (!geojson?.features?.length) return null
  return <GeoJSON key={JSON.stringify(geojson)} data={geojson} style={style} onEachFeature={onEach} />
}

export default function DamageMap({ result, loading }) {
  return (
    <div style={{ position: 'relative', height: '100%', width: '100%' }}>
      <MapContainer
        center={[20, 0]}
        zoom={2}
        style={{ height: '100%', width: '100%', background: '#1e293b' }}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        />
        {result?.geojson && <GeoJSONLayer geojson={result.geojson} />}
      </MapContainer>

      {loading && (
        <div className="loading-overlay">Running inference…</div>
      )}

      {/* Legend */}
      <div style={{
        position: 'absolute', bottom: 24, right: 12, zIndex: 1000,
        background: 'rgba(15,23,42,0.85)', border: '1px solid #334155',
        borderRadius: 8, padding: '0.6rem 0.9rem', fontSize: '0.78rem',
      }}>
        {Object.entries(CLASS_COLOURS).map(([name, colour]) => (
          <div key={name} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3 }}>
            <div style={{ width: 12, height: 12, borderRadius: 2, background: colour }} />
            <span style={{ color: '#e2e8f0' }}>{name}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
