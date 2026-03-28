import React, { useState } from 'react'
import axios from 'axios'

const API_BASE = '/api'

export default function PredictForm({ onResult, onLoading, onError }) {
  const [mode, setMode]           = useState('path')   // 'path' | 'upload'
  const [modelType, setModelType] = useState('segmentor')
  const [prePath, setPrePath]     = useState('')
  const [postPath, setPostPath]   = useState('')
  const [preFile, setPreFile]     = useState(null)
  const [postFile, setPostFile]   = useState(null)
  const [tileSize, setTileSize]   = useState(512)
  const [overlap, setOverlap]     = useState(128)

  const submit = async (e) => {
    e.preventDefault()
    onError(null)
    onLoading(true)
    onResult(null)

    try {
      let resp
      if (mode === 'path') {
        resp = await axios.post(`${API_BASE}/predict`, {
          pre_path: prePath,
          post_path: postPath,
          model_type: modelType,
          tile_size: tileSize,
          overlap: overlap,
          return_geojson: true,
        })
      } else {
        const fd = new FormData()
        fd.append('pre_file',   preFile)
        fd.append('post_file',  postFile)
        fd.append('model_type', modelType)
        fd.append('tile_size',  tileSize)
        fd.append('overlap',    overlap)
        resp = await axios.post(`${API_BASE}/predict/upload`, fd)
      }
      onResult(resp.data)
    } catch (err) {
      const msg = err.response?.data?.detail || err.message
      onError(String(msg))
    } finally {
      onLoading(false)
    }
  }

  return (
    <form className="form-card" onSubmit={submit}>
      <h2>Run Inference</h2>

      <div className="form-group">
        <label>Input Mode</label>
        <select value={mode} onChange={e => setMode(e.target.value)}>
          <option value="path">File Paths (server-side)</option>
          <option value="upload">Upload GeoTIFFs</option>
        </select>
      </div>

      <div className="form-group">
        <label>Model Type</label>
        <select value={modelType} onChange={e => setModelType(e.target.value)}>
          <option value="segmentor">UNet++ Segmentor</option>
          <option value="change_detector">Siamese Change Detector</option>
        </select>
      </div>

      {mode === 'path' ? (
        <>
          <div className="form-group">
            <label>Pre-disaster GeoTIFF path</label>
            <input
              type="text"
              placeholder="/blue/…/pre.tif"
              value={prePath}
              onChange={e => setPrePath(e.target.value)}
              required
            />
          </div>
          <div className="form-group">
            <label>Post-disaster GeoTIFF path</label>
            <input
              type="text"
              placeholder="/blue/…/post.tif"
              value={postPath}
              onChange={e => setPostPath(e.target.value)}
              required
            />
          </div>
        </>
      ) : (
        <>
          <div className="form-group">
            <label>Pre-disaster GeoTIFF</label>
            <input type="file" accept=".tif,.tiff" required
              onChange={e => setPreFile(e.target.files[0])} />
          </div>
          <div className="form-group">
            <label>Post-disaster GeoTIFF</label>
            <input type="file" accept=".tif,.tiff" required
              onChange={e => setPostFile(e.target.files[0])} />
          </div>
        </>
      )}

      <div className="form-group">
        <label>Tile Size</label>
        <select value={tileSize} onChange={e => setTileSize(Number(e.target.value))}>
          {[256, 512, 768, 1024].map(v => <option key={v} value={v}>{v}</option>)}
        </select>
      </div>

      <div className="form-group">
        <label>Overlap</label>
        <select value={overlap} onChange={e => setOverlap(Number(e.target.value))}>
          {[0, 64, 128, 256].map(v => <option key={v} value={v}>{v}</option>)}
        </select>
      </div>

      <button className="btn" type="submit">Run Inference</button>
    </form>
  )
}
