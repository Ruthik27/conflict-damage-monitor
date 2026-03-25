# dashboard/

Web dashboard for visualizing building damage across conflict regions.

## Structure

```
dashboard/
├── frontend/   # React + React-Leaflet (interactive map UI)
└── backend/    # FastAPI (serves GeoJSON tiles and stats)
```

## Features (v1)

- Basemap: Sentinel-2 cloudless or OpenStreetMap tiles
- Damage overlay: color-coded by severity (green/yellow/orange/red)
- Region selector: Gaza, Southern Lebanon (more in v2)
- Time slider: step through pre/post dates
- Stats panel: counts of buildings by damage class, trend chart

## Features (v2)

- ACLED events overlay (correlate strikes with damage)
- Additional regions: Iran, Israel, other countries
- Export damage reports as PDF or CSV

## Stack

- **Frontend**: React 18, React-Leaflet 4, Recharts
- **Backend**: FastAPI, uvicorn
- **Data**: GeoJSON/GeoTIFF from inference pipeline
