---
name: geospatial-visualization
description: >
  Interactive geospatial map visualization for the conflict-damage-monitor dashboard.
  Use this skill whenever building or modifying any map component in dashboard/ —
  including damage overlays, building footprints, region views, or any React component
  that renders spatial data. Trigger on: "add a map", "show damage on the map",
  "visualize building polygons", "display GeoJSON", "add a legend", "color by damage
  class", "show the conflict zone", "overlay predictions", or any task involving
  React-Leaflet, Leaflet.js, or map tiles. The damage color scheme (green/yellow/
  orange/red) and required map furniture (legend, scale bar, region label) are
  non-negotiable — every map in this dashboard must have all three.
---

# Geospatial Visualization — conflict-damage-monitor Dashboard

The dashboard is a React + Leaflet.js application in `dashboard/`. All interactive maps
use `react-leaflet`. The primary purpose is communicating building damage assessments to
non-technical stakeholders, so color choices and map furniture matter as much as the data.

## Damage color scheme — fixed, never change

These four colors are the project's visual language. Use them consistently across maps,
legends, charts, and reports.

```js
// dashboard/src/constants/damageColors.js
export const DAMAGE_COLORS = {
  0: "#4CAF50",   // no-damage    → green
  1: "#FFEB3B",   // minor-damage → yellow
  2: "#FF9800",   // major-damage → orange
  3: "#F44336",   // destroyed    → red
};

export const DAMAGE_LABELS = {
  0: "No Damage",
  1: "Minor Damage",
  2: "Major Damage",
  3: "Destroyed",
};

// For GeoJSON styling functions
export const getDamageColor = (damageClass) =>
  DAMAGE_COLORS[damageClass] ?? "#9E9E9E";  // grey for unknown/null
```

Import from this file everywhere — never hardcode hex values inline.

---

## Standard map component

Every map in the dashboard includes: basemap tiles, GeoJSON damage overlay, legend,
scale bar, and region label. This is the baseline template.

```jsx
// dashboard/src/components/DamageMap.jsx
import React, { useMemo } from "react";
import {
  MapContainer,
  TileLayer,
  GeoJSON,
  ScaleControl,
  useMap,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { DAMAGE_COLORS, DAMAGE_LABELS, getDamageColor } from "../constants/damageColors";
import DamageLegend from "./DamageLegend";
import RegionLabel from "./RegionLabel";

const DEFAULT_CENTER = [20, 0];
const DEFAULT_ZOOM = 4;

/**
 * Interactive damage assessment map.
 *
 * @param {object} geojson - GeoJSON FeatureCollection with damage_class property
 * @param {string} regionName - Human-readable region label shown on map
 * @param {number[]} center - [lat, lng] map center
 * @param {number} zoom - Initial zoom level
 */
export default function DamageMap({
  geojson,
  regionName,
  center = DEFAULT_CENTER,
  zoom = DEFAULT_ZOOM,
}) {
  const styleFeature = (feature) => ({
    fillColor: getDamageColor(feature.properties.damage_class),
    fillOpacity: 0.75,
    color: "#333",
    weight: 0.8,
    opacity: 1,
  });

  const onEachFeature = (feature, layer) => {
    const { damage_class, building_id, confidence } = feature.properties;
    layer.bindTooltip(
      `<strong>${DAMAGE_LABELS[damage_class] ?? "Unknown"}</strong><br/>
       Building: ${building_id ?? "—"}<br/>
       Confidence: ${confidence != null ? `${(confidence * 100).toFixed(1)}%` : "—"}`,
      { sticky: true }
    );
  };

  return (
    <MapContainer
      center={center}
      zoom={zoom}
      style={{ height: "100%", width: "100%" }}
    >
      {/* Basemap — OpenStreetMap for context, switch to satellite when needed */}
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        maxZoom={19}
      />

      {/* Damage overlay */}
      {geojson && (
        <GeoJSON
          key={JSON.stringify(geojson)}  // force re-render on data change
          data={geojson}
          style={styleFeature}
          onEachFeature={onEachFeature}
        />
      )}

      {/* Required map furniture — all three must be present */}
      <ScaleControl position="bottomleft" imperial={false} />
      <DamageLegend />
      <RegionLabel name={regionName} />
    </MapContainer>
  );
}
```

---

## Legend component

The legend is a Leaflet control rendered as a React component. It always shows all four
damage classes even if some have zero buildings — this signals to users that the model
was looking for all classes, not that the data is incomplete.

```jsx
// dashboard/src/components/DamageLegend.jsx
import { useEffect } from "react";
import { useMap } from "react-leaflet";
import L from "leaflet";
import { DAMAGE_COLORS, DAMAGE_LABELS } from "../constants/damageColors";

export default function DamageLegend() {
  const map = useMap();

  useEffect(() => {
    const legend = L.control({ position: "bottomright" });

    legend.onAdd = () => {
      const div = L.DomUtil.create("div", "damage-legend");
      div.innerHTML = `
        <h4 style="margin:0 0 6px;font-size:13px;font-weight:600">Damage Class</h4>
        ${Object.entries(DAMAGE_LABELS)
          .map(([cls, label]) => `
            <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">
              <span style="
                display:inline-block;width:14px;height:14px;
                background:${DAMAGE_COLORS[cls]};
                border:1px solid #333;border-radius:2px;flex-shrink:0
              "></span>
              <span style="font-size:12px">${label}</span>
            </div>`)
          .join("")}
      `;
      return div;
    };

    legend.addTo(map);
    return () => legend.remove();
  }, [map]);

  return null;
}
```

Add CSS to `dashboard/src/index.css`:
```css
.damage-legend {
  background: white;
  padding: 10px 14px;
  border-radius: 6px;
  box-shadow: 0 1px 5px rgba(0,0,0,0.3);
  font-family: inherit;
  line-height: 1.4;
}
```

---

## Region label component

Shows the conflict zone or city name as a top-left overlay so screenshots and exports
are self-documenting.

```jsx
// dashboard/src/components/RegionLabel.jsx
import { useEffect } from "react";
import { useMap } from "react-leaflet";
import L from "leaflet";

export default function RegionLabel({ name }) {
  const map = useMap();

  useEffect(() => {
    if (!name) return;
    const label = L.control({ position: "topleft" });
    label.onAdd = () => {
      const div = L.DomUtil.create("div", "region-label");
      div.textContent = name;
      return div;
    };
    label.addTo(map);
    return () => label.remove();
  }, [map, name]);

  return null;
}
```

```css
/* dashboard/src/index.css */
.region-label {
  background: rgba(255,255,255,0.9);
  padding: 6px 12px;
  border-radius: 4px;
  font-size: 15px;
  font-weight: 600;
  box-shadow: 0 1px 5px rgba(0,0,0,0.25);
  pointer-events: none;
}
```

---

## GeoJSON format for damage overlays

The backend API (FastAPI) must return GeoJSON in this shape. The frontend expects
`damage_class` as an integer property.

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[lon, lat], ...]]
      },
      "properties": {
        "building_id": "xbd_hurricane-harvey_00000001",
        "damage_class": 3,
        "confidence": 0.87,
        "dataset": "xbd"
      }
    }
  ]
}
```

---

## Satellite basemap (for post-disaster imagery context)

Switch to Esri satellite when showing actual imagery alongside predictions:

```jsx
<TileLayer
  url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
  attribution="Tiles &copy; Esri"
  maxZoom={18}
/>
```

---

## Map usage checklist

Before shipping any new map component, verify:

- [ ] Uses `DAMAGE_COLORS` / `getDamageColor` from `damageColors.js` — no inline hex
- [ ] `<ScaleControl position="bottomleft" imperial={false} />` is present
- [ ] `<DamageLegend />` is present, showing all 4 classes
- [ ] `<RegionLabel name={regionName} />` is present with a meaningful label
- [ ] Tooltip on each building shows damage class label + building ID + confidence
- [ ] `key={JSON.stringify(geojson)}` on `<GeoJSON>` to force re-render on data updates
- [ ] CSS height set on MapContainer parent — Leaflet needs explicit height or map renders at 0px
