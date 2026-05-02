import { useEffect } from 'react'
import { CircleMarker, MapContainer, Marker, Popup, TileLayer, useMap, useMapEvents } from 'react-leaflet'
import type { LatLngExpression } from 'leaflet'
import L from 'leaflet'

import type { LocationSite } from '../api/client'

const SALISH_CENTER: LatLngExpression = [48.8, -123.0]
const SALISH_BOUNDS: [[number, number], [number, number]] = [[47.0, -125.5], [50.0, -121.0]]

const selectedIcon = L.divIcon({
  className: 'star-browser-map-pin',
  html: '<div style="width:14px;height:14px;border-radius:999px;background:#2563eb;border:2px solid white;box-shadow:0 0 0 1px #1d4ed8;"></div>',
  iconSize: [14, 14],
  iconAnchor: [7, 7],
})

function MapFocusController({ latitude, longitude }: { latitude?: number; longitude?: number }) {
  const map = useMap()
  useEffect(() => {
    if (latitude != null && longitude != null) {
      map.setView([latitude, longitude], Math.max(map.getZoom(), 10))
    }
  }, [latitude, longitude, map])
  return null
}

function PickingHandler({ enabled, onPick }: { enabled: boolean; onPick: (lat: number, lon: number) => void }) {
  useMapEvents({
    click(event) {
      if (!enabled) return
      onPick(event.latlng.lat, event.latlng.lng)
    },
  })
  return null
}

export function LocationSiteMap({
  sites,
  selectedLatitude,
  selectedLongitude,
  picking,
  onPick,
  onSelectSite,
}: {
  sites: LocationSite[]
  selectedLatitude?: number
  selectedLongitude?: number
  picking: boolean
  onPick: (lat: number, lon: number) => void
  onSelectSite?: (site: LocationSite) => void
}) {
  return (
    <div>
      <MapContainer
        center={SALISH_CENTER}
        zoom={8}
        scrollWheelZoom
        style={{ width: '100%', height: 280, border: '1px solid #d7deea', borderRadius: 10 }}
        bounds={SALISH_BOUNDS}
      >
        <TileLayer
          attribution='&copy; OpenStreetMap contributors'
          url='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'
        />
        <MapFocusController latitude={selectedLatitude} longitude={selectedLongitude} />
        <PickingHandler enabled={picking} onPick={onPick} />
        {sites.map((site) => (
          <CircleMarker
            key={`${site.name}-${site.latitude}-${site.longitude}`}
            center={[site.latitude, site.longitude]}
            radius={6}
            pathOptions={{ color: '#0f766e', fillColor: '#14b8a6', fillOpacity: 0.85 }}
            eventHandlers={onSelectSite ? { click: () => onSelectSite(site) } : undefined}
          >
            <Popup>
              <button type="button" onClick={() => onSelectSite?.(site)}>{site.name}</button>
            </Popup>
          </CircleMarker>
        ))}
        {selectedLatitude != null && selectedLongitude != null && (
          <Marker position={[selectedLatitude, selectedLongitude]} icon={selectedIcon}>
            <Popup>Selected coordinates</Popup>
          </Marker>
        )}
      </MapContainer>
    </div>
  )
}
