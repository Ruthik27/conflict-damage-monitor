import React from 'react'
import { Doughnut, Bar } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  ArcElement, Tooltip, Legend,
  CategoryScale, LinearScale, BarElement, Title,
} from 'chart.js'

ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement, Title)

const COLOURS = ['#64748b', '#22c55e', '#facc15', '#f97316', '#ef4444']

export default function DamageChart({ stats }) {
  if (!stats?.length) return null

  // Exclude background from donut; show all in bar
  const damage = stats.filter(s => s.class_id > 0)

  const donutData = {
    labels: damage.map(s => s.class_name),
    datasets: [{
      data: damage.map(s => s.pixel_pct),
      backgroundColor: damage.map(s => COLOURS[s.class_id]),
      borderWidth: 1,
      borderColor: '#1e293b',
    }],
  }

  const barData = {
    labels: stats.map(s => s.class_name),
    datasets: [{
      label: 'Pixel %',
      data: stats.map(s => s.pixel_pct),
      backgroundColor: stats.map(s => COLOURS[s.class_id]),
    }],
  }

  const chartOpts = {
    plugins: { legend: { labels: { color: '#94a3b8', font: { size: 11 } } } },
    maintainAspectRatio: false,
  }

  const barOpts = {
    ...chartOpts,
    scales: {
      x: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: '#1e293b' } },
      y: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' }, title: { display: true, text: '%', color: '#94a3b8' } },
    },
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      <h2 style={{ fontSize: '0.9rem', fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
        Damage Distribution
      </h2>

      <div style={{ height: 180 }}>
        <Doughnut data={donutData} options={chartOpts} />
      </div>

      <div style={{ height: 160 }}>
        <Bar data={barData} options={barOpts} />
      </div>

      {/* Stats table */}
      <table style={{ fontSize: '0.8rem', width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ color: '#94a3b8', borderBottom: '1px solid #334155' }}>
            <th style={{ textAlign: 'left', padding: '4px 0' }}>Class</th>
            <th style={{ textAlign: 'right' }}>Pixels</th>
            <th style={{ textAlign: 'right' }}>%</th>
          </tr>
        </thead>
        <tbody>
          {stats.map(s => (
            <tr key={s.class_id} style={{ borderBottom: '1px solid #1e293b' }}>
              <td style={{ padding: '3px 0', display: 'flex', alignItems: 'center', gap: 6 }}>
                <div style={{ width: 10, height: 10, borderRadius: 2, background: COLOURS[s.class_id], flexShrink: 0 }} />
                {s.class_name}
              </td>
              <td style={{ textAlign: 'right', color: '#94a3b8' }}>{s.pixel_count.toLocaleString()}</td>
              <td style={{ textAlign: 'right', color: '#94a3b8' }}>{s.pixel_pct.toFixed(1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
