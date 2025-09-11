import React, { useEffect, useState, useMemo } from 'react'
import Papa from 'papaparse'
import DatePicker from 'react-datepicker'
import 'react-datepicker/dist/react-datepicker.css'
import './App.css'
import './index.css'
import GreekChart from './components/GreekChart'

const S3_CSV_URL = import.meta.env.VITE_S3_CSV_URL

const greekOrder = ['ltp', 'delta', 'gamma', 'theta']
const greekKeys = ['delta', 'gamma', 'theta', 'ltp']
const intervals = [
  { label: '5 seconds', value: 5 },
  { label: '15 seconds', value: 15 },
  { label: '30 seconds', value: 30 },
  { label: '1 minute', value: 60 },
  { label: '2 minutes', value: 120 },
  { label: '5 minutes', value: 300 }
]

const colorPalette = [
  '#22c1c3', '#60a5fa', '#7c3aed', '#fb7185',
  '#34d399', '#f59e0b', '#60a5fa', '#f97316'
]

const baseKeyColorMap = {}
let colorIndex = 0
function getColorForKey(key) {
  if (!(key in baseKeyColorMap)) {
    baseKeyColorMap[key] = colorPalette[colorIndex % colorPalette.length]
    colorIndex++
  }
  return baseKeyColorMap[key]
}

function legendFormatter(value) {
  return value.replace(/_(delta|gamma|theta|ltp)$/, '')
}

function roundValue(value, greek) {
  if (value === null || value === undefined || isNaN(value)) return value
  if (greek === 'gamma') return Number(value.toFixed(3))
  if (['ltp', 'delta', 'theta'].includes(greek)) return Number(value.toFixed(2))
  return value
}

function roundAxisDomain(min, max, greek) {
  if (greek === 'gamma') {
    return [
      Math.floor(min * 1000) / 1000,
      Math.ceil(max * 1000) / 1000
    ]
  } else {
    return [
      Math.floor(min * 100) / 100,
      Math.ceil(max * 100) / 100
    ]
  }
}

function getYAxisDomain(data, keys) {
  let min = Infinity
  let max = -Infinity
  data.forEach(row => {
    keys.forEach(key => {
      const val = row[key]
      if (typeof val === 'number' && !isNaN(val)) {
        if (val < min) min = val
        if (val > max) max = val
      }
    })
  })
  if (min === Infinity || max === -Infinity) {
    min = 0
    max = 1
  }

  let delta = max - min
  const greek = keys.length > 0 ? (keys[0].match(/_(delta|gamma|theta|ltp)$/) || [null, null])[1] : null
  let extra = 0
  if (greek === 'gamma') {
    extra = Math.max(0.01, delta * 0.15)
  } else if (greek === 'delta') {
    extra = Math.max(0.05, delta * 0.2)
  } else {
    extra = delta < 2 ? 2 : delta * 0.12
  }
  let expandedMin = Math.floor((min - extra) * 100) / 100
  let expandedMax = Math.ceil((max + extra) * 100) / 100

  if (greek !== 'theta' && expandedMin < 0) {
    expandedMin = 0
  }

  return roundAxisDomain(expandedMin, expandedMax, greek)
}

function movingAverage(data, key, windowSize) {
  const result = []
  for (let i = 0; i < data.length; i++) {
    let start = Math.max(0, i - windowSize + 1)
    let subset = data.slice(start, i + 1)
    let sum = 0
    let count = 0
    for (let j = 0; j < subset.length; j++) {
      const val = subset[j][key]
      if (typeof val === 'number' && !isNaN(val)) {
        sum += val
        count++
      }
    }
    result.push(count > 0 ? sum / count : null)
  }
  return result
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload || payload.length === 0) return null
  const sortedPayload = payload.slice().sort((a, b) => (b.value ?? 0) - (a.value ?? 0))
  return (
    <div className="custom-tooltip">
      <p style={{ margin: '0 0 6px 0' }}><strong>{new Date(label).toLocaleString()}</strong></p>
      {sortedPayload.map((entry, index) => (
        <p key={`item-${index}`} style={{ color: entry.color, margin: 0, fontWeight: '600' }}>
          {entry.name}: {Number(entry.value || 0).toFixed(entry.dataKey.endsWith('_gamma') ? 3 : 2)}
        </p>
      ))}
    </div>
  )
}

function App() {
  const [data, setData] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [startDate, setStartDate] = useState(null)
  const [endDate, setEndDate] = useState(null)
  const [interval, setInterval] = useState(5)
  const [windowHeight, setWindowHeight] = useState(window.innerHeight)
  const [hiddenLines, setHiddenLines] = useState(new Set())

  useEffect(() => {
    const handleResize = ()
