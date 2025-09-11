import React, { useEffect, useState, useMemo } from 'react'
import Papa from 'papaparse'
import DatePicker from 'react-datepicker'
import 'react-datepicker/dist/react-datepicker.css'
import './App.css'
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
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
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

  // Enforce min=0 for all Greeks except Theta
  if (greek !== 'theta' && expandedMin < 0) {
    expandedMin = 0
  }

  return roundAxisDomain(expandedMin, expandedMax, greek)
}

/**
 * Correct sliding-window trailing moving average.
 * - windowSize is the number of points to include (>=1).
 * - ignores null/undefined/NaN values (they don't count toward average).
 * - returns an array of same length with numbers or null when no valid values in window.
 */
function movingAverage(data, key, windowSize) {
  const n = data.length
  const result = new Array(n).fill(null)
  if (n === 0) return result
  if (windowSize <= 1) {
    for (let i = 0; i < n; i++) {
      const v = data[i][key]
      result[i] = (typeof v === 'number' && !isNaN(v)) ? v : null
    }
    return result
  }

  let sum = 0
  let count = 0
  const queue = []

  for (let i = 0; i < n; i++) {
    const raw = data[i][key]
    const v = (typeof raw === 'number' && !isNaN(raw)) ? raw : null
    queue.push(v)
    if (v !== null) {
      sum += v
      count++
    }

    if (queue.length > windowSize) {
      const removed = queue.shift()
      if (removed !== null) {
        sum -= removed
        count--
      }
    }

    result[i] = count > 0 ? sum / count : null
  }

  return result
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload || payload.length === 0) return null
  const sortedPayload = payload.slice().sort((a, b) => (b.value ?? 0) - (a.value ?? 0))
  return (
    <div className="custom-tooltip">
      <p><strong>{new Date(label).toLocaleString()}</strong></p>
      {sortedPayload.map((entry, index) => (
        <p key={`item-${index}`} style={{ color: entry.color, margin: 0, fontWeight: '600' }}>
          {entry.name}: {Number(entry.value).toFixed(entry.dataKey && entry.dataKey.endsWith('_gamma') ? 3 : 2)}
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
    const handleResize = () => setWindowHeight(window.innerHeight)
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  useEffect(() => {
    async function fetchAndParse() {
      if (!S3_CSV_URL) {
        setError("S3 CSV URL is not set in environment variables")
        setLoading(false)
        return
      }
      try {
        const response = await fetch(S3_CSV_URL)
        if (!response.ok) throw new Error(`HTTP error: ${response.status}`)
        const csvText = await response.text()
        const parsed = Papa.parse(csvText, { header: true, dynamicTyping: true })
        setData(parsed.data.filter(row => row.timestamp))
        setLoading(false)
      } catch (e) {
        setError(e.message)
        setLoading(false)
      }
    }
    fetchAndParse()
  }, [])

  const filteredAggregatedData = useMemo(() => {
    if (data.length === 0) return []

    let filtered = data
    if (startDate) filtered = filtered.filter(d => new Date(d.timestamp) >= startDate)
    if (endDate) filtered = filtered.filter(d => new Date(d.timestamp) <= endDate)

    // bucket by interval
    let buckets = new Map()
    filtered.forEach(row => {
      let dt = new Date(row.timestamp)
      let seconds = dt.getSeconds()
      let minutes = dt.getMinutes()
      let hours = dt.getHours()
      let totalSeconds = hours * 3600 + minutes * 60 + seconds
      let bucketStartSec = Math.floor(totalSeconds / interval) * interval
      let bucketDate = new Date(dt)
      bucketDate.setHours(0, 0, 0, 0)
      bucketDate = new Date(bucketDate.getTime() + bucketStartSec * 1000)
      let bucketKey = bucketDate.toISOString()
      if (!buckets.has(bucketKey)) {
        let newRow = { timestamp: bucketKey }
        Object.keys(row).forEach(k => {
          if (k !== 'timestamp') {
            newRow[k] = [row[k]]
          }
        })
        buckets.set(bucketKey, newRow)
      } else {
        let existingRow = buckets.get(bucketKey)
        Object.keys(row).forEach(k => {
          if (k !== 'timestamp') {
            existingRow[k].push(row[k])
          }
        })
      }
    })

    let resultRows = []
    for (let [bucketKey, valuesObj] of buckets) {
      let newRow = { timestamp: bucketKey }
      Object.entries(valuesObj).forEach(([k, vals]) => {
        if (k === 'timestamp') return
        let validVals = vals.filter(v => v !== null && v !== undefined && !isNaN(v))
        let avgVal = validVals.length > 0 ? validVals.reduce((a, b) => a + b, 0) / validVals.length : null
        const greekMatch = k.match(/_(delta|gamma|theta|ltp)$/)
        const greek = greekMatch ? greekMatch[1] : null
        newRow[k] = greek ? roundValue(avgVal, greek) : avgVal
      })
      resultRows.push(newRow)
    }
    resultRows.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp))

    // Compute moving average once per series/key (efficient & correct)
    const keysToSmooth = new Set()
    resultRows.forEach(row => {
      Object.keys(row).forEach(k => {
        if (k !== 'timestamp') {
          greekKeys.forEach(g => {
            if (k.endsWith(`_${g}`)) keysToSmooth.add(k)
          })
        }
      })
    })

    const window = Math.max(1, Math.floor(interval / 5))
    keysToSmooth.forEach(k => {
      const greekMatch = k.match(/_(delta|gamma|theta|ltp)$/)
      const greek = greekMatch ? greekMatch[1] : null
      const maValues = movingAverage(resultRows, k, window)
      for (let i = 0; i < resultRows.length; i++) {
        resultRows[i][k] = roundValue(maValues[i], greek)
      }
    })

    return resultRows
  }, [data, startDate, endDate, interval])

  if (loading) return <div className="loading">Loading data...</div>
  if (error) return <div className="error">Error loading data: {error}</div>
  if (filteredAggregatedData.length === 0) return <div className="no-data">No data for selected range</div>

  const dataKeys = Object.keys(filteredAggregatedData[0]).filter(key => key !== 'timestamp')

  const allBaseKeys = Array.from(new Set(dataKeys.map(k => k.replace(/_(delta|gamma|theta|ltp)$/, ''))))
  const colorMap = {}
  allBaseKeys.forEach(key => {
    colorMap[key] = getColorForKey(key)
  })

  const keysByGreekAndType = {}
  greekOrder.forEach(greek => {
    keysByGreekAndType[greek] = {
      CE: dataKeys.filter(k => k.endsWith(`_${greek}`) && k.startsWith('CE_')),
      PE: dataKeys.filter(k => k.endsWith(`_${greek}`) && k.startsWith('PE_'))
    }
  })

  const yDomains = {}
  greekOrder.forEach(greek => {
    const combinedKeys = [...keysByGreekAndType[greek].CE, ...keysByGreekAndType[greek].PE]
    yDomains[greek] = getYAxisDomain(filteredAggregatedData, combinedKeys)
  })

  function toggleLine(key) {
    setHiddenLines(prev => {
      const next = new Set(prev)
      if (next.has(key)) {
        next.delete(key)
      } else {
        next.add(key)
      }
      return next
    })
  }

  const availableHeight = windowHeight - 300
  const chartHeight = Math.max(260, availableHeight / 4)

  return (
    <div className="app-container" style={{ fontFamily: 'Inter, sans-serif', background: '#f9fafb', minHeight: '100vh' }}>
      <header className="header" style={{
        position: 'sticky', top: 0, zIndex: 100,
        background: '#fff', padding: '16px 24px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.05)', borderBottom: '1px solid #e5e7eb'
      }}>
        <h1 style={{ fontSize: '1.8rem', fontWeight: '700', marginBottom: '12px' }}>Upstox Option Greeks Dashboard</h1>
        <div className="controls" style={{ display: 'flex', flexWrap: 'wrap', gap: '12px', alignItems: 'center' }}>
          <label>Start Date:
            <DatePicker
              selected={startDate}
              onChange={setStartDate}
              selectsStart
              startDate={startDate}
              endDate={endDate}
              maxDate={endDate || new Date()}
              isClearable
              placeholderText="Select start date"
            />
          </label>
          <label>End Date:
            <DatePicker
              selected={endDate}
              onChange={setEndDate}
              selectsEnd
              startDate={startDate}
              endDate={endDate}
              minDate={startDate}
              maxDate={new Date()}
              isClearable
              placeholderText="Select end date"
            />
          </label>
          <label>Interval:
            <select value={interval} onChange={e => setInterval(Number(e.target.value))}>
              {intervals.map(opt => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </label>
        </div>
      </header>
      
      <main
        className="charts-grid"
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))', 
          gap: '60px 30px',       // Big vertical/horizontal spacing
          padding: '40px 30px',   // Add breathing room around all charts
          alignItems: 'center',    // Prevent stretching that causes overlap
          overflowY: 'auto',
          boxSizing: 'border-box'
        }}
      >

        {greekOrder.map(greek => (
          <React.Fragment key={greek}>
            <div
              style={{
                background: '#fff',
                borderRadius: '16px',
                boxShadow: '0 6px 18px rgba(0, 0, 0, 0.08)',
                padding: '20px',
                minHeight: `${chartHeight + 40}px`,
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                boxSizing: 'border-box',
                transition: 'transform 0.2s ease, box-shadow 0.2s ease',
              }}
              onMouseEnter={e => {
                e.currentTarget.style.transform = 'translateY(-4px)';
                e.currentTarget.style.boxShadow = '0 8px 24px rgba(0,0,0,0.12)';
              }}
              onMouseLeave={e => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 6px 18px rgba(0,0,0,0.08)';
              }}
            >
              <GreekChart
                title={`CE ${greek.toUpperCase()} Over Time`}
                data={filteredAggregatedData}
                dataKeys={keysByGreekAndType[greek].CE}
                yDomain={yDomains[greek]}
                hiddenLines={hiddenLines}
                toggleLine={toggleLine}
                colorMap={colorMap}
                legendFormatter={legendFormatter}
                customTooltip={<CustomTooltip />}
                height={chartHeight}
              />
            </div>
        
            <div
              style={{
                background: '#fff',
                borderRadius: '16px',
                boxShadow: '0 6px 18px rgba(0, 0, 0, 0.08)',
                padding: '20px',
                minHeight: `${chartHeight + 40}px`,
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                boxSizing: 'border-box',
                transition: 'transform 0.2s ease, box-shadow 0.2s ease',
              }}
              onMouseEnter={e => {
                e.currentTarget.style.transform = 'translateY(-4px)';
                e.currentTarget.style.boxShadow = '0 8px 24px rgba(0,0,0,0.12)';
              }}
              onMouseLeave={e => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 6px 18px rgba(0,0,0,0.08)';
              }}
            >
              <GreekChart
                title={`PE ${greek.toUpperCase()} Over Time`}
                data={filteredAggregatedData}
                dataKeys={keysByGreekAndType[greek].PE}
                yDomain={yDomains[greek]}
                hiddenLines={hiddenLines}
                toggleLine={toggleLine}
                colorMap={colorMap}
                legendFormatter={legendFormatter}
                customTooltip={<CustomTooltip />}
                height={chartHeight}
              />
            </div>
          </React.Fragment>
        ))}
      </main>
    </div>
  )
}

export default App
