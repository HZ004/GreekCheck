import React, { useEffect, useState, useMemo } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts'
import Papa from 'papaparse'
import DatePicker from 'react-datepicker'
import 'react-datepicker/dist/react-datepicker.css'

// Environment variable for S3 CSV URL (set in Render)
const S3_CSV_URL = import.meta.env.VITE_S3_CSV_URL

const greekOrder = ['ltp', 'delta', 'gamma', 'theta']
const greekKeys = ['delta', 'gamma', 'theta', 'ltp'] // original used for aggregation
const intervals = [
  { label: '5 seconds', value: 5 },
  { label: '15 seconds', value: 15 },
  { label: '30 seconds', value: 30 },
  { label: '1 minute', value: 60 },
  { label: '2 minutes', value: 120 },
  { label: '5 minutes', value: 300 }
]

// Helper to generate consistent color by baseKey (strike/instrument)
function stringToColor(str) {
  let hash = 0
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash)
    hash = hash & hash
  }
  let color = '#'
  for (let i = 0; i < 3; i++) {
    const value = (hash >> (i * 8)) & 0xff
    color += ('00' + value.toString(16)).substr(-2)
  }
  return color
}

// Custom legend formatter to remove suffixes
function legendFormatter(value) {
  return value.replace(/_(delta|gamma|theta|ltp)$/, '')
}

// For synchronized Y-axis domain calculation
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
  // Add some padding
  const padding = (max - min) * 0.1 || 0.1
  return [min - padding, max + padding]
}

function getLinesForKeys(keys) {
  return keys.map(key => {
    const baseKey = key.replace(/_(delta|gamma|theta|ltp)$/, '')
    return (
      <Line
        key={key}
        type="monotone"
        dataKey={key}
        dot={false}
        stroke={stringToColor(baseKey)}
        strokeWidth={2}
        name={baseKey}
      />
    )
  })
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

function roundValue(value, greek) {
  if (value === null || value === undefined || isNaN(value)) return value
  if (greek === 'gamma') return Number(value.toFixed(3))
  if (['ltp', 'delta', 'theta'].includes(greek)) return Number(value.toFixed(2))
  return value
}

function App() {
  const [data, setData] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [startDate, setStartDate] = useState(null)
  const [endDate, setEndDate] = useState(null)
  const [interval, setInterval] = useState(5) // default 5 sec

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
        // Round value according to greek suffix
        const greekSuffixMatch = k.match(/_(delta|gamma|theta|ltp)$/)
        const greek = greekSuffixMatch ? greekSuffixMatch[1] : null
        newRow[k] = greek ? roundValue(avgVal, greek) : avgVal
      })
      resultRows.push(newRow)
    }
    resultRows.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp))

    // Apply moving average smoothing on each key except timestamp
    greekKeys.forEach(greek => {
      resultRows.forEach((row, idx) => {
        Object.keys(row).forEach(k => {
          if (k.endsWith(`_${greek}`)) {
            let window = Math.max(1, Math.floor(interval / 5))
            let maValues = movingAverage(resultRows, k, window)
            resultRows.forEach((r, index) => {
              // Round after averaging
              r[k] = roundValue(maValues[index], greek)
            })
          }
        })
      })
    })

    return resultRows
  }, [data, startDate, endDate, interval])

  if (loading) return <div className="loading">Loading data...</div>
  if (error) return <div className="error">Error loading data: {error}</div>
  if (filteredAggregatedData.length === 0) return <div className="no-data">No data for selected range</div>

  const dataKeys = Object.keys(filteredAggregatedData[0]).filter(key => key !== 'timestamp')

  // Group keys by greek type and CE / PE separation
  const keysByGreekAndType = {}
  greekOrder.forEach(greek => {
    keysByGreekAndType[greek] = {
      CE: dataKeys.filter(k => k.endsWith(`_${greek}`) && k.startsWith('CE_')),
      PE: dataKeys.filter(k => k.endsWith(`_${greek}`) && k.startsWith('PE_'))
    }
  })

  // Calculate Y-axis domain for CE and PE combined for each greek
  const yDomains = {}
  greekOrder.forEach(greek => {
    const combinedKeys = [...keysByGreekAndType[greek].CE, ...keysByGreekAndType[greek].PE]
    yDomains[greek] = getYAxisDomain(filteredAggregatedData, combinedKeys)
  })

  return (
    <div className="app-container">
      <h1>Upstox Option Greeks Dashboard</h1>
      <div className="controls">
        <label>
          Start Date:
          <DatePicker
            selected={startDate}
            onChange={date => setStartDate(date)}
            selectsStart
            startDate={startDate}
            endDate={endDate}
            maxDate={endDate || new Date()}
            isClearable
            placeholderText="Select start date"
          />
        </label>
        <label>
          End Date:
          <DatePicker
            selected={endDate}
            onChange={date => setEndDate(date)}
            selectsEnd
            startDate={startDate}
            endDate={endDate}
            minDate={startDate}
            maxDate={new Date()}
            isClearable
            placeholderText="Select end date"
          />
        </label>
        <label>
          Interval:
          <select value={interval} onChange={e => setInterval(Number(e.target.value))}>
            {intervals.map(opt => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gridTemplateRows: 'repeat(4, 320px)',
          gap: '20px',
          marginTop: '20px'
        }}
      >
        {greekOrder.map(greek => (
          <React.Fragment key={greek}>
            {/* CE plot left */}
            <section className="chart-section" style={{ border: '1px solid #ccc', padding: '10px' }}>
              <h2>CE {greek.toUpperCase()} Over Time</h2>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart
                  data={filteredAggregatedData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <XAxis
                    dataKey="timestamp"
                    tickFormatter={str => new Date(str).toLocaleTimeString()}
                    minTickGap={20}
                  />
                  <YAxis domain={yDomains[greek]} />
                  <CartesianGrid strokeDasharray="3 3" />
                  <Tooltip labelFormatter={label => new Date(label).toLocaleString()} />
                  <Legend formatter={legendFormatter} wrapperStyle={{ fontSize: '0.8em' }} />
                  {getLinesForKeys(keysByGreekAndType[greek].CE)}
                </LineChart>
              </ResponsiveContainer>
            </section>

            {/* PE plot right */}
            <section className="chart-section" style={{ border: '1px solid #ccc', padding: '10px' }}>
              <h2>PE {greek.toUpperCase()} Over Time</h2>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart
                  data={filteredAggregatedData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <XAxis
                    dataKey="timestamp"
                    tickFormatter={str => new Date(str).toLocaleTimeString()}
                    minTickGap={20}
                  />
                  <YAxis domain={yDomains[greek]} />
                  <CartesianGrid strokeDasharray="3 3" />
                  <Tooltip labelFormatter={label => new Date(label).toLocaleString()} />
                  <Legend formatter={legendFormatter} wrapperStyle={{ fontSize: '0.8em' }} />
                  {getLinesForKeys(keysByGreekAndType[greek].PE)}
                </LineChart>
              </ResponsiveContainer>
            </section>
          </React.Fragment>
        ))}
      </div>
    </div>
  )
}

export default App
