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
  '#0072B2', '#E69F00', '#F0E442', '#D55E00',
  '#CC79A7', '#56B4E9', '#009E73', '#999999'
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
  let extra = delta < 2 ? 2 : delta * 0.12
  const expandedMin = Math.floor((min - extra) * 100) / 100
  const expandedMax = Math.ceil((max + extra) * 100) / 100
  const greek = keys.length > 0 ? (keys[0].match(/_(delta|gamma|theta|ltp)$/) || [null, null])[1] : null
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
    <div className="custom-tooltip" style={{ backgroundColor: 'white', padding: 10, borderRadius: 6, boxShadow: '0 2px 5px rgba(0,0,0,0.15)', border: '1px solid #ddd' }}>
      <p><strong>{new Date(label).toLocaleString()}</strong></p>
      {sortedPayload.map((entry, index) => (
        <p key={`item-${index}`} style={{ color: entry.color, margin: 0, fontWeight: '500' }}>
          {entry.name}: {Number(entry.value).toFixed(entry.dataKey.endsWith('_gamma') ? 3 : 2)}
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

    greekKeys.forEach(greek => {
      resultRows.forEach((row, idx) => {
        Object.keys(row).forEach(k => {
          if (k.endsWith(`_${greek}`)) {
            let window = Math.max(1, Math.floor(interval / 5))
            let maValues = movingAverage(resultRows, k, window)
            resultRows.forEach((r, index) => {
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

  // Prepare color mapping for all base keys (all unique instruments/strikes)
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

  const availableHeight = windowHeight - 250
  const chartHeight = Math.max(240, availableHeight / 4 - 30)

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

      <div className="charts-grid" style={{ maxHeight: availableHeight + 100, overflowY: 'auto' }}>
        {greekOrder.map(greek => (
          <React.Fragment key={greek}>
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
              height={chartHeight - 50}
            />
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
              height={chartHeight - 50}
            />
          </React.Fragment>
        ))}
      </div>
    </div>
  )
}

export default App
