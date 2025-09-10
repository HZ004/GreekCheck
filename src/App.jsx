import React, { useEffect, useState, useMemo } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts'
import Papa from 'papaparse'
import DatePicker from 'react-datepicker'
import 'react-datepicker/dist/react-datepicker.css'
import { addSeconds, startOfMinute, differenceInSeconds } from 'date-fns'

// Replace with your actual public S3 URL or presigned URL to the CSV file
const S3_CSV_URL = 'https://your-s3-bucket.s3.amazonaws.com/upstox-greeks/greeks_data_all.csv'

const greekKeys = ['delta', 'gamma', 'theta', 'ltp']

const intervals = [
  { label: '5 seconds', value: 5 },
  { label: '15 seconds', value: 15 },
  { label: '30 seconds', value: 30 },
  { label: '1 minute', value: 60 },
  { label: '2 minutes', value: 120 },
  { label: '5 minutes', value: 300 }
]

// Helper to extract columns for a specific greek and plot lines for each strike
function getLinesForGreek(dataKeys, greek) {
  return dataKeys
    .filter(k => k.endsWith(`_${greek}`))
    .map(key => {
      return (
        <Line
          key={key}
          type="monotone"
          dataKey={key}
          dot={false}
          stroke={`#${stringToColor(key)}`}
          strokeWidth={2}
        />
      )
    })
}

// String hash to color function
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

// Moving average smoothing function for numeric values based on window size
function movingAverage(data, key, windowSize) {
  const result = []
  for (let i = 0; i < data.length; i++) {
    let start = Math.max(0, i - windowSize + 1)
    let subset = data.slice(start, i + 1)
    let sum = 0
    let count = 0
    for (let j = 0; j < subset.length; j++) {
      let val = subset[j][key]
      if (typeof val === 'number' && !isNaN(val)) {
        sum += val
        count++
      }
    }
    result.push(count > 0 ? sum / count : null)
  }
  return result
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

  // Filter by date range and aggregate by interval with moving average smoothing
  const filteredAggregatedData = useMemo(() => {
    if (data.length === 0) return []

    let filtered = data

    // Filter by date range
    if (startDate) filtered = filtered.filter(d => new Date(d.timestamp) >= startDate)
    if (endDate) filtered = filtered.filter(d => new Date(d.timestamp) <= endDate)

    // Aggregate by interval seconds (sample and average)
    // Group timestamps by nearest interval bucket
    let buckets = new Map()
    filtered.forEach(row => {
      let dt = new Date(row.timestamp)
      let seconds = dt.getSeconds()
      let minutes = dt.getMinutes()
      let hours = dt.getHours()
      
      // Calculate total seconds since start of day
      let totalSeconds = hours * 3600 + minutes * 60 + seconds
      // Calculate bucket start by flooring totalSeconds to nearest multiple of interval
      let bucketStartSec = Math.floor(totalSeconds / interval) * interval
      // Construct bucket timestamp
      let bucketDate = new Date(dt)
      bucketDate.setHours(0,0,0,0)
      bucketDate = new Date(bucketDate.getTime() + bucketStartSec * 1000)
      let bucketKey = bucketDate.toISOString()

      if (!buckets.has(bucketKey)) {
        let newRow = { timestamp: bucketKey }
        // Initialize keys with arrays for averaging
        Object.keys(row).forEach(k => {
          if(k !== 'timestamp'){
            newRow[k] = [row[k]]
          }
        })
        buckets.set(bucketKey, newRow)
      } else {
        let existingRow = buckets.get(bucketKey)
        Object.keys(row).forEach(k => {
          if(k !== 'timestamp'){
            existingRow[k].push(row[k])
          }
        })
      }
    })

    // Build averaged rows with moving average smoothing
    let resultRows = []
    for (let [bucketKey, valuesObj] of buckets) {
      let newRow = { timestamp: bucketKey }
      Object.entries(valuesObj).forEach(([k, vals]) => {
        if (k === 'timestamp') return
        // Average ignoring null/NaN
        let validVals = vals.filter(v => v !== null && v !== undefined && !isNaN(v))
        let avgVal = validVals.length > 0 ? validVals.reduce((a,b) => a + b, 0) / validVals.length : null
        newRow[k] = avgVal
      })
      resultRows.push(newRow)
    }

    // Sort result by timestamp ascending
    resultRows.sort((a,b) => new Date(a.timestamp) - new Date(b.timestamp))

    // Apply moving average smoothing on each key except timestamp
    resultRows.forEach((row, idx) => {
      greekKeys.forEach(greek => {
        Object.keys(row).forEach(k => {
          if (k.endsWith(`_${greek}`)) {
            // Create an array for the key across all rows for moving average
            let window = Math.max(1, Math.floor(interval / 5)) // window size in points
            let maValues = movingAverage(resultRows, k, window)
            resultRows.forEach((r, index) => { r[k] = maValues[index] })
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

      {greekKeys.map(greek => (
        <section key={greek} className="chart-section">
          <h2>{greek.toUpperCase()} Over Time</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart
              data={filteredAggregatedData}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <XAxis
                dataKey="timestamp"
                tickFormatter={str => new Date(str).toLocaleTimeString()}
                minTickGap={20}
              />
              <YAxis />
              <CartesianGrid strokeDasharray="3 3" />
              <Tooltip labelFormatter={label => new Date(label).toLocaleString()} />
              <Legend />
              {getLinesForGreek(dataKeys, greek)}
            </LineChart>
          </ResponsiveContainer>
        </section>
      ))}
    </div>
  )
}

export default App
