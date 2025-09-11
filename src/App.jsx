import React, { useEffect, useState, useMemo } from 'react'
import Papa from 'papaparse'
import DatePicker from 'react-datepicker'
import 'react-datepicker/dist/react-datepicker.css'
import './App.css'
import GreekChart from './components/GreekChart'

// 🔑 Configs
const S3_CSV_URL = import.meta.env.VITE_S3_CSV_URL
const greekOrder = ['ltp', 'delta', 'gamma', 'theta']
const intervals = [
  { value: 1, label: '1 Minute' },
  { value: 5, label: '5 Minutes' },
  { value: 15, label: '15 Minutes' },
  { value: 30, label: '30 Minutes' },
  { value: 60, label: '1 Hour' },
]

// 🔑 Color Palette
const primaryColor = '#4A90E2'
const secondaryColor = '#2C3E50'
const bgLight = '#F5F7FA'
const cardBg = '#FFFFFF'
const textPrimary = '#2C3E50'
const textSecondary = '#5A6C7F'
const borderColor = '#E1E8EE'
const hoverShadow = '0 8px 24px rgba(0,0,0,0.12)'
const normalShadow = '0 4px 12px rgba(0,0,0,0.08)'

// 🔑 Utility Functions
function roundValue(num) {
  if (num === null || num === undefined || isNaN(num)) return null
  if (Math.abs(num) >= 1) return Math.round(num * 100) / 100
  return Math.round(num * 10000) / 10000
}

function movingAverage(data, key, period = 5) {
  // ✅ Fixed Moving Average: simple rolling window
  if (!data || data.length === 0) return []
  const values = data.map(d => d[key] ?? null)
  const result = values.map((val, i) => {
    if (i < period - 1 || val === null) return null
    let sum = 0
    let count = 0
    for (let j = 0; j < period; j++) {
      const v = values[i - j]
      if (v !== null) {
        sum += v
        count++
      }
    }
    return count === period ? sum / count : null
  })
  return result
}

function getYAxisDomain(data, keys) {
  let min = Infinity
  let max = -Infinity
  data.forEach(row => {
    keys.forEach(key => {
      const value = row[key]
      if (typeof value === 'number') {
        min = Math.min(min, value)
        max = Math.max(max, value)
      }
    })
  })
  if (!isFinite(min) || !isFinite(max)) return [0, 1]
  const padding = (max - min) * 0.1 || 1
  return [min - padding, max + padding]
}

// 🔑 Custom Tooltip
const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload || payload.length === 0) return null
  return (
    <div style={{
      background: '#fff',
      padding: '10px 12px',
      border: `1px solid ${borderColor}`,
      borderRadius: '8px',
      boxShadow: normalShadow,
    }}>
      <p style={{ fontWeight: 600, marginBottom: '6px', color: textPrimary }}>
        {label}
      </p>
      {payload.map((p, i) => (
        <p key={i} style={{ margin: 0, color: p.color }}>
          {p.name}: <strong>{roundValue(p.value)}</strong>
        </p>
      ))}
    </div>
  )
}

function App() {
  const [csvData, setCsvData] = useState([])
  const [filteredData, setFilteredData] = useState([])
  const [startDate, setStartDate] = useState(null)
  const [endDate, setEndDate] = useState(null)
  const [interval, setInterval] = useState(5)
  const [windowHeight, setWindowHeight] = useState(window.innerHeight)
  const [hiddenLines, setHiddenLines] = useState({})

  // Resize listener
  useEffect(() => {
    const handleResize = () => setWindowHeight(window.innerHeight)
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  // Load CSV
  useEffect(() => {
    if (!S3_CSV_URL) return
    Papa.parse(S3_CSV_URL, {
      download: true,
      header: true,
      dynamicTyping: true,
      complete: result => {
        const rows = result.data
          .filter(r => r.timestamp)
          .map(r => ({ ...r, timestamp: new Date(r.timestamp) }))
          .sort((a, b) => a.timestamp - b.timestamp)
        setCsvData(rows)
        setFilteredData(rows)
      },
    })
  }, [])

  // Filter by date range
  useEffect(() => {
    if (!csvData.length) return
    let data = csvData
    if (startDate) data = data.filter(d => d.timestamp >= startDate)
    if (endDate) data = data.filter(d => d.timestamp <= endDate)
    setFilteredData(data)
  }, [startDate, endDate, csvData])

  const aggregatedData = useMemo(() => {
    if (!filteredData.length) return []
    const bucketed = []
    let bucket = []
    let bucketStart = filteredData[0].timestamp

    filteredData.forEach(row => {
      const diff = (row.timestamp - bucketStart) / 60000
      if (diff >= interval) {
        const avg = bucket.reduce((acc, cur) => {
          Object.keys(cur).forEach(k => {
            if (typeof cur[k] === 'number') {
              acc[k] = (acc[k] || 0) + cur[k]
            }
          })
          return acc
        }, {})
        Object.keys(avg).forEach(k => avg[k] /= bucket.length)
        avg.timestamp = bucketStart
        bucketed.push(avg)
        bucket = []
        bucketStart = row.timestamp
      }
      bucket.push(row)
    })
    return bucketed
  }, [filteredData, interval])

  const yDomains = useMemo(() => {
    const domains = {}
    greekOrder.forEach(greek => {
      const keys = [
        `ce_${greek}`, `pe_${greek}`,
        `ce_${greek}_ma`, `pe_${greek}_ma`
      ]
      domains[greek] = getYAxisDomain(aggregatedData, keys)
    })
    return domains
  }, [aggregatedData])

  const availableHeight = windowHeight - 300
  const chartHeight = Math.max(280, availableHeight / 4)

  const toggleLine = key => {
    setHiddenLines(prev => ({ ...prev, [key]: !prev[key] }))
  }

  const keysByGreekAndType = {
    ltp: { CE: ['ce_ltp', 'ce_ltp_ma'], PE: ['pe_ltp', 'pe_ltp_ma'] },
    delta: { CE: ['ce_delta', 'ce_delta_ma'], PE: ['pe_delta', 'pe_delta_ma'] },
    gamma: { CE: ['ce_gamma', 'ce_gamma_ma'], PE: ['pe_gamma', 'pe_gamma_ma'] },
    theta: { CE: ['ce_theta', 'ce_theta_ma'], PE: ['pe_theta', 'pe_theta_ma'] },
  }

  return (
    <div style={{
      fontFamily: "'Inter', sans-serif",
      backgroundColor: bgLight,
      color: textPrimary,
      minHeight: '100vh',
    }}>
      {/* HEADER */}
      <header style={{
        position: 'sticky',
        top: 0,
        zIndex: 100,
        backgroundColor: cardBg,
        padding: '24px 32px',
        boxShadow: normalShadow,
        borderBottom: `1px solid ${borderColor}`,
      }}>
        <h1 style={{ fontSize: '2.5rem', fontWeight: 600, margin: 0 }}>
          Option Greeks Dashboard
        </h1>
        <div style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '16px',
          alignItems: 'center',
          marginTop: '16px',
        }}>
          <label style={{ fontWeight: 500, color: textSecondary }}>Start Date:
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
          <label style={{ fontWeight: 500, color: textSecondary }}>End Date:
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
          <label style={{ fontWeight: 500, color: textSecondary }}>Interval:
            <select
              value={interval}
              onChange={e => setInterval(Number(e.target.value))}
              style={{
                marginLeft: '8px',
                padding: '6px 10px',
                borderRadius: '4px',
                border: `1px solid ${borderColor}`,
                backgroundColor: cardBg,
                color: textPrimary,
                fontSize: '1rem'
              }}
            >
              {intervals.map(opt => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </label>
        </div>
      </header>

      {/* CHARTS GRID */}
      <main style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))',
        gap: '60px 30px',
        padding: '40px 32px',
        alignItems: 'start',
        overflowY: 'auto',
      }}>
        {greekOrder.map(greek => (
          <React.Fragment key={greek}>
            {/* CE Card */}
            <ChartCard>
              <GreekChart
                title={`CE ${greek.toUpperCase()} Over Time`}
                data={aggregatedData}
                dataKeys={keysByGreekAndType[greek].CE}
                yDomain={yDomains[greek]}
                hiddenLines={hiddenLines}
                toggleLine={toggleLine}
                customTooltip={<CustomTooltip />}
                height={chartHeight}
              />
            </ChartCard>

            {/* PE Card */}
            <ChartCard>
              <GreekChart
                title={`PE ${greek.toUpperCase()} Over Time`}
                data={aggregatedData}
                dataKeys={keysByGreekAndType[greek].PE}
                yDomain={yDomains[greek]}
                hiddenLines={hiddenLines}
                toggleLine={toggleLine}
                customTooltip={<CustomTooltip />}
                height={chartHeight}
              />
            </ChartCard>
          </React.Fragment>
        ))}
      </main>
    </div>
  )
}

// Extracted ChartCard wrapper
function ChartCard({ children }) {
  return (
    <div style={{
      backgroundColor: cardBg,
      borderRadius: '12px',
      boxShadow: normalShadow,
      padding: '24px',
      minHeight: '320px',
      maxWidth: '95%',
      width: '100%',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      border: `1px solid ${borderColor}`,
      transition: 'transform 0.2s ease, box-shadow 0.2s ease',
    }}
      onMouseEnter={e => {
        e.currentTarget.style.transform = 'translateY(-4px)'
        e.currentTarget.style.boxShadow = hoverShadow
      }}
      onMouseLeave={e => {
        e.currentTarget.style.transform = 'translateY(0)'
        e.currentTarget.style.boxShadow = normalShadow
      }}
    >
      <div style={{ width: '100%' }}>{children}</div>
    </div>
  )
}

export default App
