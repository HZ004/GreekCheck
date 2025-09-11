import React, { useEffect, useState, useMemo } from 'react'
import Papa from 'papaparse'
import DatePicker from 'react-datepicker'
import 'react-datepicker/dist/react-datepicker.css'
import './App.css'
import GreekChart from './components/GreekChart'

const S3_CSV_URL = import.meta.env.VITE_S3_CSV_URL

// same greekOrder, greekKeys, colorPalette etc. (maybe refine palette below)

const primaryColor = '#4A90E2'         // a soft, confident blue accent
const secondaryColor = '#2C3E50'       // dark slate for headers/text
const bgLight = '#F5F7FA'              // very light grey background
const cardBg = '#FFFFFF'
const textPrimary = '#2C3E50'
const textSecondary = '#5A6C7F'
const borderColor = '#E1E8EE'
const hoverShadow = '0 8px 24px rgba(0,0,0,0.12)'
const normalShadow = '0 4px 12px rgba(0,0,0,0.08)'

/* your existing helper functions: roundValue, getYAxisDomain, movingAverage etc. 
   (unchanged except maybe chart styling) */

function App() {
  // same state, data fetching as before ...

  // after filteredAggregatedData is computed

  const availableHeight = windowHeight - 300
  const chartHeight = Math.max(280, availableHeight / 4)

  return (
    <div
      className="app-container"
      style={{
        fontFamily: "'Inter', sans-serif",
        backgroundColor: bgLight,
        color: textPrimary,
        minHeight: '100vh',
        padding: '0',
        margin: '0',
      }}
    >
      <header
        className="header"
        style={{
          position: 'sticky',
          top: 0,
          zIndex: 100,
          backgroundColor: cardBg,
          padding: '24px 32px',
          boxShadow: normalShadow,
          borderBottom: `1px solid ${borderColor}`,
        }}
      >
        <h1 style={{ fontSize: '2.5rem', fontWeight: 600, margin: 0 }}>
          Upstox Option Greeks Dashboard
        </h1>
        <div className="controls" style={{
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
              className="date-picker"
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
              className="date-picker"
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

      <main
        className="charts-grid"
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))',
          gap: '60px 30px',
          padding: '40px 32px',
          alignItems: 'start',
          overflowY: 'auto',
          boxSizing: 'border-box',
        }}
      >
        {greekOrder.map(greek => (
          <React.Fragment key={greek}>
            <div
              style={{
                backgroundColor: cardBg,
                borderRadius: '12px',
                boxShadow: normalShadow,
                padding: '24px',
                minHeight: `${chartHeight + 50}px`,
                maxWidth: '95%',
                width: '100%',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
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
              <div style={{ width: '100%' }}>
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
                  // chart styling props below:
                  lineStyles={{
                    strokeWidth: 2,
                    curve: 'monotone',        // smoother line
                  }}
                  gridStyles={{
                    stroke: borderColor,
                    strokeDasharray: '4 4'
                  }}
                  axisStyles={{
                    tickColor: textSecondary,
                    labelColor: textSecondary,
                  }}
                />
              </div>
            </div>

            <div
              style={{
                backgroundColor: cardBg,
                borderRadius: '12px',
                boxShadow: normalShadow,
                padding: '24px',
                minHeight: `${chartHeight + 50}px`,
                maxWidth: '95%',
                width: '100%',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
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
              <div style={{ width: '100%' }}>
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
                  lineStyles={{
                    strokeWidth: 2,
                    curve: 'monotone',
                  }}
                  gridStyles={{
                    stroke: borderColor,
                    strokeDasharray: '4 4'
                  }}
                  axisStyles={{
                    tickColor: textSecondary,
                    labelColor: textSecondary,
                  }}
                />
              </div>
            </div>
          </React.Fragment>
        ))}
      </main>
    </div>
  )
}

export default App

