import React from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts'

const GreekChart = ({
  title,
  data,
  dataKeys,
  yDomain,
  hiddenLines,
  toggleLine,
  colorMap,
  legendFormatter,
  customTooltip,
  height
}) => {
  return (
    <section className="chart-section">
      <h2>{title}</h2>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <XAxis
            dataKey="timestamp"
            tickFormatter={str => new Date(str).toLocaleTimeString()}
            minTickGap={20}
            tick={{ fill: '#333' }}
          />
          <YAxis domain={yDomain} tick={{ fill: '#333' }} />
          <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
          <Tooltip content={customTooltip} />
          <Legend formatter={legendFormatter} wrapperStyle={{ fontSize: '0.8em' }} />
          {dataKeys.map(key => {
            const baseKey = key.replace(/_(delta|gamma|theta|ltp)$/, '')
            const color = colorMap[baseKey]
            const isHidden = hiddenLines.has(key)
            return (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                dot={false}
                stroke={color}
                strokeWidth={2}
                name={baseKey}
                hide={isHidden}
                onClick={() => toggleLine(key)}
                style={{ cursor: 'pointer' }}
              />
            )
          })}
        </LineChart>
      </ResponsiveContainer>
    </section>
  )
}

export default GreekChart
