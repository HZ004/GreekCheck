import React from 'react'
import {
LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts'


const GreekChart = ({ title, data, dataKeys, yDomain, hiddenLines, toggleLine, colorMap, legendFormatter, customTooltip, height }) => {
return (
<section className="card">
<h2>{title}</h2>
<div style={{ width: '100%', height }}>
<ResponsiveContainer width="100%" height="100%">
<LineChart data={data} margin={{ top: 6, right: 18, left: 6, bottom: 6 }}>
<XAxis dataKey="timestamp" tickFormatter={str => new Date(str).toLocaleTimeString()} minTickGap={40} tick={{ fill: '#cbd5e1', fontSize: 12 }} />
<YAxis domain={yDomain} tick={{ fill: '#cbd5e1', fontSize: 12 }} />
<CartesianGrid strokeDasharray="3 3" stroke="#0f172a" />
<Tooltip content={customTooltip} />
<Legend formatter={legendFormatter} wrapperStyle={{ fontSize: '0.85rem', userSelect: 'none' }} />
{dataKeys.map(key => {
const baseKey = key.replace(/_(delta|gamma|theta|ltp)$/, '')
const color = colorMap[baseKey]
const isHidden = hiddenLines.has(key)
return (
<Line key={key} type="monotone" dataKey={key} dot={false} stroke={color} strokeWidth={2.25} name={baseKey} hide={isHidden} onClick={() => toggleLine(key)} style={{ cursor: 'pointer' }} />
)
})}
</LineChart>
</ResponsiveContainer>
</div>
</section>
)
}
export default GreekChart
