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


const colorPalette = ['#22c1c3', '#60a5fa', '#7c3aed', '#fb7185', '#34d399', '#f59e0b', '#60a5fa', '#f97316']
const baseKeyColorMap = {}
let colorIndex = 0
function getColorForKey(key) {
if (!(key in baseKeyColorMap)) {
baseKeyColorMap[key] = colorPalette[colorIndex % colorPalette.length]
colorIndex++
}
return baseKeyColorMap[key]
}
function legendFormatter(value) { return value.replace(/_(delta|gamma|theta|ltp)$/, '') }
function roundValue(value, greek) {
if (value === null || value === undefined || isNaN(value)) return value
if (greek === 'gamma') return Number(value.toFixed(3))
return Number(value.toFixed(2))
}
if(key==='timestamp')return;const valid=vals.filter(v=>typeof v==='number'&&!isNaN(v));const avg=valid.length?valid.reduce((a,b)=>a+b,0)/valid.length:null;const g=key.match(/_(delta|gamma|theta|ltp)$/)?.[1];newRow[key]=g?roundVa
