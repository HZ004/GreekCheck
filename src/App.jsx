import React, { useState, useMemo, useEffect } from "react";
import Plot from "react-plotly.js";

// Complete end-to-end App.jsx
// - 4 rows x 2 columns grid (LTP, Delta, Gamma, Theta)
// - Fixed moving average function (simple moving average)
// - Gamma chart autoscaling/clipping to 1st-99th percentiles to avoid huge Y axis
// - Title input, date range selection, and time selection controls
// - Mock data generator but accepts optional `data` prop if you want to feed real data

// NOTE: This file expects the following packages:
// npm install react-plotly.js plotly.js
// Tailwind classes are used for styling (optional) — you can replace with plain CSS.

// ---------- Utility functions ----------

function computeSMA(arr, window) {
  // Robust simple moving average. Returns an array of same length with nulls for indexes < window-1.
  if (!Array.isArray(arr) || window <= 1) return arr.map((v) => (isFinite(v) ? v : null));
  const out = new Array(arr.length).fill(null);
  let sum = 0;
  let count = 0;
  // Use sliding window
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (Number.isFinite(v)) {
      sum += v;
      count += 1;
    }
    if (i >= window) {
      const rem = arr[i - window];
      if (Number.isFinite(rem)) {
        sum -= rem;
        count -= 1;
      }
    }
    if (i >= window - 1) {
      // average only of the numeric values within full window
      out[i] = count > 0 ? sum / count : null;
    }
  }
  return out;
}

function diff(arr) {
  const out = new Array(arr.length).fill(null);
  for (let i = 1; i < arr.length; i++) {
    const a = arr[i];
    const b = arr[i - 1];
    out[i] = Number.isFinite(a) && Number.isFinite(b) ? a - b : null;
  }
  return out;
}

function secondDiff(arr) {
  // second derivative approximation (gamma)
  const d1 = diff(arr);
  return diff(d1);
}

function percentile(arr, p) {
  const vals = arr.filter(Number.isFinite).slice().sort((a, b) => a - b);
  if (vals.length === 0) return null;
  const idx = (p / 100) * (vals.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return vals[lo];
  return vals[lo] * (hi - idx) + vals[hi] * (idx - lo);
}

// ---------- Mock data generator (for demo) ----------
function generateMockSeries(points = 600, start = new Date()) {
  // Generate a minute-by-minute series with a gentle random walk
  const times = [];
  const ltp = [];
  let price = 1000 + Math.random() * 20;
  for (let i = 0; i < points; i++) {
    const t = new Date(start.getTime() + i * 60 * 1000);
    times.push(t.toISOString());
    // random walk with occasional spikes
    const step = (Math.random() - 0.5) * 2;
    if (Math.random() < 0.005) price += (Math.random() - 0.5) * 200; // spike
    price = Math.max(1, price + step);
    ltp.push(Number(price.toFixed(4)));
  }
  return { times, ltp };
}

// ---------- Main App ----------

export default function App({ data: incomingData = null }) {
  // Controls
  const [title, setTitle] = useState("Market Dashboard");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [startTime, setStartTime] = useState("");
  const [endTime, setEndTime] = useState("");
  const [maWindow, setMaWindow] = useState(20);
  const [points] = useState(600);

  // Data: either use incomingData prop (expected shape { times:[], ltp:[] }) or generate mock
  const raw = useMemo(() => {
    if (incomingData && incomingData.times && incomingData.ltp) return incomingData;
    return generateMockSeries(points, new Date(Date.now() - points * 60 * 1000));
  }, [incomingData, points]);

  // Derived indicators
  const indicators = useMemo(() => {
    const { times, ltp } = raw;
    const sma = computeSMA(ltp, Math.max(1, Math.round(maWindow)));
    const delta = diff(ltp);
    const gamma = secondDiff(ltp);
    // Theta: simple negative time decay proxy (just a synthetic series)
    const theta = ltp.map((v, i) => {
      // approximate theta as negative slope per minute scaled
      if (i === 0 || !Number.isFinite(v) || !Number.isFinite(ltp[i - 1])) return null;
      return -1 * (v - ltp[i - 1]);
    });
    return { times, ltp, sma, delta, gamma, theta };
  }, [raw, maWindow]);

  // Date/time filtering
  const filtered = useMemo(() => {
    const { times, ltp, sma, delta, gamma, theta } = indicators;
    const out = { times: [], ltp: [], sma: [], delta: [], gamma: [], theta: [] };
    for (let i = 0; i < times.length; i++) {
      const t = new Date(times[i]);
      if (startDate) {
        const sd = new Date(startDate + (startTime ? `T${startTime}:00` : "T00:00:00"));
        if (t < sd) continue;
      }
      if (endDate) {
        const ed = new Date(endDate + (endTime ? `T${endTime}:00` : "T23:59:59"));
        if (t > ed) continue;
      }
      if (startTime || endTime) {
        // if only times are provided, apply them on the day of the data point
        if (startTime) {
          const [h, m] = startTime.split(":").map(Number);
          const tt = t.getHours() * 60 + t.getMinutes();
          const st = h * 60 + m;
          if (tt < st) continue;
        }
        if (endTime) {
          const [h, m] = endTime.split(":").map(Number);
          const tt = t.getHours() * 60 + t.getMinutes();
          const et = h * 60 + m;
          if (tt > et) continue;
        }
      }
      out.times.push(times[i]);
      out.ltp.push(ltp[i]);
      out.sma.push(sma[i]);
      out.delta.push(delta[i]);
      out.gamma.push(gamma[i]);
      out.theta.push(theta[i]);
    }
    return out;
  }, [indicators, startDate, endDate, startTime, endTime]);

  // Gamma axis clipping (1st-99th percentiles) to avoid huge spikes blowing up the chart
  const gammaRange = useMemo(() => {
    const g = filtered.gamma.filter(Number.isFinite);
    if (g.length === 0) return null;
    const p1 = percentile(g, 1);
    const p99 = percentile(g, 99);
    if (!Number.isFinite(p1) || !Number.isFinite(p99)) return null;
    // If p1==p99 (flat data), return autorange
    if (Math.abs(p99 - p1) < 1e-12) return null;
    return [p1, p99];
  }, [filtered.gamma]);

  // Simple helper to create plot config for an indicator (two-column usage)
  function makePlotConfig(indicatorName, yData, extraTraces = []) {
    const layout = {
      margin: { t: 30, b: 35, l: 50, r: 20 },
      hovermode: "x unified",
      showlegend: false,
      title: { text: indicatorName, x: 0.01, xanchor: "left" },
    };

    // Special-case gamma axis clipping
    if (indicatorName.toLowerCase().includes("gamma") && gammaRange) {
      layout.yaxis = { range: gammaRange };
    } else {
      layout.yaxis = { automargin: true, autorange: true }; // let Plotly decide
    }

    const traces = [
      {
        x: filtered.times,
        y: yData,
        type: "scatter",
        mode: "lines",
        line: { width: 1 },
        hovertemplate: "%{x}<br>%{y:.6f}<extra></extra>",
      },
      ...extraTraces,
    ];

    return { data: traces, layout, config: { displaylogo: false }, style: { width: "100%", height: "100%" } };
  }

  // Two columns per row — we'll render the same indicator twice (placeholder for multiple instruments)
  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header / Controls */}
        <div className="flex flex-col sm:flex-row sm:items-end sm:space-x-6 gap-4 mb-6">
          <div className="flex-1">
            <label className="block text-sm font-medium text-gray-700">Dashboard Title</label>
            <input
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              className="mt-1 p-2 w-full rounded border"
              placeholder="Enter dashboard title..."
            />
          </div>

          <div className="flex space-x-2 items-end">
            <div>
              <label className="block text-sm">Start date</label>
              <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} className="mt-1 p-2 rounded border" />
            </div>
            <div>
              <label className="block text-sm">End date</label>
              <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} className="mt-1 p-2 rounded border" />
            </div>
            <div>
              <label className="block text-sm">Start time</label>
              <input type="time" value={startTime} onChange={(e) => setStartTime(e.target.value)} className="mt-1 p-2 rounded border" />
            </div>
            <div>
              <label className="block text-sm">End time</label>
              <input type="time" value={endTime} onChange={(e) => setEndTime(e.target.value)} className="mt-1 p-2 rounded border" />
            </div>
          </div>

          <div className="w-48">
            <label className="block text-sm">MA window</label>
            <input
              type="number"
              min={1}
              value={maWindow}
              onChange={(e) => setMaWindow(Number(e.target.value))}
              className="mt-1 p-2 rounded border w-full"
            />
          </div>
        </div>

        <header className="mb-4">
          <h1 className="text-2xl font-bold">{title}</h1>
          <p className="text-sm text-gray-600">Showing {filtered.times.length} points</p>
        </header>

        {/* Grid: 4 rows x 2 columns */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Row 1: LTP - Left */}
          <div className="h-64 bg-white rounded shadow p-2">
            <Plot
              {...makePlotConfig("LTP", filtered.ltp, [
                {
                  x: filtered.times,
                  y: filtered.sma,
                  type: "scatter",
                  mode: "lines",
                  name: "SMA",
                  line: { dash: "dash", width: 1.5 },
                },
              ])}
            />
          </div>

          {/* Row 1: LTP - Right (example: overlay open/close - here same as LTP for placeholder) */}
          <div className="h-64 bg-white rounded shadow p-2">
            <Plot
              {...makePlotConfig("LTP (compare)", filtered.ltp, [
                {
                  x: filtered.times,
                  y: filtered.sma,
                  type: "scatter",
                  mode: "lines",
                  name: "SMA",
                  line: { dash: "dash", width: 1.5 },
                },
              ])}
            />
          </div>

          {/* Row 2: Delta */}
          <div className="h-64 bg-white rounded shadow p-2">
            <Plot {...makePlotConfig("Delta", filtered.delta)} />
          </div>
          <div className="h-64 bg-white rounded shadow p-2">
            <Plot {...makePlotConfig("Delta (compare)", filtered.delta)} />
          </div>

          {/* Row 3: Gamma (with clipping handling) */}
          <div className="h-64 bg-white rounded shadow p-2">
            <Plot {...makePlotConfig("Gamma", filtered.gamma)} />
          </div>
          <div className="h-64 bg-white rounded shadow p-2">
            <Plot {...makePlotConfig("Gamma (compare)", filtered.gamma)} />
          </div>

          {/* Row 4: Theta */}
          <div className="h-64 bg-white rounded shadow p-2">
            <Plot {...makePlotConfig("Theta", filtered.theta)} />
          </div>
          <div className="h-64 bg-white rounded shadow p-2">
            <Plot {...makePlotConfig("Theta (compare)", filtered.theta)} />
          </div>
        </div>

        {/* Footer: quick actions */}
        <div className="mt-6 flex space-x-3">
          <button
            className="px-4 py-2 rounded bg-blue-600 text-white shadow"
            onClick={() => {
              // reset filters
              setStartDate("");
              setEndDate("");
              setStartTime("");
              setEndTime("");
            }}
          >
            Reset filters
          </button>

          <button
            className="px-4 py-2 rounded bg-gray-200"
            onClick={() => {
              // Download visible data as CSV
              const rows = [
                ["datetime", "ltp", "sma", "delta", "gamma", "theta"],
                ...filtered.times.map((t, i) => [t, filtered.ltp[i], filtered.sma[i], filtered.delta[i], filtered.gamma[i], filtered.theta[i]]),
              ];
              const csv = rows.map((r) => r.map((c) => (c === null || c === undefined ? "" : String(c))).join(",")).join("\n");
              const blob = new Blob([csv], { type: "text/csv" });
              const url = URL.createObjectURL(blob);
              const a = document.createElement("a");
              a.href = url;
              a.download = `${title.replace(/\s+/g, "_") || "dashboard"}.csv`;
              a.click();
              URL.revokeObjectURL(url);
            }}
          >
            Download CSV
          </button>
        </div>
      </div>
    </div>
  );
}
