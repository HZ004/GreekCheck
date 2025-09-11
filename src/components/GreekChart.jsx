import React from "react";
import Plot from "react-plotly.js";

/**
 * GreekChart.jsx
 * A reusable Plotly-based chart component for plotting any option Greek (Delta, Gamma, Theta, etc.).
 *
 * Props:
 * - title: string - The chart title (e.g., "Delta")
 * - times: array of timestamps (x-axis)
 * - values: array of numeric values (y-axis)
 * - yRange: optional [min, max] for Y-axis (null for auto)
 * - compareValues: optional array for comparison series
 * - height: optional number (chart height in px)
 * - lineColor: optional string (hex or named color)
 */

export default function GreekChart({
  title = "Greek",
  times = [],
  values = [],
  yRange = null,
  compareValues = null,
  height = 300,
  lineColor = "#2563eb", // Tailwind's blue-600
}) {
  const traces = [
    {
      x: times,
      y: values,
      type: "scatter",
      mode: "lines",
      name: title,
      line: { color: lineColor, width: 1.5 },
      hovertemplate: "%{x}<br>%{y:.6f}<extra></extra>",
    },
  ];

  if (Array.isArray(compareValues)) {
    traces.push({
      x: times,
      y: compareValues,
      type: "scatter",
      mode: "lines",
      name: `${title} (compare)` || "Compare",
      line: { dash: "dash", color: "#dc2626", width: 1.5 },
      hovertemplate: "%{x}<br>%{y:.6f}<extra></extra>",
    });
  }

  const layout = {
    title: { text: title, x: 0.01, xanchor: "left" },
    margin: { t: 30, b: 35, l: 50, r: 20 },
    hovermode: "x unified",
    showlegend: compareValues ? true : false,
    autosize: true,
    height,
    yaxis: yRange ? { range: yRange } : { automargin: true, autorange: true },
    xaxis: { automargin: true },
  };

  const config = { displaylogo: false, responsive: true };

  return (
    <div className="bg-white rounded shadow p-2 w-full h-full">
      <Plot data={traces} layout={layout} config={config} style={{ width: "100%", height: "100%" }} />
    </div>
  );
}
