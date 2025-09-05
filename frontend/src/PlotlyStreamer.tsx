import React, { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";

interface TraceData {
  name: string;
  xs: string[];
  ys: number[];
}

interface AllSeries {
  [metric: string]: TraceData[];
}

interface PlotlyStreamerProps {
  dataJson: string;
}

const containerStyle = { width: "100%", height: "600px", marginBottom: "20px" };

const PlotlyStreamer = ({ dataJson }: PlotlyStreamerProps) => {
  const plotRefs = {
    ltpCE: useRef<HTMLDivElement>(null),
    ltpPE: useRef<HTMLDivElement>(null),
    deltaCE: useRef<HTMLDivElement>(null),
    deltaPE: useRef<HTMLDivElement>(null),
    gammaCE: useRef<HTMLDivElement>(null),
    gammaPE: useRef<HTMLDivElement>(null),
    thetaCE: useRef<HTMLDivElement>(null),
    thetaPE: useRef<HTMLDivElement>(null),
  };

  // Store loaded traces for each plot to track existing names and index
  const dataStore = React.useRef<{ [divId: string]: { [traceName: string]: number } }>({});

  useEffect(() => {
    if (!dataJson) return;
    const newData: AllSeries = JSON.parse(dataJson);

    const plotDivs: { [metric: string]: [string, string] } = {
      ltp: ["ltpCE", "ltpPE"],
      delta: ["deltaCE", "deltaPE"],
      gamma: ["gammaCE", "gammaPE"],
      theta: ["thetaCE", "thetaPE"],
    };

    for (const metric in plotDivs) {
      const [divCE, divPE] = plotDivs[metric];

      const containerCE = plotRefs[divCE].current;
      const containerPE = plotRefs[divPE].current;

      if (!containerCE || !containerPE) continue;

      // Initialize plots if not already
      if (!dataStore.current[divCE]) {
        Plotly.newPlot(containerCE, [], {
          margin: { t: 40 },
          title: `${metric.toUpperCase()} Calls (CE)`,
          xaxis: { title: "Time" },
          yaxis: { title: metric.charAt(0).toUpperCase() + metric.slice(1) },
          showlegend: true
        });
        dataStore.current[divCE] = {};
      }
      if (!dataStore.current[divPE]) {
        Plotly.newPlot(containerPE, [], {
          margin: { t: 40 },
          title: `${metric.toUpperCase()} Puts (PE)`,
          xaxis: { title: "Time" },
          yaxis: { title: metric.charAt(0).toUpperCase() + metric.slice(1) },
          showlegend: true
        });
        dataStore.current[divPE] = {};
      }

      // Append or add traces for CE and PE
      newData[metric].forEach((serie) => {
        const optionType = serie.name.startsWith("CE") ? "CE" : "PE";
        const divId = optionType === "CE" ? divCE : divPE;
        const container = optionType === "CE" ? containerCE : containerPE;

        if (!dataStore.current[divId][serie.name]) {
          // Add new trace
          Plotly.addTraces(container, {
            x: [],
            y: [],
            mode: "lines",
            name: serie.name,
          });
          dataStore.current[divId][serie.name] = Object.keys(dataStore.current[divId]).length;
        }

        const traceIndex = dataStore.current[divId][serie.name];
        const plotDiv = container;

        // Current x and y arrays for the trace
        const currentTraceData = plotDiv.data[traceIndex];
        const lastLen = currentTraceData ? currentTraceData.x.length : 0;
        const newX = serie.xs.slice(lastLen);
        const newY = serie.ys.slice(lastLen);

        if (newX.length > 0) {
          Plotly.extendTraces(plotDiv, { x: [newX], y: [newY] }, [traceIndex]);
        }
      });
    }
  }, [dataJson]);

  return (
    <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "space-around" }}>
      <div ref={plotRefs.ltpCE} style={containerStyle} />
      <div ref={plotRefs.ltpPE} style={containerStyle} />
      <div ref={plotRefs.deltaCE} style={containerStyle} />
      <div ref={plotRefs.deltaPE} style={containerStyle} />
      <div ref={plotRefs.gammaCE} style={containerStyle} />
      <div ref={plotRefs.gammaPE} style={containerStyle} />
      <div ref={plotRefs.thetaCE} style={containerStyle} />
      <div ref={plotRefs.thetaPE} style={containerStyle} />
    </div>
  );
};

export default PlotlyStreamer;
