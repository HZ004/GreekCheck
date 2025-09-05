import React from "react";
import ReactDOM from "react-dom/client";
import PlotlyStreamer from "./PlotlyStreamer";

const container = document.getElementById("root");

const render = (data: any) => {
  if (container) {
    const root = ReactDOM.createRoot(container);
    root.render(<PlotlyStreamer dataJson={data.dataJson || ""} />);
  }
};

window.streamlitDraw = render;
