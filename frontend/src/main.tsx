import React from "react";
import ReactDOM from "react-dom/client";
import PlotlyStreamer from "./PlotlyStreamer";

const container = document.getElementById("root");

const root = ReactDOM.createRoot(container!);
root.render(<PlotlyStreamer dataJson={window.dataJson || ""} />);
