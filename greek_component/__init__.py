import os
import streamlit.components.v1 as components

_RELEASE = False

if not _RELEASE:
    _component_func = components.declare_component(
        "plotly_streamer",
        path=os.path.join(os.path.dirname(__file__), "frontend", "build")
    )
else:
    _component_func = components.declare_component("plotly_streamer")

def plotly_streamer(dataJson):
    return _component_func(dataJson=dataJson)
