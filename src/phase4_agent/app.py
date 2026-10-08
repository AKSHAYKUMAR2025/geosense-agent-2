# app.py
# Purpose: Streamlit demo app — text question box,
# agent response, and embedded 3D twin

import streamlit as st
import streamlit.components.v1 as components

from geoai_agent import ask_agent


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="GeoSense Agent 2.0",
    layout="wide"
)


# ---------------------------------------------------------
# Title
# ---------------------------------------------------------

st.title(
    "GeoSense Agent 2.0 — Conversational Geospatial Intelligence"
)


# ---------------------------------------------------------
# Question input
# ---------------------------------------------------------

question = st.text_input(
    "Ask a question about any location:",
    "Is this area in Chennai suitable for a new clinic?"
)


# ---------------------------------------------------------
# Ask GeoSense Agent
# ---------------------------------------------------------

if st.button("Ask GeoSense Agent"):

    with st.spinner(
        "Agent is reasoning and calling tools..."
    ):

        answer = ask_agent(question)

    st.success(answer)


# ---------------------------------------------------------
# 3D Digital Twin
# ---------------------------------------------------------

st.subheader("3D Digital Twin")

components.iframe(
    "http://localhost:5173",
    height=600,
    scrolling=True
)