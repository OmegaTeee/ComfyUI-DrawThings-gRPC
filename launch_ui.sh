#!/bin/bash
# Launch Draw Things Streamlit UI
cd "$(dirname "$0")"
source ../../.venv/bin/activate
streamlit run streamlit_app.py --server.headless true
