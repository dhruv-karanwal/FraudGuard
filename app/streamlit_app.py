"""FraudGuard: fraud operations console.

Run: streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="FraudGuard", page_icon=str(ROOT / "app" / "assets" / "icon.svg"), layout="wide")

from app import core  # noqa: E402

st.logo(str(core.ASSETS / "logo.svg"), icon_image=str(core.ASSETS / "icon.svg"), size="large")
st.markdown(core.CSS, unsafe_allow_html=True)

if core.TAG is None:
    st.error("No trained model found. Run `python -m src.prepare`, `python -m src.train` and `python -m src.evaluate`.")
    st.stop()

nav = st.navigation({
    "Operations": [
        st.Page("views/overview.py", title="Overview", icon=":material/space_dashboard:", default=True),
        st.Page("views/alerts.py", title="Alert queue", icon=":material/notifications_active:", url_path="alerts"),
        st.Page("views/investigate.py", title="Investigate", icon=":material/policy:", url_path="investigate"),
        st.Page("views/live.py", title="Live monitor", icon=":material/sensors:", url_path="live"),
    ],
    "Tools": [
        st.Page("views/check.py", title="Check transaction", icon=":material/fact_check:", url_path="check"),
        st.Page("views/model.py", title="Model & reports", icon=":material/analytics:", url_path="model"),
    ],
})
nav.run()
