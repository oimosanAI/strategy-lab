"""Streamlit dashboard entrypoint.

Run: poetry run streamlit run app.py
(requires the optional `dashboard` dependency group: `poetry install
--with dashboard` -- see pyproject.toml, streamlit is deliberately kept
out of the default install so a plain `poetry install` for development
doesn't pull in the whole Streamlit stack.)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

st.set_page_config(page_title="strategy-lab", layout="wide")

PAGES_DIR = Path(__file__).resolve().parent / "dashboard" / "pages"

pages = [
    st.Page(str(PAGES_DIR / "overview.py"), title="Overview", default=True),
    st.Page(str(PAGES_DIR / "pairs_trading.py"), title="Pairs Trading"),
    st.Page(str(PAGES_DIR / "factor_momentum.py"), title="Factor Momentum"),
    st.Page(str(PAGES_DIR / "vol_arbitrage.py"), title="Vol Arbitrage"),
    st.Page(str(PAGES_DIR / "limitations.py"), title="既知の限界"),
]

navigation = st.navigation(pages)
navigation.run()
