"""Layer 2 (streamlit.testing.v1.AppTest) wiring tests for dashboard
pages. Deliberately thin: only checks (1) Tier 1/Tier 2 banners are
actually rendered, (2) a slider move re-triggers the SAME computation a
direct function call would produce, (3) the UI applies no special
highlighting/formatting that would depend on how "good" a slider value's
result looks. Layout/CSS/navigation are out of scope -- see
dashboard/components.py's own Layer 1 tests for the content logic these
pages merely wire up.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from core.backtest.engine import BacktestConfig, run_backtest
from core.backtest.position_sizing import PositionSizingConfig, VolTargetSizer
from core.backtest.sample_split import SamplePeriod, TrainTestSplit, split_returns
from core.evaluation import metrics as metrics_mod
from strategies.vol_arbitrage.strategy import VolArbitrageConfig, VolArbitrageStrategy

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PAGES_DIR = REPO_ROOT / "dashboard" / "pages"


def _run_page(page_name: str, data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setenv("DASHBOARD_DATA_DIR", str(data_dir))
    at = AppTest.from_file(str(PAGES_DIR / page_name))
    at.run()
    assert not at.exception
    return at


# ---------------------------------------------------------------------------
# 0. Static pages (no Tier data, no sliders) -- smoke-test only
# ---------------------------------------------------------------------------


def test_overview_page_loads_without_exception() -> None:
    at = AppTest.from_file(str(PAGES_DIR / "overview.py"))
    at.run()
    assert not at.exception


def test_limitations_page_loads_without_exception() -> None:
    at = AppTest.from_file(str(PAGES_DIR / "limitations.py"))
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# 1. Tier 1 / Tier 2 banners actually render
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("page_name", ["pairs_trading.py", "factor_momentum.py", "vol_arbitrage.py"])
def test_page_shows_both_tier_banners(page_name: str, dashboard_test_data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _run_page(page_name, dashboard_test_data_dir, monkeypatch)

    info_texts = [m.value for m in at.info]
    assert any("事前計算" in text and "反応しません" in text for text in info_texts)
    assert any("ライブ取得ではありません" in text for text in info_texts)


def test_pairs_trading_page_shows_reference_only_warning(dashboard_test_data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _run_page("pairs_trading.py", dashboard_test_data_dir, monkeypatch)

    warning_texts = [m.value for m in at.warning]
    assert any("参考情報" in text and "生存ペア0本" in text for text in warning_texts)


# ---------------------------------------------------------------------------
# 2. Slider -> run_backtest wiring
# ---------------------------------------------------------------------------


def test_vol_arbitrage_slider_matches_direct_backtest(dashboard_test_data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _run_page("vol_arbitrage.py", dashboard_test_data_dir, monkeypatch)

    at.slider[0].set_value(0.02).run()

    close_panel = pd.read_parquet(dashboard_test_data_dir / "snapshot_vol_arbitrage_close.parquet")
    open_panel = pd.read_parquet(dashboard_test_data_dir / "snapshot_vol_arbitrage_open.parquet")
    split = TrainTestSplit(
        in_sample=SamplePeriod(pd.Timestamp("2023-01-03"), pd.Timestamp("2024-12-31")),
        out_of_sample=SamplePeriod(pd.Timestamp("2025-01-02"), close_panel.index[-1]),
    )
    strategy = VolArbitrageStrategy(config=VolArbitrageConfig(vrp_threshold=0.02))
    sizer = VolTargetSizer(PositionSizingConfig())
    result = run_backtest(close_panel, open_panel, strategy, sizer, BacktestConfig())
    _, oos_returns = split_returns(result.returns, split)
    expected_sharpe = metrics_mod.summary(oos_returns)["sharpe_ratio"]

    displayed = [m.value for m in at.markdown if m.value.startswith("OOS Sharpe:")]
    assert len(displayed) == 1
    assert displayed[0] == f"OOS Sharpe: {expected_sharpe:.2f}"


def test_vol_arbitrage_slider_change_updates_displayed_sharpe(dashboard_test_data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _run_page("vol_arbitrage.py", dashboard_test_data_dir, monkeypatch)
    at.slider[0].set_value(-0.05).run()
    sharpe_low = next(m.value for m in at.markdown if m.value.startswith("OOS Sharpe:"))

    at.slider[0].set_value(0.05).run()
    sharpe_high = next(m.value for m in at.markdown if m.value.startswith("OOS Sharpe:"))

    assert sharpe_low != sharpe_high


# ---------------------------------------------------------------------------
# 3. No special highlighting based on how "good" a slider value looks
# ---------------------------------------------------------------------------


def test_vol_arbitrage_page_renders_identically_structured_output_regardless_of_slider_value(
    dashboard_test_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = _run_page("vol_arbitrage.py", dashboard_test_data_dir, monkeypatch)

    at.slider[0].set_value(-0.05).run()  # a "bad-looking" value
    n_markdown_bad = len(at.markdown)
    n_success_bad = len(at.success)

    at.slider[0].set_value(0.05).run()  # a "good-looking" value
    n_markdown_good = len(at.markdown)
    n_success_good = len(at.success)

    # Same number of rendered elements either way, and no st.success/
    # celebratory element ever appears -- the UI has no code path that
    # reacts to the Sharpe value's sign or magnitude.
    assert n_markdown_bad == n_markdown_good
    assert n_success_bad == n_success_good == 0
