"""Layer 1 (pure, Streamlit-independent) content for the dashboard, plus
thin Layer 2 st.* wrappers around it.

Layer 1 functions return plain dataclasses/strings and are unit-testable
with plain pytest (see tests/dashboard/test_components.py). Layer 2
functions (prefixed render_) call st.* to actually draw the content and
are only exercised via streamlit.testing.v1.AppTest wiring tests -- they
contain no logic of their own beyond unpacking a Layer 1 result.

SignificancePanelContent deliberately has NO color/style/highlight field:
Sharpe is never shown with conditional formatting because there is no
attribute to hang one on, not merely because nothing currently sets one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import streamlit as st

from core.evaluation.statistical_tests import SignificanceCheck

STANDING_MULTIPLE_COMPARISONS_CAUTION = (
    "複数のパラメータ設定を試すこと自体が多重検定の一種であり、"
    "この設定が他より良く見えても統計的に優れていることを意味しません。"
)


@dataclass(frozen=True)
class SignificancePanelContent:
    sharpe: float
    permutation_p: float
    bootstrap_ci: tuple[float, float]
    # Carried from the result rather than hardcoded in the label:
    # check_significance derives the interval's level from the permutation
    # test's alternative, so a one-sided test at alpha=0.05 yields a 90%
    # interval, not 95%. Labelling it "95%" would mislabel the number
    # actually being displayed.
    bootstrap_confidence_level: float
    agree: bool
    caution_text: str


def build_significance_panel_content(
    oos_metrics: dict[str, float], significance: SignificanceCheck
) -> SignificancePanelContent:
    """Assemble the always-together 5-value set: OOS Sharpe never appears
    without its significance context alongside it."""
    return SignificancePanelContent(
        sharpe=oos_metrics["sharpe_ratio"],
        permutation_p=significance.permutation.p_value,
        bootstrap_ci=(significance.bootstrap.lower, significance.bootstrap.upper),
        bootstrap_confidence_level=significance.bootstrap.confidence_level,
        agree=significance.agree,
        caution_text=STANDING_MULTIPLE_COMPARISONS_CAUTION,
    )


def tier1_banner_text() -> str:
    """Fixed banner for precomputed, non-reactive sections."""
    return (
        "この結果は事前計算されたものであり、このページ上でのパラメータ変更には反応しません。"
        "実データでの検証経緯は各戦略のREADME.mdを参照してください。"
    )


def tier2_banner_text(as_of: date) -> str:
    """Fixed banner for live-recomputed sections, naming the frozen
    snapshot's as-of date so live-ness is never implied."""
    return (
        f"以下はスライダー操作に応じてその場で再計算されます"
        f"（凍結データスナップショット上、データ基準日：{as_of.isoformat()}、ライブ取得ではありません）。"
    )


def render_significance_panel(oos_metrics: dict[str, float], significance: SignificanceCheck) -> None:
    """Layer 2: render the 5-value set built by
    build_significance_panel_content. Deliberately uses st.write, never
    st.metric -- st.metric's `delta` argument applies an automatic
    green/red color based on sign, which would silently reintroduce the
    "ranking/highlighting" this project has avoided everywhere else."""
    content = build_significance_panel_content(oos_metrics, significance)
    st.write(f"OOS Sharpe: {content.sharpe:.2f}")
    st.write(f"Permutation p-value: {content.permutation_p:.4f}")
    st.write(
        f"Bootstrap {content.bootstrap_confidence_level:.0%} CI: "
        f"[{content.bootstrap_ci[0]:.4f}, {content.bootstrap_ci[1]:.4f}]"
    )
    st.write(f"Agreement: {'Yes' if content.agree else 'No'}")
    st.caption(content.caution_text)
