"""Tests for dashboard.components (Layer 1: pure, Streamlit-independent logic).

Fixes the public API surface:

- SignificancePanelContent(sharpe, permutation_p, bootstrap_ci, agree,
  caution_text): a plain dataclass with NO color/style field of any kind
  -- the absence of such a field is itself the guarantee that Sharpe is
  never shown with conditional formatting; there is no attribute to hang
  a color on.
- build_significance_panel_content(oos_metrics, significance) ->
  SignificancePanelContent: assembles the above from the same
  oos_metrics/SignificanceCheck types already used throughout
  core.evaluation.
- tier1_banner_text() -> str: fixed banner for precomputed, non-reactive
  sections.
- tier2_banner_text(as_of: date) -> str: fixed banner for live-recomputed
  sections, must include the as-of date of the frozen snapshot.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from dashboard.components import (
    SignificancePanelContent,
    build_significance_panel_content,
    tier1_banner_text,
    tier2_banner_text,
)
from core.evaluation.statistical_tests import BootstrapResult, PermutationResult, SignificanceCheck


def _significance(p_value: float = 0.3, lower: float = -0.001, upper: float = 0.001, agree: bool = True) -> SignificanceCheck:
    permutation = PermutationResult(
        observed=0.01, p_value=p_value, n_permutations=2000, alternative="greater",
        null_distribution=np.array([0.0]), seed=0,
    )
    bootstrap = BootstrapResult(
        point_estimate=0.005, lower=lower, upper=upper, confidence_level=0.95,
        standard_error=0.002, distribution=np.array([0.0]), n_resamples=2000, seed=0,
    )
    return SignificanceCheck(permutation=permutation, bootstrap=bootstrap, agree=agree)


# ---------------------------------------------------------------------------
# 1. build_significance_panel_content
# ---------------------------------------------------------------------------


def test_build_significance_panel_content_populates_all_five_fields() -> None:
    oos_metrics = {"sharpe_ratio": 0.42}
    significance = _significance(p_value=0.123, lower=-0.01, upper=0.02, agree=False)

    content = build_significance_panel_content(oos_metrics, significance)

    assert content.sharpe == 0.42
    assert content.permutation_p == 0.123
    assert content.bootstrap_ci == (-0.01, 0.02)
    assert content.agree is False
    assert content.caution_text  # non-empty


def test_build_significance_panel_content_caution_text_mentions_multiple_comparisons() -> None:
    content = build_significance_panel_content({"sharpe_ratio": 0.1}, _significance())

    assert "多重検定" in content.caution_text or "multiple" in content.caution_text.lower()


def test_significance_panel_content_has_no_color_or_style_field() -> None:
    # The absence of any color/style/highlight attribute is the actual
    # guarantee here -- not a policy that could be silently violated by a
    # future edit, but a type that structurally cannot carry one.
    field_names = {f for f in SignificancePanelContent.__dataclass_fields__}
    assert field_names == {"sharpe", "permutation_p", "bootstrap_ci", "agree", "caution_text"}


# ---------------------------------------------------------------------------
# 2. tier1_banner_text / tier2_banner_text
# ---------------------------------------------------------------------------


def test_tier1_banner_text_states_precomputed_and_non_reactive() -> None:
    text = tier1_banner_text()

    assert "事前計算" in text
    assert "反応しません" in text


def test_tier2_banner_text_includes_as_of_date() -> None:
    text = tier2_banner_text(date(2026, 7, 25))

    assert "2026-07-25" in text


def test_tier1_and_tier2_banner_texts_are_distinct() -> None:
    assert tier1_banner_text() != tier2_banner_text(date(2026, 7, 25))
