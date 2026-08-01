"""Performance and risk metrics.

Every function operates on a plain ``pd.Series`` of periodic (typically
daily) simple returns and is agnostic to how many tickers produced it --
this module is deliberately decoupled from ``core.backtest``: it never
takes a ``BacktestResult``, only a returns Series (and, for
``benchmark_comparison``, a second Series). That keeps it reusable for
anything that produces a daily-return record, live trading included, not
just this repo's own backtests.

Every annualisation factor is a named constant with the reasoning spelled
out, because a silently-wrong annualisation convention is one of the
quietest ways a backtest misleads.

IMPORTANT: a Sharpe/Sortino/Calmar number computed here is not, by itself,
evidence of a real edge. Per the project's backtest-verification standard,
these metrics are only reportable after Pillar 1 (causality --
``core.backtest.engine.assert_causal`` / ``assert_backtest_causal``) and
Pillar 2 (statistical significance -- ``core.evaluation.statistical_tests``)
have been checked. This module computes the numbers; it does not certify
them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# Annualisation constants
# ---------------------------------------------------------------------------

# US equity markets trade on roughly 252 days per year (365.25 calendar days
# minus weekends and ~9 public holidays). 252 is the long-standing industry
# convention for annualising daily statistics; using 365 here would overstate
# annualised volatility by ~sqrt(365/252) ~ 1.20x.
TRADING_DAYS_PER_YEAR: int = 252

# Volatility scales with the square root of time under the i.i.d. assumption
# (variance is additive, so standard deviation grows as sqrt(n)). Hence daily
# volatility is annualised by multiplying by sqrt(252), not by 252.
VOL_ANNUALISATION_FACTOR: float = float(np.sqrt(TRADING_DAYS_PER_YEAR))

# Sample standard deviation uses ddof=1 (Bessel's correction) because a
# backtest return series is a *sample*, not the full population, of outcomes.
_DDOF: int = 1

# Standard deviations (and downside deviations) at or below this absolute
# floor are treated as zero. A genuinely constant series should have std
# exactly 0, but finite-precision arithmetic leaves ~1e-18 of noise; no real
# daily-return series has a standard deviation anywhere near 1e-12, so this
# cleanly separates "constant"/"no downside" (undefined ratio) from "real"
# without risking false positives.
_STD_FLOOR: float = 1e-12

# benchmark_comparison fits two parameters (intercept + slope), so it needs
# n - 2 >= 1 observations to keep a residual degree of freedom. At n == 2 the
# line passes exactly through both points: r_squared is identically 1.0 and
# both p-values are NaN, i.e. every diagnostic the caller would use to judge
# the fit is uninformative. Three is therefore the true mathematical floor.
_MIN_REGRESSION_OBSERVATIONS: int = 3


@dataclass(frozen=True)
class DrawdownInfo:
    """Result of a maximum-drawdown calculation.

    Attributes
    ----------
    max_drawdown:
        The largest peak-to-trough decline, as a negative fraction (e.g.
        ``-0.25`` for a 25% drawdown). Zero if the equity never declines.
    peak_date, trough_date:
        Index labels of the peak preceding, and the worst point of, the
        drawdown.
    duration:
        Number of bars from the peak to the trough.
    """

    max_drawdown: float
    peak_date: object
    trough_date: object
    duration: int


@dataclass(frozen=True)
class BenchmarkComparison:
    """OLS regression of strategy returns on benchmark returns.

    Attributes
    ----------
    alpha:
        Annualised intercept, via LINEAR scaling (``daily_alpha *
        TRADING_DAYS_PER_YEAR``) -- the standard CAPM-alpha convention, and
        deliberately NOT the geometric compounding ``annualized_return``
        uses for CAGR. The two are not comparable numbers.
    beta:
        Regression slope (market/benchmark sensitivity).
    r_squared, alpha_pvalue, beta_pvalue:
        Standard OLS diagnostics.
    """

    alpha: float
    beta: float
    r_squared: float
    alpha_pvalue: float
    beta_pvalue: float


def _to_returns(returns: pd.Series) -> pd.Series:
    """Coerce to a float return series, dropping a possible leading NaN."""
    return returns.astype(float).dropna()


def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Compound ``returns`` into a wealth index starting at ``initial``."""
    return (1.0 + _to_returns(returns)).cumprod() * initial


def annualized_return(returns: pd.Series) -> float:
    """Compound annual growth rate (CAGR) implied by the return series.

    Computed from the *total* compounded return over the sample, annualised
    by the number of years the sample spans (``n_bars / 252``) -- unlike a
    naive ``mean * 252`` which ignores compounding.
    """
    r = _to_returns(returns)
    if r.empty:
        return float("nan")
    total_growth = float(np.prod((1.0 + r).to_numpy()))
    years = len(r) / TRADING_DAYS_PER_YEAR
    if total_growth <= 0:
        return -1.0
    return total_growth ** (1.0 / years) - 1.0


def annualized_volatility(returns: pd.Series) -> float:
    """Annualised standard deviation of returns. NaN for fewer than two
    observations, where sample variance is undefined."""
    r = _to_returns(returns)
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=_DDOF)) * VOL_ANNUALISATION_FACTOR


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised Sharpe ratio: ``mean(excess) / std(excess) * sqrt(periods_per_year)``.

    ``risk_free_rate`` is an *annual* rate, de-annualised to a per-period
    rate before subtraction. NaN when the excess-return standard deviation
    is at or below ``_STD_FLOOR`` (a risk-free/constant series has an
    undefined Sharpe -- this returns NaN rather than an infinite ratio).
    """
    r = _to_returns(returns)
    if len(r) < 2:
        return float("nan")
    per_period_rf = risk_free_rate / periods_per_year
    excess = r - per_period_rf
    std = excess.std(ddof=_DDOF)
    if std <= _STD_FLOOR:
        return float("nan")
    return float(excess.mean() / std) * float(np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised Sortino ratio: like sharpe_ratio, but the denominator is
    the *downside deviation* rather than total standard deviation, so only
    below-target volatility is penalised.

    The minimum acceptable return (MAR) is shared with ``risk_free_rate``
    (a common simplification -- with the default ``risk_free_rate=0.0``
    this is the conventional MAR=0 downside deviation). Downside deviation
    is computed over ALL periods (upside deviations clipped to 0, per the
    Sortino & van der Meer definition), not just the count of below-target
    periods -- dividing by a smaller denominator would overstate the ratio.

    NaN when downside deviation is at or below ``_STD_FLOOR`` (no
    meaningful downside risk -- e.g. every period met or beat the MAR) --
    returned as undefined rather than an infinite ratio, mirroring
    sharpe_ratio's std==0 convention.
    """
    r = _to_returns(returns)
    if len(r) < 2:
        return float("nan")
    per_period_rf = risk_free_rate / periods_per_year
    excess = r - per_period_rf
    downside = excess.clip(upper=0.0)
    downside_dev = float(np.sqrt((downside**2).mean()))
    if downside_dev <= _STD_FLOOR:
        return float("nan")
    return float(excess.mean() / downside_dev) * float(np.sqrt(periods_per_year))


def calmar_ratio(returns: pd.Series) -> float:
    """Annualised return divided by the absolute maximum drawdown.

    NaN when there is no drawdown at all (a monotonically non-decreasing
    equity curve) -- undefined rather than an infinite ratio, mirroring
    sharpe_ratio's / sortino_ratio's zero-denominator convention.
    """
    dd = max_drawdown(returns).max_drawdown
    if dd == 0.0:
        return float("nan")
    return annualized_return(returns) / abs(dd)


def max_drawdown(returns: pd.Series) -> DrawdownInfo:
    """Maximum peak-to-trough decline of the compounded equity curve."""
    r = _to_returns(returns)
    if r.empty:
        return DrawdownInfo(0.0, None, None, 0)

    wealth = (1.0 + r).cumprod()
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1.0

    trough_pos = int(drawdown.to_numpy().argmin())
    trough_date = drawdown.index[trough_pos]
    max_dd = float(drawdown.iloc[trough_pos])

    if max_dd == 0.0:
        return DrawdownInfo(0.0, wealth.index[0], wealth.index[0], 0)

    peak_pos = int(wealth.iloc[: trough_pos + 1].to_numpy().argmax())
    peak_date = wealth.index[peak_pos]
    duration = trough_pos - peak_pos

    return DrawdownInfo(max_dd, peak_date, trough_date, duration)


def win_rate(returns: pd.Series) -> float:
    """Fraction of non-zero return periods that are positive. Flat periods
    are excluded so the statistic isn't diluted by time spent flat. NaN if
    there are no non-zero periods."""
    r = _to_returns(returns)
    active = r[r != 0.0]
    if active.empty:
        return float("nan")
    return float((active > 0).mean())


def profit_loss_ratio(returns: pd.Series) -> float:
    """Average winning return divided by the absolute average losing
    return. NaN if there are no losers or no winners."""
    r = _to_returns(returns)
    wins = r[r > 0.0]
    losses = r[r < 0.0]
    if wins.empty or losses.empty:
        return float("nan")
    return float(wins.mean() / abs(losses.mean()))


def benchmark_comparison(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> BenchmarkComparison:
    """OLS regression: ``returns = alpha + beta * benchmark_returns + eps``.

    IMPORTANT: the ValueError below only guards against a regression whose
    own diagnostics are undefined. This model fits TWO parameters
    (intercept and slope), so it needs at least three observations to
    retain one residual degree of freedom; on exactly two points it
    interpolates them, forcing ``r_squared`` to 1.0 and both p-values to
    NaN. Clearing that floor is NOT a claim that the sample is
    statistically adequate -- 3 or 20 overlapping days both pass it. Treat
    r_squared/alpha_pvalue/beta_pvalue, and plain common sense about the
    sample size, as the actual adequacy check; do not read "no ValueError"
    as "enough data". The floor exists precisely so that those
    diagnostics are real numbers when you go to read them.

    Raises
    ------
    ValueError
        If fewer than three dates overlap between the two series.
    """
    aligned = pd.concat(
        [returns.astype(float).rename("strategy"), benchmark_returns.astype(float).rename("benchmark")],
        axis=1,
    ).dropna()
    if len(aligned) < _MIN_REGRESSION_OBSERVATIONS:
        raise ValueError(
            f"need at least three overlapping observations to run the regression, "
            f"got {len(aligned)} (a two-parameter fit on two points has zero "
            "residual degrees of freedom: r_squared is forced to 1.0 and both "
            "p-values are NaN). This is a mathematical floor, not a statistical "
            "adequacy check -- see the function docstring"
        )

    predictors = sm.add_constant(aligned["benchmark"])
    model = sm.OLS(aligned["strategy"], predictors).fit()

    return BenchmarkComparison(
        alpha=float(model.params["const"]) * periods_per_year,
        beta=float(model.params["benchmark"]),
        r_squared=float(model.rsquared),
        alpha_pvalue=float(model.pvalues["const"]),
        beta_pvalue=float(model.pvalues["benchmark"]),
    )


def summary(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """Compute the full metric suite as a flat dictionary."""
    dd = max_drawdown(returns)
    return {
        "annualized_return": annualized_return(returns),
        "annualized_volatility": annualized_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate, periods_per_year),
        "calmar_ratio": calmar_ratio(returns),
        "max_drawdown": dd.max_drawdown,
        "max_drawdown_duration": float(dd.duration),
        "win_rate": win_rate(returns),
        "profit_loss_ratio": profit_loss_ratio(returns),
    }
