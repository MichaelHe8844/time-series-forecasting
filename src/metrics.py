"""
Shared financial metrics — pure NumPy/SciPy, no torch dependency.
Safe for XGBoost and all deep learning models.
"""
import numpy as np
from scipy.stats import pearsonr, spearmanr
from configs.config import cfg


def calc_ic(pred, target):
    mask = np.isfinite(pred) & np.isfinite(target)
    if mask.sum() < 2:
        return 0.0
    ic = spearmanr(pred[mask], target[mask]).correlation
    if ic is None or np.isnan(ic):
        return 0.0
    return float(ic)


def calc_pearson_ic(pred, target):
    mask = np.isfinite(pred) & np.isfinite(target)
    if mask.sum() < 2:
        return 0.0
    ic = pearsonr(pred[mask], target[mask])[0]
    if ic is None or np.isnan(ic):
        return 0.0
    return float(ic)


def calc_da(pred, target):
    mask = np.isfinite(pred) & np.isfinite(target)
    if mask.sum() == 0:
        return 0.5
    return float(np.mean((pred[mask] > 0) == (target[mask] > 0)))


def calc_mse(pred, target):
    mask = np.isfinite(pred) & np.isfinite(target)
    if mask.sum() == 0:
        return 0.0
    return float(np.mean((pred[mask] - target[mask]) ** 2))


def calc_strategy_returns(pred, target, fee=0.0005):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    positions = np.sign(pred)
    raw_returns = positions * target
    if len(positions) > 1:
        position_changes = np.abs(positions[1:] - positions[:-1])
        costs = fee * position_changes
        costs = np.concatenate(([0.0], costs))
    else:
        costs = np.zeros_like(raw_returns)
    net_returns = raw_returns - costs
    net_returns = np.clip(net_returns, -0.3, 0.3)
    return net_returns


def calc_sharpe(returns, periods_per_year=cfg["PERIODS_PER_YEAR"]):
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[np.isfinite(returns)]
    if returns.size == 0:
        return 0.0
    std = returns.std(ddof=1) if returns.size > 1 else 0.0
    if std <= 1e-12 or np.isnan(std):
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / std)


def calc_ic_ir(pred, target, window=48):
    """
    IC-based Information Ratio (Grinold & Kahn framework).

    IR = mean(rolling_IC_t) / std(rolling_IC_t)

    Each IC_t is the Rank IC (Spearman) over a rolling window of ``window``
    consecutive predictions.  This measures signal *consistency* rather than
    strategy-level risk-adjusted returns, and is the standard IR definition
    in quantitative finance literature.

    Parameters
    ----------
    pred : ndarray
        Model predictions.
    target : ndarray
        True forward returns.
    window : int
        Rolling-window size (number of periods).  Default 48 matches LOOKBACK.
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    mask = np.isfinite(pred) & np.isfinite(target)
    pred = pred[mask]
    target = target[mask]

    n = len(pred)
    if n < window or n < 2:
        return 0.0

    rolling_ics = []
    for i in range(window, n + 1):
        p = pred[i - window:i]
        t = target[i - window:i]
        if len(p) < 2:
            continue
        ic = spearmanr(p, t).correlation
        if ic is None or np.isnan(ic):
            continue
        rolling_ics.append(ic)

    if len(rolling_ics) < 2:
        return 0.0

    rolling_ics = np.array(rolling_ics, dtype=np.float64)
    mean_ic = rolling_ics.mean()
    std_ic = rolling_ics.std(ddof=1)
    if std_ic <= 1e-12 or np.isnan(std_ic):
        return 0.0

    return float(mean_ic / std_ic)


def calc_max_drawdown(returns):
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[np.isfinite(returns)]
    if returns.size == 0:
        return 0.0
    equity_curve = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve / np.where(running_max == 0, 1.0, running_max) - 1.0
    return float(abs(drawdown.min()))


def calc_annual_return(returns, periods_per_year=cfg["PERIODS_PER_YEAR"]):
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[np.isfinite(returns)]
    if returns.size == 0:
        return 0.0
    avg_return = np.mean(returns)
    return float(avg_return * periods_per_year)
