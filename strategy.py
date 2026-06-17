"""Strategy engine for the oracle lag scalper.

Features:
- Brownian motion probability estimation
- Composite weighted signal (Archetapp 7-indicator model)
- Kelly criterion position sizing (quarter-Kelly default)
- Fee-aware edge evaluation
- Hourly stats tracking for Telegram summaries
"""

import time
import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class TradeSignal:
    side: str
    confidence: float
    btc_delta_pct: float
    market_price: float
    edge: float
    true_prob: float
    seconds_remaining: float
    kelly_size: float


@dataclass
class StrategyConfig:
    min_edge: float = 0.25          # Minimum edge (prob - price) — must be meaningful (raised from 0.15 after dry-run analysis)
    min_prob: float = 0.80          # Minimum model probability to consider entry
    entry_window_start: int = 240
    entry_window_end: int = 10
    max_price: float = 0.90
    min_price: float = 0.10
    min_btc_delta: float = 0.06     # minimum |btc_delta_pct| — below this the oracle and Binance can disagree
    kelly_fraction: float = 0.25    # Quarter-Kelly (conservative)
    min_bet: float = 5.0            # Polymarket minimum notional
    max_bet: float = 25.0           # Hard cap per trade


@dataclass
class HourlyStats:
    """Tracks metrics for the current hour. Resets every hour."""
    hour_start: float = 0.0
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl: float = 0.0
    windows_seen: int = 0
    windows_skipped: int = 0
    edges: list = field(default_factory=list)
    deltas: list = field(default_factory=list)
    trade_profits: list = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return (self.wins / self.trades * 100) if self.trades > 0 else 0.0

    @property
    def avg_edge(self) -> float:
        return sum(self.edges) / len(self.edges) if self.edges else 0.0

    @property
    def avg_delta(self) -> float:
        return sum(abs(d) for d in self.deltas) / len(self.deltas) if self.deltas else 0.0

    @property
    def best_trade(self) -> float:
        return max(self.trade_profits) if self.trade_profits else 0.0

    @property
    def worst_trade(self) -> float:
        return min(self.trade_profits) if self.trade_profits else 0.0

    def record_trade(self, edge: float, delta: float):
        self.trades += 1
        self.edges.append(edge)
        self.deltas.append(delta)

    def record_result(self, profit: float, won: bool):
        if won:
            self.wins += 1
        else:
            self.losses += 1
        self.pnl += profit
        self.trade_profits.append(profit)

    def record_window(self, traded: bool):
        self.windows_seen += 1
        if not traded:
            self.windows_skipped += 1

    def reset(self):
        self.hour_start = time.time()
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.pnl = 0.0
        self.windows_seen = 0
        self.windows_skipped = 0
        self.edges.clear()
        self.deltas.clear()
        self.trade_profits.clear()

    def to_dict(self) -> dict:
        return {
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "pnl": self.pnl,
            "windows_seen": self.windows_seen,
            "windows_skipped": self.windows_skipped,
            "avg_edge": self.avg_edge,
            "avg_delta": self.avg_delta,
            "best_trade": self.best_trade,
            "worst_trade": self.worst_trade,
        }


@dataclass
class TradingStats:
    """Overall lifetime stats with embedded hourly tracker."""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    bankroll: float = 100.0
    hourly: HourlyStats = field(default_factory=HourlyStats)

    @property
    def win_rate(self) -> float:
        return (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0.0

    def record_win(self, profit: float):
        self.total_trades += 1
        self.wins += 1
        self.total_pnl += profit
        self.bankroll += profit
        self.hourly.record_result(profit, won=True)

    def record_loss(self, loss: float):
        self.total_trades += 1
        self.losses += 1
        self.total_pnl -= abs(loss)
        self.bankroll -= abs(loss)
        self.hourly.record_result(-abs(loss), won=False)

    def to_dict(self) -> dict:
        return {
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "pnl": self.total_pnl,
            "bankroll": self.bankroll,
        }


def kelly_bet_size(
    true_prob: float,
    market_price: float,
    bankroll: float,
    fraction: float = 0.25,
    min_bet: float = 1.0,
    max_bet: float = 25.0,
    fee_rate_bps: int = 0,
) -> float:
    """Calculate Kelly criterion bet size.

    Binary market: buy at market_price, win pays $1.
      b = (1 - market_price) / market_price   (net odds)
      kelly_f = (b * p - q) / b

    Uses fractional Kelly (default 0.25 = quarter Kelly) for safety.
    """
    if market_price <= 0 or market_price >= 1:
        return 0.0

    effective_price = market_price * (1 + fee_rate_bps / 10_000)
    if effective_price >= 1:
        return 0.0

    b = (1.0 - effective_price) / effective_price
    q = 1.0 - true_prob
    kelly_f = (b * true_prob - q) / b

    if kelly_f <= 0:
        return 0.0

    bet = bankroll * kelly_f * fraction
    return max(min(bet, max_bet), min_bet)


def estimate_true_probability(
    btc_delta_pct: float, seconds_remaining: float, vol: float = 0.12
) -> float:
    """Estimate true probability using Brownian motion model.

    vol defaults to 0.12 (calibrated static fallback). When the bot has
    enough recent window data, it passes a realized rolling vol instead,
    which adapts to the current regime (higher in volatile sessions,
    lower in trending sessions).
    """
    time_factor = max(seconds_remaining, 1) / 300
    effective_vol = vol * math.sqrt(time_factor)

    if effective_vol == 0:
        return 1.0 if btc_delta_pct > 0 else 0.0

    z_score = abs(btc_delta_pct) / effective_vol
    prob = 0.5 * (1 + math.erf(z_score / math.sqrt(2)))

    return min(max(prob, 0.01), 0.99)


def get_skip_reason(
    btc_price: float,
    opening_price: float,
    up_market_price: float,
    down_market_price: float,
    seconds_remaining: float,
    config: "StrategyConfig" = None,
    realized_vol: float = None,
) -> str:
    """Return why evaluate() returned None, for signal logging.

    Returns one of: "delta_too_small", "prob_below_min", "edge_below_min",
    "price_out_of_range", or "" (no skip reason — should have traded).
    "edge_gone_at_market" is set by the caller in bot.py after the live
    price re-check.
    """
    if config is None:
        config = StrategyConfig()
    if opening_price <= 0:
        return ""
    btc_delta_pct = ((btc_price - opening_price) / opening_price) * 100
    if abs(btc_delta_pct) < config.min_btc_delta:
        return "delta_too_small"
    side = "UP" if btc_delta_pct > 0 else "DOWN"
    market_price = up_market_price if btc_delta_pct > 0 else down_market_price
    if market_price > config.max_price or market_price < config.min_price:
        return "price_out_of_range"
    vol = realized_vol if realized_vol is not None else 0.12
    true_prob = estimate_true_probability(btc_delta_pct, seconds_remaining, vol=vol)
    if true_prob < config.min_prob:
        return "prob_below_min"
    edge = true_prob - market_price
    if edge < config.min_edge:
        return "edge_below_min"
    return ""


def evaluate(
    btc_price: float,
    opening_price: float,
    up_market_price: float,
    down_market_price: float,
    seconds_remaining: float,
    bankroll: float = 100.0,
    config: StrategyConfig = None,
    realized_vol: float = None,
    fee_rate_bps: int = 0,
) -> Optional[TradeSignal]:
    """Evaluate whether to enter a trade.

    Two-layer filter:
      1. Model probability must exceed min_prob (default 80%)
      2. Edge (prob - market_price) must exceed min_edge (default 5%)
      3. Fee-aware: net edge after taker fee must still exceed min_edge

    realized_vol: rolling std dev of recent window closing deltas.
    When None, falls back to the hardcoded 0.12 default.
    """
    if config is None:
        config = StrategyConfig()

    if seconds_remaining > config.entry_window_start:
        return None
    if seconds_remaining < config.entry_window_end:
        return None
    if opening_price <= 0:
        return None

    btc_delta_pct = ((btc_price - opening_price) / opening_price) * 100

    if abs(btc_delta_pct) < config.min_btc_delta:
        return None

    side = "UP" if btc_delta_pct > 0 else "DOWN"
    market_price = up_market_price if btc_delta_pct > 0 else down_market_price

    if market_price > config.max_price or market_price < config.min_price:
        return None

    vol = realized_vol if realized_vol is not None else 0.12
    true_prob = estimate_true_probability(btc_delta_pct, seconds_remaining, vol=vol)

    # Filter 1: Model must be confident enough
    if true_prob < config.min_prob:
        return None

    # Filter 2: Must have real edge over market price
    edge = true_prob - market_price

    # Filter 3: Fee-aware — taker fee reduces edge
    if fee_rate_bps > 0:
        fee_cost = market_price * (fee_rate_bps / 10_000)
        edge = edge - fee_cost

    if edge < config.min_edge:
        return None

    bet_size = kelly_bet_size(
        true_prob=true_prob,
        market_price=market_price,
        bankroll=bankroll,
        fraction=config.kelly_fraction,
        min_bet=config.min_bet,
        max_bet=config.max_bet,
        fee_rate_bps=fee_rate_bps,
    )

    confidence = min(edge / 0.10, 1.0)

    return TradeSignal(
        side=side,
        confidence=confidence,
        btc_delta_pct=btc_delta_pct,
        market_price=market_price,
        edge=edge,
        true_prob=true_prob,
        seconds_remaining=seconds_remaining,
        kelly_size=bet_size,
    )


# ── Composite Weighted Signal (Archetapp 7-indicator model) ──────────

@dataclass
class CompositeSignal:
    """Result from composite_weighted_signal()."""
    score: float          # positive = UP, negative = DOWN
    side: str             # "UP" or "DOWN"
    confidence: float     # 0.0 to 1.0
    components: Dict[str, float]  # individual indicator scores
    spike: bool = False   # True if score jumped >= 1.5 since last check


def _window_delta_score(btc_price: float, opening_price: float) -> float:
    """Indicator 1: Window Delta (weight 5-7). THE dominant signal.
    
    Directly answers the market's question: 'Is BTC up or down vs window open?'
    At T-10s, if BTC is already up 0.10%+, reversal in 10s is almost impossible.
    """
    if opening_price <= 0:
        return 0.0
    delta_pct = (btc_price - opening_price) / opening_price * 100
    sign = 1.0 if delta_pct > 0 else -1.0 if delta_pct < 0 else 0.0
    abs_delta = abs(delta_pct)

    if abs_delta > 0.10:
        return sign * 7.0   # Decisive — nearly certain
    elif abs_delta > 0.02:
        return sign * 5.0   # Strong
    elif abs_delta > 0.005:
        return sign * 3.0   # Moderate
    elif abs_delta > 0.001:
        return sign * 1.0   # Slight
    return 0.0


def _micro_momentum_score(klines: List[List]) -> float:
    """Indicator 2: Micro Momentum (weight 2). Last 2 candle direction.
    
    Each kline: [open_time, open, high, low, close, volume, close_time, ...]
    """
    if len(klines) < 2:
        return 0.0
    c1 = float(klines[-2][4])  # close of previous candle
    c2 = float(klines[-1][4])  # close of current candle
    if c1 <= 0:
        return 0.0
    pct = (c2 - c1) / c1 * 100
    if abs(pct) > 0.05:
        return 2.0 if pct > 0 else -2.0
    return 0.0


def _acceleration_score(klines: List[List]) -> float:
    """Indicator 3: Acceleration (weight 1.5). Is momentum building or fading?
    
    Compares momentum (close-to-close %) of the last 2 candle pairs.
    """
    if len(klines) < 3:
        return 0.0
    c_prev = float(klines[-3][4])
    c_mid = float(klines[-2][4])
    c_curr = float(klines[-1][4])
    if c_prev <= 0 or c_mid <= 0:
        return 0.0
    mom1 = (c_mid - c_prev) / c_prev * 100
    mom2 = (c_curr - c_mid) / c_mid * 100
    accel = mom2 - mom1
    if abs(accel) > 0.03:
        return 1.5 if accel > 0 else -1.5
    return 0.0


def _ema(values: List[float], period: int) -> float:
    """Exponential Moving Average."""
    if not values:
        return 0.0
    if len(values) < period:
        return sum(values) / len(values)
    k = 2.0 / (period + 1)
    ema_val = values[0]
    for v in values[1:]:
        ema_val = v * k + ema_val * (1 - k)
    return ema_val


def _ema_crossover_score(klines: List[List]) -> float:
    """Indicator 4: EMA Crossover 9/21 (weight 1). Short-term trend.
    
    EMA9 > EMA21 = bullish (UP), EMA9 < EMA21 = bearish (DOWN).
    """
    if len(klines) < 21:
        return 0.0
    closes = [float(k[4]) for k in klines]
    ema9 = _ema(closes, 9)
    ema21 = _ema(closes, 21)
    if ema21 <= 0:
        return 0.0
    diff_pct = (ema9 - ema21) / ema21 * 100
    if abs(diff_pct) > 0.02:
        return 1.0 if diff_pct > 0 else -1.0
    return 0.0


def _rsi_score(klines: List[List]) -> float:
    """Indicator 5: RSI 14-period (weight 1-2). Overbought/oversold extremes.
    
    Only signals at extremes (>75 overbought, <25 oversold).
    """
    if len(klines) < 15:
        return 0.0
    closes = [float(k[4]) for k in klines]
    changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [max(c, 0) for c in changes[-14:]]
    losses = [-min(c, 0) for c in changes[-14:]]
    avg_gain = sum(gains) / 14
    avg_loss = sum(losses) / 14
    if avg_loss == 0:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

    if rsi > 75:
        return -2.0   # Overbought → expect DOWN
    elif rsi > 70:
        return -1.0
    elif rsi < 25:
        return 2.0    # Oversold → expect UP
    elif rsi < 30:
        return 1.0
    return 0.0


def _volume_surge_score(klines: List[List]) -> float:
    """Indicator 6: Volume Surge (weight 1). Confirms current direction.
    
    Recent 3-bar avg volume 1.5x prior average → volume confirms direction.
    """
    if len(klines) < 6:
        return 0.0
    vols = [float(k[5]) for k in klines]
    recent_avg = sum(vols[-3:]) / 3
    prior_avg = sum(vols[-6:-3]) / 3
    if prior_avg <= 0:
        return 0.0
    ratio = recent_avg / prior_avg
    if ratio >= 1.5:
        # Volume surge — confirm current candle direction
        c1 = float(klines[-1][4])
        o1 = float(klines[-1][1])
        return 1.0 if c1 > o1 else -1.0
    return 0.0


def _tick_trend_score(tick_prices: List[float]) -> float:
    """Indicator 7: Real-Time Tick Trend (weight 2). Micro-trends from 2s polling.
    
    Uses linear regression slope on tick prices to detect micro-trend direction.
    """
    if len(tick_prices) < 3:
        return 0.0
    n = len(tick_prices)
    x_mean = (n - 1) / 2
    y_mean = sum(tick_prices) / n
    num = sum((i - x_mean) * (p - y_mean) for i, p in enumerate(tick_prices))
    den = sum((i - x_mean) ** 2 for i in range(n))
    if den == 0:
        return 0.0
    slope = num / den
    if y_mean <= 0:
        return 0.0
    slope_pct = slope / y_mean * 100
    if abs(slope_pct) > 0.005:
        return 2.0 if slope_pct > 0 else -2.0
    return 0.0


def composite_weighted_signal(
    btc_price: float,
    opening_price: float,
    klines: List[List],
    tick_prices: List[float],
    prev_score: float = 0.0,
) -> CompositeSignal:
    """Composite weighted signal from 7 indicators (Archetapp model).

    Positive score = UP, negative = DOWN.
    Confidence = min(abs(score) / 7.0, 1.0) — divides by 7 for 5-min relevance.
    Spike detection: |score - prev_score| >= 1.5 means immediate trade signal.

    Args:
        btc_price: Current BTC price
        opening_price: BTC price at window open (from kline or first tick)
        klines: List of 1-min klines [[open_time, O, H, L, C, V, ...], ...]
        tick_prices: List of recent tick prices (2s apart) for micro-trend
        prev_score: Score from previous analysis check (for spike detection)

    Returns:
        CompositeSignal with score, side, confidence, components, spike flag.
    """
    s1 = _window_delta_score(btc_price, opening_price)
    s2 = _micro_momentum_score(klines)
    s3 = _acceleration_score(klines)
    s4 = _ema_crossover_score(klines)
    s5 = _rsi_score(klines)
    s6 = _volume_surge_score(klines)
    s7 = _tick_trend_score(tick_prices)

    score = s1 + s2 + s3 + s4 + s5 + s6 + s7
    abs_score = abs(score)

    if score > 0:
        side = "UP"
    elif score < 0:
        side = "DOWN"
    else:
        side = ""  # No signal

    confidence = min(abs_score / 7.0, 1.0)
    spike = abs(score - prev_score) >= 1.5

    return CompositeSignal(
        score=score,
        side=side,
        confidence=confidence,
        components={
            "window_delta": s1,
            "micro_momentum": s2,
            "acceleration": s3,
            "ema_crossover": s4,
            "rsi": s5,
            "volume_surge": s6,
            "tick_trend": s7,
        },
        spike=spike,
    )
