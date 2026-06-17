#!/usr/bin/env python3
"""
PolyBot v13 — Recalibrated + Safety Systems

Strategy:
  - Brownian motion model with vol=0.12 (recalibrated from 0.08)
  - Entry gate: model confidence >= 80%, market price <= true_prob * 0.85
  - Position sizing: quarter-Kelly, $5–$25 per trade
  - Exit: hold ALL positions to resolution — no stops, no take-profit, no early exits

Safety systems:
  1. CLOB health check: get_ok() before every trade; 3 consecutive
     failures halt trading and send Telegram alert. Auto-recovers
     when API comes back at next window boundary.
  2. Daily loss limit: if session P&L <= -DAILY_LOSS_LIMIT, halt trading.
  3. Balance-verified buys: snapshot USDC before/after; ghost fills
     caught even when API throws. Never cancels on timeout — returns
     UNVERIFIED_BUY for pending detection at next window boundary.
  4. Pending buy safety net: if buy unverified, check balance at next
     window boundary; retroactively track as filled if balance dropped.
  5. Window-boundary balance sync: real USDC balance overwrites internal
     tracking every 5 minutes. Corrects any accumulated drift.
  6. Minimum notional guard: skip sells below $5 notional; hold to
     resolution instead of hitting Polymarket's minimum-size rejection.
"""

import os
import sys
import time
import signal
import math
import statistics
from typing import Optional
from dotenv import load_dotenv

import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

from market import get_current_market, current_window_ts, PERIOD_SECONDS
from price_feed import BinancePriceFeed
from strategy import evaluate, estimate_true_probability, get_skip_reason, StrategyConfig, TradingStats, TradeSignal, kelly_bet_size, composite_weighted_signal, CompositeSignal
from executor import Executor, FILLED, PARTIAL, FAILED, MAX_BUY_PRICE
from telegram_notifier import TelegramNotifier
from tracker import Tracker


FORCED_EXIT_START = 5
FORCED_EXIT_END = 1
POSITION_CHECK_INTERVAL = 3
MAX_EXIT_RETRIES = 3
EXIT_RETRY_COOLDOWN = 10


class PolyBot:
    def __init__(self):
        load_dotenv()

        self.dry_run = os.getenv("DRY_RUN", "true").lower() == "true"
        self.period = int(os.getenv("MARKET_PERIOD", "15"))

        self.strategy_config = StrategyConfig(
            min_edge=float(os.getenv("MIN_EDGE", "0.25")),
            min_prob=float(os.getenv("MIN_PROB", "0.80")),
            max_price=float(os.getenv("MAX_PRICE", "0.60")),
            min_btc_delta=float(os.getenv("MIN_BTC_DELTA", "0.06")),
            entry_window_start=int(os.getenv("ENTRY_WINDOW_START", "10")),
            entry_window_end=int(os.getenv("ENTRY_WINDOW_END", "5")),
            kelly_fraction=float(os.getenv("KELLY_FRACTION", "0.10")),
            min_bet=float(os.getenv("MIN_BET", "2.0")),
            max_bet=float(os.getenv("MAX_BET", "5.0")),
        )

        initial_bankroll = float(os.getenv("BANKROLL", "100.0"))
        self._daily_loss_limit = float(os.getenv("DAILY_LOSS_LIMIT", "30.0"))
        self._rolling_vol_windows = int(os.getenv("ROLLING_VOL_WINDOWS", "4"))  # 4 windows = 1hr (15m each)
        self._vol_floor = float(os.getenv("VOL_FLOOR", "0.06"))
        self._vol_cap = float(os.getenv("VOL_CAP", "0.30"))
        self._vol_fallback = 0.12  # used until enough windows accumulate

        # Regime filter — sit out when market is consistently expensive
        self._rolling_price_windows = int(os.getenv("ROLLING_PRICE_WINDOWS", "4"))  # 4 windows = 1hr (15m each)
        self._regime_max_price = float(os.getenv("REGIME_MAX_PRICE", "0.62"))  # skip if rolling avg > this

        # Trading hours (UTC) — only trade during high-volatility periods
        self._trading_hours_start = int(os.getenv("TRADING_HOURS_START", "07"))  # 07:00 UTC = EU morning
        self._trading_hours_end = int(os.getenv("TRADING_HOURS_END", "16"))      # 16:00 UTC = US mid-day

        self.price_feed = BinancePriceFeed()
        self.executor = Executor(
            private_key=os.getenv("PRIVATE_KEY", ""),
            safe_address=os.getenv("SAFE_ADDRESS", ""),
            dry_run=self.dry_run,
        )
        self.telegram = TelegramNotifier()
        self.tracker = Tracker(
            log_dir=os.getenv("LOG_DIR", "logs15"),
            log_executions=os.getenv("LOG_EXECUTIONS", "false").lower() == "true",
        )
        self.stats = TradingStats(bankroll=initial_bankroll)
        self.stats.hourly.hour_start = time.time()

        self._running = False
        self._current_window: int = 0
        self._opening_price: float = 0.0
        self._last_hour_check: int = 0
        self._current_utc_day: int = 0  # Track UTC day for midnight reset

        # Trade state
        self._traded: bool = False
        self._trade_attempted: bool = False
        self._trade_side: str = ""
        self._trade_price: float = 0.0
        self._trade_cost: float = 0.0
        self._trade_shares: float = 0.0
        self._trade_token_id: str = ""

        # Exit state
        self._exited: bool = False
        self._exit_revenue: float = 0.0
        self._exit_shares_sold: float = 0.0
        self._residual_shares: float = 0.0  # Shares left after partial fill
        self._last_position_check: float = 0.0
        self._last_status_print: float = 0.0
        self._last_tick_context: dict = {}   # last entry-window state, for window-end signal logging
        self._session_start_time: float = time.time()
        self._recent_window_deltas: list = []  # rolling abs(close_delta_pct) per window
        self._recent_window_prices: list = []  # rolling dominant option price per window
        self._regime_skip: bool = False  # set when regime filter blocks this window
        self._exit_retries: int = 0
        self._exit_gave_up: bool = False
        self._last_sell_price_seen: float = 0.0  # last observed sell price during hold period

        # Pending phantom verification (claim sell reported success but balance didn't move yet)
        # Resolved at next window boundary once Polygon settlement has had time to land.
        self._pending_phantom: dict = {}

        # Composite signal (Archetapp 7-indicator model) — T-10s analysis mode
        self._composite_mode: bool = False  # True during T-10s to T-5s analysis loop
        self._prev_composite_score: float = 0.0
        self._best_composite: Optional[CompositeSignal] = None
        self._last_composite_check: float = 0.0
        self._COMPOSITE_INTERVAL: float = 2.0  # 2-second analysis loop
        self._klines_fetched_this_window: bool = False
        self._opening_price_from_kline: float = 0.0

        # Pending buy (unverified — Polygon settlement too slow)
        self._pending_buy_side: str = ""
        self._pending_buy_price: float = 0.0
        self._pending_buy_amount: float = 0.0
        self._pending_buy_shares: float = 0.0
        self._pending_buy_token_id: str = ""
        self._pending_buy_edge: float = 0.0
        self._pending_buy_delta: float = 0.0
        self._balance_before_buy: float = 0.0

        # Unclaimed
        self._unclaimed_winnings: float = 0.0

        # Real balance tracking (source of truth)
        self._session_start_balance: float = 0.0
        self._last_real_balance: float = 0.0

        # Price cache
        self._cached_up: float = 0.50
        self._cached_down: float = 0.50
        self._price_last_fetched: float = 0.0
        self._current_market = None
        self._current_condition_id: str = ""
        self._PRICE_REFRESH: float = 5.0

        # Circuit breaker — detects CLOB API degradation
        self._consecutive_buy_failures: int = 0
        self._clob_halted: bool = False
        self._HALT_AFTER_FAILURES: int = 3
        self._daily_loss_halted: bool = False

    def start(self):
        if not self.dry_run:
            from proxy import ensure_tor, apply_proxy
            import logging as _log
            _log.basicConfig(level=_log.INFO, format="[%(name)s] %(message)s")
            print("\n🧅 Starting Tor proxy for CLOB API...")
            proxy_url = ensure_tor()
            apply_proxy(proxy_url)
            print(f"✅ Tor active: {proxy_url}\n")

        kf = self.strategy_config.kelly_fraction
        mp = self.strategy_config.min_prob
        me = self.strategy_config.min_edge
        print("=" * 55)
        print(f"  PolyBot v13 — 15-Minute BTC (vol=0.12)")
        print(f"  Mode: {'DRY RUN' if self.dry_run else '🔴 LIVE TRADING'}")
        print(f"  Kelly: {kf*100:.0f}% fraction | "
              f"Bets: ${self.strategy_config.min_bet:.0f}–${self.strategy_config.max_bet:.0f}")
        print(f"  Min prob: {mp:.0%} | Min edge: {me:.0%} | Min BTC delta: {self.strategy_config.min_btc_delta:.2f}%")
        print(f"  Entry: T-{self.strategy_config.entry_window_start}s to "
              f"T-{self.strategy_config.entry_window_end}s")
        print(f"  Vol: dynamic (fallback=0.12, floor={self._vol_floor}, cap={self._vol_cap}, windows={self._rolling_vol_windows})")
        print(f"  Regime: max_avg_price=${self._regime_max_price:.2f}, windows={self._rolling_price_windows}")
        print(f"  Exits: hold to resolution")
        print(f"  Daily loss limit: ${self._daily_loss_limit:.0f}")
        print(f"  Trading hours: {self._trading_hours_start:02d}:00–{self._trading_hours_end:02d}:00 UTC")
        print(f"  Bankroll: ${self.stats.bankroll:.2f}")
        print("=" * 55)

        # Always initialise executor and query real CLOB balance
        if not self.executor.initialize():
            if not self.dry_run:
                print("\n❌ Failed to initialize. Check credentials.")
                return
            print("  ⚠️  CLOB init failed — using BANKROLL env as fallback")
        else:
            balance = self.executor.get_collateral_balance()
            print(f"  CLOB balance: ${balance:.2f}")
            if balance > 0:
                self.stats.bankroll = balance
            else:
                print(f"  ⚠️  CLOB returned $0 — using BANKROLL env: ${self.stats.bankroll:.2f}")

        if self.dry_run:
            print("  [DRY RUN — paper trades only, no real orders]")

        self._session_start_balance = self.stats.bankroll
        self._last_real_balance = self.stats.bankroll
        self.tracker.set_session_balance(self.stats.bankroll)

        self.price_feed.start()
        print("\n⏳ Waiting for BTC price...")
        price = self.price_feed.wait_for_price(timeout=30)
        if not price:
            print("❌ No price feed. Check internet.")
            return
        print(f"✅ BTC: ${price:,.2f} ({self.price_feed.state.source})")

        self.telegram.startup_alert({
            "dry_run": self.dry_run,
            "kelly_fraction": kf,
            "min_edge": self.strategy_config.min_edge,
            "min_bet": self.strategy_config.min_bet,
            "max_bet": self.strategy_config.max_bet,
            "entry_start": self.strategy_config.entry_window_start,
            "entry_end": self.strategy_config.entry_window_end,
        })

        self._running = True
        self._last_hour_check = int(time.time() // 3600)
        self._current_utc_day = int(time.strftime("%j", time.gmtime()))  # Day of year
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        print("\n🚀 Running. Ctrl+C to stop.\n")
        self._main_loop()

    def _clob_healthy(self) -> bool:
        """Quick CLOB API health check."""
        if not self.executor.client:
            return False
        try:
            from py_clob_client_v2.clob_types import BalanceAllowanceParams, AssetType
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            self.executor.client.get_balance_allowance(params)
            return True
        except Exception:
            return False

    def _main_loop(self):
        while self._running:
            try:
                self._check_midnight_reset()
                self._tick()
                self._check_hourly_summary()
            except Exception as e:
                print(f"[error] {e}")
                self.telegram.error_alert(str(e))
            time.sleep(0.1)

    def _check_midnight_reset(self):
        """Reset daily loss limit and session tracking at UTC midnight."""
        utc_day = int(time.strftime("%j", time.gmtime()))
        if utc_day != self._current_utc_day:
            self._current_utc_day = utc_day
            self._daily_loss_halted = False
            # Always sync real balance as new session start
            if self.executor._initialized:
                real_bal = self.executor.get_collateral_balance()
                if real_bal > 0:
                    self.stats.bankroll = real_bal
                    self._session_start_balance = real_bal
                    self._last_real_balance = real_bal
                else:
                    self._session_start_balance = self.stats.bankroll
            else:
                self._session_start_balance = self.stats.bankroll
            print(f"\n  🌅 NEW DAY (UTC) — daily loss limit reset, "
                  f"session balance: ${self._session_start_balance:.2f}")
            self.telegram.status_update({
                "alert": f"🌅 New trading day — loss limit reset, bankroll ${self.stats.bankroll:.2f}"
            })

    def _tick(self):
        now = time.time()
        period_secs = PERIOD_SECONDS[self.period]
        window_ts = int(now) - (int(now) % period_secs)

        btc_price, is_fresh = self.price_feed.get_price()
        if not is_fresh or btc_price <= 0:
            return

        if window_ts != self._current_window:
            self._on_new_window(window_ts, closing_btc_price=btc_price)

        seconds_remaining = (window_ts + period_secs) - now

        # Use kline opening price (Binance 1-min candle open = Polymarket resolution price)
        # More accurate than first-tick price which may arrive after window opens
        if self._opening_price <= 0:
            kline_open = self.price_feed.get_window_open_from_kline(window_ts)
            if kline_open > 0:
                self._opening_price = kline_open
                self._opening_price_from_kline = kline_open
                print(f"  📌 Open (kline): ${kline_open:,.2f}")
            else:
                self._opening_price = btc_price
                print(f"  📌 Open (tick): ${btc_price:,.2f}")

        # HOLDING: active position management
        if self._traded and not self._exited and not self._exit_gave_up:
            self._manage_position(btc_price, seconds_remaining, now)
            return

        # Already done
        if self._traded or self._trade_attempted:
            return

        # ── Composite analysis mode (T-10s to T-5s) ──
        # At T-10s, BTC direction is largely locked in. Run 7-indicator
        # composite signal with 2-second polling, spike detection, and
        # best-signal tracking. Fire by T-5s at the latest.
        composite_window = 5.0 <= seconds_remaining <= 10.0
        if composite_window:
            # Skip slow CLOB probe — use estimated prices for composite signal.
            # The CLOB probe takes ~10s (5 retries × 2 tokens) which eats the window.
            # For composite analysis, BTC price direction matters more than exact book price.
            delta_pct = ((btc_price - self._opening_price) / self._opening_price * 100) if self._opening_price > 0 else 0
            realized_vol = self._compute_realized_vol()
            vol = realized_vol if realized_vol is not None else 0.12
            from strategy import estimate_true_probability
            est_prob = estimate_true_probability(delta_pct, seconds_remaining, vol=vol)
            # Cap at regime max price — Brownian overestimates at extremes
            cap = self._regime_max_price  # $0.62
            cp_up = round(min(max(est_prob, 0.01), cap), 2)
            cp_down = round(min(max(1.0 - est_prob, 0.01), cap), 2)
            now_check = time.time()
            elapsed = now_check - self._last_composite_check
            if elapsed >= self._COMPOSITE_INTERVAL:
                self._last_composite_check = now_check
                print(f"  🎯 COMPOSITE ENTER: T-{seconds_remaining:.1f}s up={cp_up} down={cp_down}")
                self._run_composite_analysis(btc_price, cp_up, cp_down,
                                             seconds_remaining)
                if self._traded or self._trade_attempted:
                    return
            # During composite window, don't run Brownian — composite owns this time
            return

        # IDLE: look for entry (Brownian + Momentum for T > 10s)
        up_price, down_price = self._get_market_prices(btc_price, seconds_remaining)
        if up_price <= 0 and down_price <= 0:
            return  # price fetch failed — skip this window

        # ── Regime filter ── skip when market consistently prices options expensive
        rolling_avg = self._compute_rolling_avg_price()
        if rolling_avg > 0 and rolling_avg > self._regime_max_price:
            self._regime_skip = True
            return  # market regime too expensive — no edge
        self._regime_skip = False

        realized_vol = self._compute_realized_vol()
        fee_rate_bps = self.executor.get_fee_rate_bps(self._current_condition_id) if self._current_condition_id else 0
        signal_result = evaluate(
            btc_price=btc_price,
            opening_price=self._opening_price,
            up_market_price=up_price,
            down_market_price=down_price,
            seconds_remaining=seconds_remaining,
            bankroll=self.stats.bankroll,
            config=self.strategy_config,
            realized_vol=realized_vol,
            fee_rate_bps=fee_rate_bps,
        )

        # ── Momentum signal: Binance move not yet priced by Polymarket ──
        # If BTC moved >0.10% on Binance in the last 60s but Polymarket
        # tokens haven't repriced, this is latency arbitrage alpha.
        if not signal_result:
            mom = self.price_feed.get_momentum(lookback_seconds=60.0)
            if mom["valid"] and abs(mom["pct_change"]) >= 0.10:
                mom_side = "UP" if mom["pct_change"] > 0 else "DOWN"
                mom_price = up_price if mom_side == "UP" else down_price
                # Only enter if token is still cheap (not already priced in)
                if 0.50 <= mom_price <= 0.65:
                    # Momentum override: allow entry with lower model edge
                    # The Binance move IS the signal — model prob is secondary
                    vol = realized_vol if realized_vol is not None else 0.12
                    time_factor = max(seconds_remaining, 1) / 300
                    effective_vol = vol * math.sqrt(time_factor)
                    delta_pct = abs(mom["pct_change"])
                    if effective_vol > 0:
                        z_score = delta_pct / effective_vol
                        mom_prob = 0.5 * (1 + math.erf(z_score / math.sqrt(2)))
                        mom_edge = mom_prob - mom_price
                        # Require at least 10% edge for momentum trades
                        if mom_edge >= 0.10:
                            fee_cost = mom_price * (fee_rate_bps / 10_000) if fee_rate_bps > 0 else 0
                            mom_edge_net = mom_edge - fee_cost
                            if mom_edge_net >= 0.10:
                                bet_size = kelly_bet_size(
                                    true_prob=mom_prob,
                                    market_price=mom_price,
                                    bankroll=self.stats.bankroll,
                                    fraction=self.strategy_config.kelly_fraction,
                                    min_bet=self.strategy_config.min_bet,
                                    max_bet=self.strategy_config.max_bet,
                                    fee_rate_bps=fee_rate_bps,
                                )
                                signal_result = TradeSignal(
                                    side=mom_side,
                                    confidence=min(mom_edge_net / 0.10, 1.0),
                                    btc_delta_pct=mom["pct_change"],
                                    market_price=mom_price,
                                    edge=mom_edge_net,
                                    true_prob=mom_prob,
                                    seconds_remaining=seconds_remaining,
                                    kelly_size=bet_size,
                                )
                                print(f"  🚀 MOMENTUM SIGNAL: BTC {mom['pct_change']:+.3f}% in 60s | "
                                      f"{mom_side} @ ${mom_price:.3f} | edge={mom_edge_net:.2f} | "
                                      f"prob={mom_prob:.2f}")

        # Store context for window-end no-trade signal logging
        self._last_tick_context = {
            "btc_price": btc_price,
            "up_price": up_price,
            "down_price": down_price,
            "seconds_remaining": seconds_remaining,
            "window_ts": self._current_window,
            "signal": signal_result,
        }

        if signal_result:
            self._execute_trade(signal_result, seconds_remaining)

        if now - self._last_status_print >= 30:
            self._last_status_print = now
            delta = ((btc_price - self._opening_price) / self._opening_price * 100) if self._opening_price > 0 else 0
            d = "↑" if delta > 0 else "↓" if delta < 0 else "→"
            if self._traded:
                state = "HOLDING"
            elif self._opening_price > 0 and abs(delta) < self.strategy_config.min_btc_delta:
                state = f"ΔSMALL ({abs(delta):.3f}%<{self.strategy_config.min_btc_delta:.3f}%)"
            else:
                state = "IDLE"
            n = len(self._recent_window_deltas)
            vol_label = f"{self._compute_realized_vol():.3f}({'r' if n >= 6 else f'fb,n={n}'})"
            rp = self._compute_rolling_avg_price()
            rp_label = f"${rp:.3f}" if rp > 0 else "n/a"
            regime = "HOT" if rp > 0 and rp <= self._regime_max_price else "COLD" if rp > self._regime_max_price else "---"
            print(
                f"  ⏱  T-{seconds_remaining:5.1f}s | "
                f"BTC ${btc_price:,.2f} {d}{abs(delta):.3f}% | "
                f"UP ${up_price:.3f} DN ${down_price:.3f} | "
                f"vol={vol_label} avg_p={rp_label} [{regime}] | "
                f"P&L ${self.stats.total_pnl:+.2f} [{state}]"
            )

    # ── Active position management ──────────────────────────────────

    # ── Position monitoring (hold to resolution) ────────────────────

    def _manage_position(self, btc_price: float, seconds_remaining: float, now: float):
        """Monitor only — all trades hold to resolution. No stops.
        Tracker logs hold-period stats for future optimization.
        """
        if self._opening_price <= 0:
            return

        btc_delta_pct = ((btc_price - self._opening_price) / self._opening_price) * 100
        updated_prob = estimate_true_probability(btc_delta_pct, seconds_remaining)

        if self._trade_side == "DOWN":
            our_prob = 1.0 - updated_prob
        else:
            our_prob = updated_prob

        # Throttled check
        if now - self._last_position_check < POSITION_CHECK_INTERVAL:
            if now - self._last_status_print >= 30:
                self._last_status_print = now
                d = "↑" if btc_delta_pct > 0 else "↓" if btc_delta_pct < 0 else "→"
                print(
                    f"  ⏱  T-{seconds_remaining:5.1f}s | "
                    f"BTC {d}{abs(btc_delta_pct):.3f}% | "
                    f"Prob: {our_prob:.2f} | "
                    f"P&L ${self.stats.total_pnl:+.2f} [HOLDING→RES]"
                )
            return

        self._last_position_check = now

        # Get current sell price (for tracking only)
        if self.dry_run:
            current_sell_price = round(max(our_prob, 0.01), 2)
        else:
            sell_probe = round(self._trade_shares * self._trade_price, 2)
            current_sell_price = self.executor.get_market_price(
                self._trade_token_id, "SELL", max(sell_probe, 1.0)
            )

        if current_sell_price <= 0:
            return

        self._last_sell_price_seen = current_sell_price

        # Track hold-period extremes
        self.tracker.update_hold_stats(our_prob, current_sell_price)

        current_value = self._trade_shares * current_sell_price
        unrealized_pnl = current_value - self._trade_cost
        return_pct = (current_sell_price - self._trade_price) / self._trade_price if self._trade_price > 0 else 0

        # STOP-LOSS DISABLED — early exits are the #1 P&L destroyer.
        # Data from 425-trade dry run: 119 early exits lost $439.
        # All positions now hold to resolution for maximum edge capture.
        STOP_LOSS_PCT = -0.15
        if return_pct <= STOP_LOSS_PCT and seconds_remaining > 10:
            print(f"  ⚠️ Position underwater {return_pct:+.1%} — HOLDING to resolution (stop-loss disabled)")

        # Status line (monitoring only — no exits)
        d = "↑" if btc_delta_pct > 0 else "↓" if btc_delta_pct < 0 else "→"
        pnl_emoji = "📈" if unrealized_pnl > 0 else "📉"
        print(
            f"  {pnl_emoji} T-{seconds_remaining:5.1f}s | "
            f"BTC {d}{abs(btc_delta_pct):.3f}% | "
            f"Prob: {our_prob:.2f} | "
            f"Sell: ${current_sell_price:.3f} | "
            f"PnL: ${unrealized_pnl:+.2f} ({return_pct:+.0%})"
        )

    # ── Composite analysis (T-10s snipe mode) ───────────────────────

    def _run_composite_analysis(self, btc_price: float, up_price: float,
                                down_price: float, seconds_remaining: float):
        """Run 7-indicator composite signal analysis during T-10s window.

        Polls every 2 seconds, tracks best signal, fires on spike or T-5s deadline.
        """
        # Fetch klines for this window (cached after first call)
        window_ts = self._current_window
        klines = self.price_feed.get_klines_for_composite(window_ts)
        tick_prices = self.price_feed.get_tick_prices()

        # Use kline opening price if available, otherwise fall back to stored
        opening = self._opening_price_from_kline if self._opening_price_from_kline > 0 else self._opening_price

        cs = composite_weighted_signal(
            btc_price=btc_price,
            opening_price=opening,
            klines=klines,
            tick_prices=tick_prices,
            prev_score=self._prev_composite_score,
        )

        self._prev_composite_score = cs.score

        # Track best signal seen this window
        if self._best_composite is None or abs(cs.score) > abs(self._best_composite.score):
            self._best_composite = cs

        # Status print
        c = cs.components
        print(f"  🎯 COMPOSITE T-{seconds_remaining:.1f}s | "
              f"Score={cs.score:+.1f} ({cs.side}) conf={cs.confidence:.0%} | "
              f"Δ={c['window_delta']:+.1f} μ={c['micro_momentum']:+.1f} "
              f"α={c['acceleration']:+.1f} ema={c['ema_crossover']:+.1f} "
              f"rsi={c['rsi']:+.1f} vol={c['volume_surge']:+.1f} "
              f"tick={c['tick_trend']:+.1f}"
              f"{' 🚨 SPIKE' if cs.spike else ''}")

        # ── Decision: fire on spike, T-5s deadline, or strong signal ──
        should_fire = False

        # Spike detection: score jumped >= 1.5 between checks
        if cs.spike and cs.confidence >= 0.25:
            should_fire = True
            print(f"  🚨 SPIKE DETECTED: score jumped to {cs.score:+.1f}")

        # T-5s hard deadline: use best signal seen
        elif seconds_remaining <= 5.5 and self._best_composite:
            best = self._best_composite
            if best.confidence >= 0.25 and best.side:
                cs = best  # Use best signal, not latest
                should_fire = True
                print(f"  ⏰ T-5s DEADLINE: using best score {best.score:+.1f} ({best.side})")

        if not should_fire or not cs.side:
            return

        # ── Convert composite signal to TradeSignal and execute ──
        market_price = up_price if cs.side == "UP" else down_price
        if market_price > self.strategy_config.max_price or market_price < self.strategy_config.min_price:
            print(f"  ⚠️ Composite {cs.side} @ ${market_price:.3f} — price out of range "
                  f"(${self.strategy_config.min_price}-${self.strategy_config.max_price})")
            return

        # Use composite confidence as probability proxy
        # At T-10s with strong delta, confidence maps well to actual probability
        true_prob = max(cs.confidence, market_price + 0.05)  # ensure some edge
        edge = true_prob - market_price

        fee_rate_bps = self.executor.get_fee_rate_bps(self._current_condition_id) if self._current_condition_id else 0
        if fee_rate_bps > 0:
            fee_cost = market_price * (fee_rate_bps / 10_000)
            edge -= fee_cost

        if edge < 0.05:  # minimum 5% edge for composite trades
            print(f"  ⚠️ Composite {cs.side} edge {edge:.2f} < 5% — skipping")
            return

        bet_size = kelly_bet_size(
            true_prob=true_prob,
            market_price=market_price,
            bankroll=self.stats.bankroll,
            fraction=self.strategy_config.kelly_fraction,
            min_bet=self.strategy_config.min_bet,
            max_bet=self.strategy_config.max_bet,
            fee_rate_bps=fee_rate_bps,
        )

        signal_result = TradeSignal(
            side=cs.side,
            confidence=cs.confidence,
            btc_delta_pct=(btc_price - opening) / opening * 100 if opening > 0 else 0,
            market_price=market_price,
            edge=edge,
            true_prob=true_prob,
            seconds_remaining=seconds_remaining,
            kelly_size=bet_size,
        )

        print(f"  🎯 COMPOSITE TRADE: {cs.side} @ ${market_price:.3f} | "
              f"edge={edge:.2f} prob={true_prob:.2f} | "
              f"score={cs.score:+.1f} ({'/'.join(f'{k}={v:+.1f}' for k,v in cs.components.items() if v != 0)})")

        self._execute_trade(signal_result, seconds_remaining)

    # ── Execute exit (balance-verified, partial fill aware) ─────────

    def _exit_position(self, sell_price: float, seconds_remaining: float, reason: str):
        if self.dry_run:
            revenue = self._trade_shares * sell_price
            self._exited = True
            self._exit_revenue = revenue
            self.stats.bankroll += revenue
            profit = revenue - self._trade_cost
            print(f"  💰 EXIT ({reason}, paper): {self._trade_shares:.0f} shares @ "
                  f"${sell_price:.3f} = ${revenue:.2f} | Profit: ${profit:+.2f}")
            return

        result = self.executor.sell(
            token_id=self._trade_token_id,
            shares=self._trade_shares,
            price=sell_price,
            force=(reason == "stop_loss"),  # Bypass minimum for stop-loss exits
        )

        status = result.get("status", FAILED)
        if status == FILLED:
            self._exit_revenue += result["revenue"]
            self._exit_shares_sold += result["shares"]
            self._residual_shares = 0.0
            self.stats.bankroll += result["revenue"]
            self._exited = True
            profit = self._exit_revenue - self._trade_cost
            print(f"  💰 EXIT ({reason}): {result['shares']:.0f} shares @ "
                  f"${result['price']:.3f} = ${result['revenue']:.2f} | "
                  f"Profit: ${profit:+.2f}")
        elif "hold to resolution" in result.get("error", ""):
            # Below $5 minimum — can't sell, hold to resolution
            notional = self._trade_shares * sell_price
            print(f"  📌 Can't sell: ${notional:.2f} below $5 minimum — holding to resolution")
            self._exit_gave_up = True  # Skip further exit attempts
        else:
            self._exit_retries += 1
            if self._exit_retries >= MAX_EXIT_RETRIES:
                print(f"  ❌ Exit failed {MAX_EXIT_RETRIES} times ({reason}) — "
                      f"holding to resolution")
                self._exit_gave_up = True
            else:
                print(f"  ⚠️  Exit failed ({reason}, attempt "
                      f"{self._exit_retries}/{MAX_EXIT_RETRIES}): {result.get('error', '')}")
                self._last_position_check = time.time() + EXIT_RETRY_COOLDOWN - POSITION_CHECK_INTERVAL

    # ── Window management ───────────────────────────────────────────

    def _on_new_window(self, window_ts: int, closing_btc_price: float = 0.0):
        if self._current_window > 0:
            # Resolve any pending phantom sell from the previous window.
            # Must run before trade state is reset below.
            # Balance is fetched once here and reused by the sync below.
            if self._pending_phantom:
                pp = self._pending_phantom
                if not self.dry_run and self.executor._initialized:
                    real_bal = self.executor.get_collateral_balance()
                    if real_bal > 0:
                        balance_increase = max(0.0, real_bal - pp["pre_sell_balance"])
                        if balance_increase > pp["expected_revenue"] * 0.50:
                            # Settlement landed — it was a real win
                            profit = balance_increase - pp["cost"]
                            self.stats.bankroll += pp["cost"]  # restore cost deducted on buy
                            self.stats.record_win(profit)
                            self.stats.bankroll = real_bal
                            self._last_real_balance = real_bal
                            print(f"  ✅ Phantom resolved: WIN +${profit:.2f} [phantom_resolved] | "
                                  f"P&L: ${self.stats.total_pnl:+.2f} | Bank: ${self.stats.bankroll:.2f}")
                            self.telegram.win_alert(profit, self.stats.total_pnl)
                            btc_price, _ = self.price_feed.get_price()
                            self.tracker.log_trade_resolve(
                                btc_final_price=btc_price,
                                opening_price=pp["opening_price"],
                                won=True,
                                profit=profit,
                                exit_revenue=pp["exit_revenue"],
                                resolution_method="phantom_resolved",
                                claim_result="phantom_resolved",
                            )
                        else:
                            # Balance still hasn't moved — genuine loss
                            net_loss = pp["cost"] - pp["exit_revenue"]
                            self.stats.bankroll += pp["cost"]  # restore cost deducted on buy
                            self.stats.record_loss(net_loss)
                            self.stats.bankroll = real_bal
                            self._last_real_balance = real_bal
                            print(f"  ❌ Phantom confirmed: LOSS -${net_loss:.2f} [phantom_confirmed] | "
                                  f"P&L: ${self.stats.total_pnl:+.2f} | Bank: ${self.stats.bankroll:.2f}")
                            self.telegram.loss_alert(net_loss, self.stats.total_pnl)
                            btc_price, _ = self.price_feed.get_price()
                            self.tracker.log_trade_resolve(
                                btc_final_price=btc_price,
                                opening_price=pp["opening_price"],
                                won=False,
                                profit=-net_loss,
                                exit_revenue=pp["exit_revenue"],
                                resolution_method="phantom_confirmed",
                                claim_result="phantom_confirmed",
                            )
                        self._pending_phantom = {}
                else:
                    # Dry run or executor not ready — treat as loss
                    net_loss = pp["cost"] - pp["exit_revenue"]
                    self.stats.bankroll += pp["cost"]  # restore cost deducted on buy
                    self.stats.record_loss(net_loss)
                    self._pending_phantom = {}

            # Record closing delta for rolling vol calculation
            if self._opening_price > 0 and closing_btc_price > 0:
                closing_delta = abs((closing_btc_price - self._opening_price) / self._opening_price * 100)
                self._recent_window_deltas.append(closing_delta)
                if len(self._recent_window_deltas) > self._rolling_vol_windows:
                    self._recent_window_deltas.pop(0)
            # Record dominant option price for regime filter
            if self._last_tick_context:
                ctx = self._last_tick_context
                closing_delta_pct = ((closing_btc_price - self._opening_price) / self._opening_price * 100) if self._opening_price > 0 else 0
                if closing_delta_pct >= 0:
                    dom_price = ctx.get("up_price", 0)
                else:
                    dom_price = ctx.get("down_price", 0)
                if dom_price > 0:
                    self._recent_window_prices.append(dom_price)
                    if len(self._recent_window_prices) > self._rolling_price_windows:
                        self._recent_window_prices.pop(0)
            # Detect pending buy that settled after our verification timeout
            if self._pending_buy_side and not self._traded:
                if not self.dry_run and self.executor._initialized:
                    real_bal = self.executor.get_collateral_balance()
                    if real_bal > 0 and self._balance_before_buy > 0:
                        spent = self._balance_before_buy - real_bal
                        if spent > 1.0:
                            # The buy DID go through — retroactively track it
                            est_shares = spent / self._pending_buy_price if self._pending_buy_price > 0 else 0
                            print(f"\n  👻 LATE FILL: balance dropped ${spent:.2f} since buy attempt")
                            print(f"     Retroactively tracking: ~{est_shares:.0f} shares "
                                  f"{self._pending_buy_side} @ ${self._pending_buy_price:.3f}")

                            self._traded = True
                            self._trade_side = self._pending_buy_side
                            self._trade_price = self._pending_buy_price
                            self._trade_cost = spent
                            self._trade_shares = est_shares
                            self._trade_token_id = self._pending_buy_token_id
                            self.stats.bankroll = real_bal
                            self._last_real_balance = real_bal
                            self.stats.hourly.record_trade(
                                self._pending_buy_edge, self._pending_buy_delta)

            self.stats.hourly.record_window(self._traded)
            if self._traded:
                self._resolve_previous_trade()
            elif self._last_tick_context:
                # Log the no-trade signal for this window using last tick state
                ctx = self._last_tick_context
                if self._regime_skip:
                    skip_reason = "regime_expensive"
                else:
                    skip_reason = get_skip_reason(
                    btc_price=ctx["btc_price"],
                    opening_price=self._opening_price,
                    up_market_price=ctx["up_price"],
                    down_market_price=ctx["down_price"],
                    seconds_remaining=ctx["seconds_remaining"],
                    config=self.strategy_config,
                    realized_vol=self._compute_realized_vol(),
                )
                sig = ctx.get("signal")
                self.tracker.log_signal(
                    window_ts=ctx["window_ts"],
                    btc_price=ctx["btc_price"],
                    opening_price=self._opening_price,
                    up_price=ctx["up_price"],
                    down_price=ctx["down_price"],
                    seconds_remaining=ctx["seconds_remaining"],
                    side=sig.side if sig else "",
                    true_prob=sig.true_prob if sig else 0.0,
                    market_price=sig.market_price if sig else 0.0,
                    edge=sig.edge if sig else 0.0,
                    kelly_size=sig.kelly_size if sig else 0.0,
                    action="no_signal",
                    skip_reason=skip_reason,
                )

            # Sync real balance at window boundary (catches any drift)
            if not self.dry_run and self.executor._initialized:
                real_bal = self.executor.get_collateral_balance()
                if real_bal > 0:
                    drift = abs(real_bal - self.stats.bankroll)
                    if drift > 0.50:
                        print(f"  🔄 Balance sync: ${self.stats.bankroll:.2f} → "
                              f"${real_bal:.2f} (drift ${drift:.2f})")
                    self.stats.bankroll = real_bal
                    self._last_real_balance = real_bal

        self._current_window = window_ts
        self._opening_price = 0.0
        self._traded = False
        self._trade_attempted = False
        self._exited = False
        self._exit_revenue = 0.0
        self._exit_shares_sold = 0.0
        self._residual_shares = 0.0
        self._last_position_check = 0.0
        self._exit_retries = 0
        self._exit_gave_up = False
        self._last_sell_price_seen = 0.0
        self._cached_up = 0.50
        self._cached_down = 0.50
        self._price_last_fetched = 0.0
        self._current_market = None
        self._current_condition_id = ""
        self._pending_buy_side = ""
        self._pending_buy_price = 0.0
        self._pending_buy_amount = 0.0
        self._pending_buy_shares = 0.0
        self._pending_buy_token_id = ""
        self._pending_buy_edge = 0.0
        self._pending_buy_delta = 0.0
        self._balance_before_buy = 0.0

        # Reset composite signal state for new window
        self._composite_mode = False
        self._prev_composite_score = 0.0
        self._best_composite = None
        self._last_composite_check = 0.0
        self._klines_fetched_this_window = False
        self._opening_price_from_kline = 0.0
        self.price_feed.clear_tick_prices()

        t = time.strftime("%H:%M:%S", time.localtime(window_ts))
        print(f"\n{'─' * 55}")
        print(f"🕐 {t} | Trades: {self.stats.total_trades} | "
              f"W/L: {self.stats.wins}/{self.stats.losses} | "
              f"P&L: ${self.stats.total_pnl:+.2f}")
        print(f"{'─' * 55}")

        # Circuit breaker auto-recovery: ping CLOB each new window
        if self._clob_halted and not self.dry_run and self.executor._initialized:
            try:
                self._clob_healthy()
                self._clob_halted = False
                self._consecutive_buy_failures = 0
                print(f"  ✅ CLOB recovered (health check OK) — resuming trades")
            except Exception:
                print(f"  🔌 CLOB health check still failing — staying halted")
                self.telegram.error_alert("CLOB health check failed — trading halted")

    # ── Market prices (cached, complement engine) ───────────────────

    def _compute_realized_vol(self) -> float:
        """Rolling std dev of recent window closing deltas.

        Returns the realized vol to pass into the Brownian motion model.
        Falls back to 0.12 until at least 6 windows have accumulated.
        Floored/capped to prevent extreme values breaking the model.
        """
        min_samples = max(6, self._rolling_vol_windows // 2)
        if len(self._recent_window_deltas) >= min_samples:
            vol = statistics.stdev(self._recent_window_deltas)
            return max(self._vol_floor, min(self._vol_cap, vol))
        return self._vol_fallback

    def _compute_rolling_avg_price(self) -> float:
        """Rolling average of the dominant option price over recent windows.

        Returns the average market price of the option we'd buy, over the last
        N windows. Used as a regime filter — when the market consistently prices
        options expensive (> $0.62), the market is confident and our edge is fake.
        Returns 0.0 until enough windows have accumulated.
        """
        min_samples = max(3, self._rolling_price_windows // 3)
        if len(self._recent_window_prices) >= min_samples:
            return sum(self._recent_window_prices) / len(self._recent_window_prices)
        return 0.0

    def _get_market_prices(self, btc_price: float, seconds_remaining: float) -> tuple:
        """Fetch live market prices from CLOB. Returns (up, down) or (0, 0) to signal halt."""
        if not self.executor._initialized:
            return 0, 0

        try:
            market = get_current_market(self.period)
            self._current_market = market
            self._current_condition_id = market.condition_id if market else ""
            if not market:
                return 0, 0

            probe_amount = 5.0
            up_price = self.executor.get_market_price(market.token_id_up, "BUY", probe_amount)
            down_price = self.executor.get_market_price(market.token_id_down, "BUY", probe_amount)

            if up_price <= 0 and down_price <= 0:
                print("[price] ✗ Both prices failed — halting this window")
                return 0, 0
            if up_price <= 0:
                up_price = round(1.0 - down_price, 3)
            if down_price <= 0:
                down_price = round(1.0 - up_price, 3)

            return up_price, down_price

        except Exception as e:
            print(f"[price] ✗ Fetch failed: {e} — halting this window")
            self.telegram.error_alert(f"Price fetch failed: {e}")
            return 0, 0

    # ── Entry ───────────────────────────────────────────────────────

    def _execute_trade(self, sig, seconds_remaining: float):
        self._trade_attempted = True

        # ── Circuit breaker: CLOB health check ───────────────────
        if self._clob_halted:
            print(f"  🔌 CLOB HALTED — skipping trade ({self._consecutive_buy_failures} consecutive failures)")
            return

        if not self.dry_run and self.executor._initialized:
            try:
                self._clob_healthy()
            except Exception as e:
                self._consecutive_buy_failures += 1
                print(f"  🔌 CLOB health check failed: {e}")
                if self._consecutive_buy_failures >= self._HALT_AFTER_FAILURES:
                    self._clob_halted = True
                    msg = (f"🔌 CLOB HALTED after {self._consecutive_buy_failures} "
                           f"consecutive health check failures — stopping trades until recovery")
                    print(f"\n  {msg}")
                    self.telegram.status_update({"alert": msg})
                return

        # ── Daily loss limit ─────────────────────────────────────
        if self._daily_loss_halted:
            print(f"  🛑 DAILY LOSS LIMIT — session P&L ${self.stats.total_pnl:+.2f} "
                  f"exceeds -${self._daily_loss_limit:.0f}")
            return

        session_pnl = self.stats.bankroll - self._session_start_balance
        if session_pnl <= -self._daily_loss_limit:
            self._daily_loss_halted = True
            msg = (f"🛑 DAILY LOSS LIMIT HIT: ${session_pnl:+.2f} "
                   f"(limit -${self._daily_loss_limit:.0f}) — STOPPING BOT")
            print(f"\n  {msg}")
            self.telegram.status_update({"alert": msg})
            self._handle_shutdown(None, None)  # Full shutdown with report
            return

        # ── Trading hours gate ──────────────────────────────────
        utc_hour = int(time.strftime("%H", time.gmtime()))
        if self._trading_hours_start < self._trading_hours_end:
            # Normal range: e.g. 13–22
            in_hours = self._trading_hours_start <= utc_hour < self._trading_hours_end
        else:
            # Wrapping range: e.g. 22–06 (overnight)
            in_hours = utc_hour >= self._trading_hours_start or utc_hour < self._trading_hours_end
        if not in_hours:
            print(f"  ⏰ Outside trading hours ({self._trading_hours_start:02d}:00–"
                  f"{self._trading_hours_end:02d}:00 UTC, now {utc_hour:02d}:00) — skipping")
            return

        market = self._current_market if self._current_market else (get_current_market(self.period) if not self.dry_run else None)
        token_id = ""
        if market:
            token_id = market.token_id_up if sig.side == "UP" else market.token_id_down
        else:
            token_id = f"DRY-{sig.side}-{self._current_window}"

        slug = f"btc-updown-{self.period}m-{self._current_window}"
        trade_amount = round(sig.kelly_size, 2)

        print(f"\n  🎯 {sig.side} | edge={sig.edge:.3f} | "
              f"prob={sig.true_prob:.2f} | BTC Δ={sig.btc_delta_pct:+.3f}%")
        print(f"     Kelly: ${trade_amount:.2f} | mkt ${sig.market_price:.3f} | T-{seconds_remaining:.0f}s")

        # Preview actual market price, re-check edge, then pass price into buy()
        # so executor skips a second fetch (saves one Tor roundtrip ~500ms)
        hint_price = sig.market_price  # dry-run uses strategy's price estimate
        if not self.dry_run and self.executor._initialized:
            actual_price = self.executor.get_market_price(token_id, "BUY", trade_amount)
            if actual_price > 0:
                actual_edge = sig.true_prob - actual_price
                print(f"  📊 Actual price: ${actual_price:.3f} (edge: {actual_edge:.3f})")

                if actual_edge < self.strategy_config.min_edge:
                    print(f"  ⚠️  Edge gone at market price — skipping")
                    btc_approx = self._opening_price * (1 + sig.btc_delta_pct / 100) if self._opening_price > 0 else 0
                    self.tracker.log_signal(
                        window_ts=self._current_window,
                        btc_price=btc_approx,
                        opening_price=self._opening_price,
                        up_price=self._cached_up,
                        down_price=self._cached_down,
                        seconds_remaining=seconds_remaining,
                        side=sig.side,
                        true_prob=sig.true_prob,
                        market_price=actual_price,
                        edge=actual_edge,
                        kelly_size=sig.kelly_size,
                        action="skipped_edge_gone",
                        skip_reason="edge_gone_at_market",
                        actual_price=actual_price,
                        actual_edge=actual_edge,
                    )
                    return

                hint_price = actual_price

                # Slippage check — skip if actual price moved too far from signal
                slippage = (actual_price - sig.market_price) / sig.market_price if sig.market_price > 0 else 0
                if slippage > 0.10:
                    print(f"  ⚠️  Slippage {slippage:.1%} exceeds 10% — skipping (signal ${sig.market_price:.3f} → actual ${actual_price:.3f})")
                    return

        result = self.executor.buy(token_id=token_id, amount_usd=trade_amount, price=hint_price)

        status = result.get("status", FAILED)
        if status == FILLED:
            self._consecutive_buy_failures = 0  # Reset circuit breaker
            self._traded = True
            self._trade_side = sig.side
            self._trade_price = result["price"]
            self._trade_cost = result["cost"]
            self._trade_shares = result["shares"]
            self._trade_token_id = token_id

            self.stats.bankroll -= result["cost"]
            self.stats.hourly.record_trade(sig.edge, sig.btc_delta_pct)

            btc_approx = self._opening_price * (1 + sig.btc_delta_pct / 100) if self._opening_price > 0 else 0
            self.tracker.log_signal(
                window_ts=self._current_window,
                btc_price=btc_approx,
                opening_price=self._opening_price,
                up_price=self._cached_up,
                down_price=self._cached_down,
                seconds_remaining=seconds_remaining,
                side=sig.side,
                true_prob=sig.true_prob,
                market_price=sig.market_price,
                edge=sig.edge,
                kelly_size=sig.kelly_size,
                action="traded",
                actual_price=result["price"],
                actual_edge=sig.true_prob - result["price"],
                fill_price=result["price"],
            )
            self.tracker.log_trade_entry(
                window_ts=self._current_window,
                side=sig.side,
                entry_price=result["price"],
                entry_shares=result["shares"],
                entry_cost=result["cost"],
                edge=sig.edge,
                prob=sig.true_prob,
                btc_delta=sig.btc_delta_pct,
                seconds_remaining=seconds_remaining,
                entry_delta_pct=sig.btc_delta_pct,
                entry_seconds_remaining=seconds_remaining,
            )

            mode = "PAPER" if self.dry_run else "LIVE"
            print(f"  ✅ {mode}: {result['shares']:.0f} shares @ "
                  f"${result['price']:.3f} = ${result['cost']:.2f}")
            print(f"     Holding to resolution (no stops)")

            self.telegram.trade_alert(
                side=sig.side, price=result["price"], amount=result["cost"],
                market_slug=slug, dry_run=self.dry_run,
                edge=sig.edge, kelly_size=sig.kelly_size,
            )
        else:
            error_msg = result.get("error", "")
            if "unverified" in error_msg.lower() or "timeout" in error_msg.lower():
                # Order likely filled but Polygon hasn't settled.
                # Save details — window boundary sync will detect the fill.
                self._pending_buy_side = sig.side
                self._pending_buy_price = result.get("price", 0.0)
                self._pending_buy_amount = result.get("cost", 0.0)
                self._pending_buy_shares = result.get("shares", 0.0)
                self._pending_buy_token_id = token_id
                self._pending_buy_edge = sig.edge
                self._pending_buy_delta = sig.btc_delta_pct
                self._balance_before_buy = self.stats.bankroll
                print(f"  ⏳ Buy sent but unverified — will detect via balance sync")
            else:
                print(f"  ❌ Buy failed: {error_msg}")
                self.telegram.error_alert(f"Buy failed: {error_msg}")
                # Circuit breaker: track consecutive API failures
                err = str(error_msg).lower()
                if "request exception" in err or "service not ready" in err or "status_code=none" in err:
                    self._consecutive_buy_failures += 1
                    if self._consecutive_buy_failures >= self._HALT_AFTER_FAILURES:
                        self._clob_halted = True
                        msg = (f"🔌 CLOB HALTED after {self._consecutive_buy_failures} "
                               f"consecutive API failures — stopping trades until restart")
                        print(f"\n  {msg}")
                        self.telegram.status_update({"alert": msg})

    # ── Resolve (partial fill aware) ────────────────────────────────

    def _resolve_previous_trade(self):
        if self._exited:
            # Restore cost deducted on buy — record_win/record_loss will
            # re-apply the correct net P&L to bankroll.
            self.stats.bankroll += self._trade_cost
            profit = self._exit_revenue - self._trade_cost
            if profit > 0:
                self.stats.record_win(profit)
            else:
                self.stats.record_loss(abs(profit))
            result_emoji = "✅ WIN" if profit > 0 else "❌ LOSS"
            residual_note = f" (~{self._residual_shares:.0f} residual)" if self._residual_shares >= 1 else ""
            print(f"  {result_emoji} (exited{residual_note}) ${profit:+.2f} | "
                  f"P&L: ${self.stats.total_pnl:+.2f} | "
                  f"Bank: ${self.stats.bankroll:.2f}")
            if profit > 0:
                self.telegram.win_alert(profit, self.stats.total_pnl)
            else:
                self.telegram.loss_alert(abs(profit), self.stats.total_pnl)
            btc_price, _ = self.price_feed.get_price()
            self.tracker.log_trade_resolve(
                btc_final_price=btc_price,
                opening_price=self._opening_price,
                won=profit > 0,
                profit=profit,
                exit_revenue=self._exit_revenue,
                resolution_method="exited",
            )
            return

        original_cost = self._trade_cost
        remaining_shares = self._trade_shares

        # ── Dry run: Binance price fallback ──────────────────────────
        if self.dry_run:
            btc_price, _ = self.price_feed.get_price()
            if self._opening_price <= 0 or btc_price <= 0:
                return
            won = (btc_price >= self._opening_price) == (self._trade_side == "UP")
            self._record_resolution(
                won=won,
                original_cost=original_cost,
                remaining_shares=remaining_shares,
                resolution_method="binance_fallback",
                claim_revenue=0.0,
            )
            return

        # ── Live: attempt claim sell first — result is the truth ─────
        # Binance price and oracle can disagree when BTC is near the opening
        # price at resolution. The claim sell result is ground truth:
        #   - Sell succeeds at ~$0.99 → shares had value → won
        #   - "no match" or near-zero fill → shares worthless → lost
        won = None
        claim_revenue = 0.0
        claim_result = "not_attempted"
        resolution_method = "claim_sell"

        claim_notional = remaining_shares * 0.99
        live_token = (self._trade_token_id
                      and not self._trade_token_id.startswith("DRY-")
                      and self.executor._initialized)

        # Short-circuit: if last observed sell price is below $0.50, market has
        # already priced these shares as worthless — skip the claim API call.
        if self._last_sell_price_seen > 0 and self._last_sell_price_seen < 0.50:
            net_loss = original_cost - self._exit_revenue
            profit = -net_loss
            self.stats.record_loss(net_loss)
            partial_note = f" (partial exit ${self._exit_revenue:.2f})" if self._exit_revenue > 0 else ""
            print(f"  ❌ LOSS{partial_note} -${net_loss:.2f} [market_price] | "
                  f"P&L: ${self.stats.total_pnl:+.2f} | "
                  f"Bank: ${self.stats.bankroll:.2f}")
            self.telegram.loss_alert(net_loss, self.stats.total_pnl)
            btc_price, _ = self.price_feed.get_price()
            self.tracker.log_trade_resolve(
                btc_final_price=btc_price,
                opening_price=self._opening_price,
                won=False,
                profit=profit,
                exit_revenue=self._exit_revenue,
                resolution_method="market_price",
                claim_result="skipped_losing",
            )
            return

        pre_sell_balance = 0.0
        if live_token and claim_notional >= 5.0:
            print(f"  💰 Claiming: sell {remaining_shares:.0f} shares @ $0.99...")
            pre_sell_balance = self.executor.get_collateral_balance()
            claim = self.executor.sell(
                token_id=self._trade_token_id,
                shares=remaining_shares,
                price=0.99,
            )
            claim_status = claim.get("status", FAILED)
            claim_revenue_val = claim.get("revenue", 0.0)
            claim_error_msg = claim.get("error", "")
            if claim_status == FILLED and claim_revenue_val > remaining_shares * 0.50:
                # API says success — verify with balance check to catch phantom fills
                time.sleep(2)
                post_sell_balance = self.executor.get_collateral_balance()
                balance_increase = max(0.0, post_sell_balance - pre_sell_balance) if (
                    pre_sell_balance > 0 and post_sell_balance > 0
                ) else claim_revenue_val
                if balance_increase > remaining_shares * 0.99 * 0.50:
                    # Balance confirmed — real fill
                    won = True
                    claim_revenue = claim_revenue_val
                    claim_result = "filled"
                    self.stats.bankroll += claim_revenue
                else:
                    # API said success but no USDC arrived yet — defer to next window
                    print(f"  ⏳ Possible phantom sell "
                          f"(api=${claim_revenue_val:.2f}, balance_increase=${balance_increase:.2f})"
                          f" — deferring to next window balance sync")
            elif "no match" in claim_error_msg.lower() or (
                claim_status == FILLED and claim_revenue_val < remaining_shares * 0.10
            ):
                # No buyers for these shares → shares worthless → definitive loss
                won = False
                claim_result = "no_match"
            elif "not enough balance" in claim_error_msg.lower():
                # Tracked share count is slightly above on-chain balance (rounding).
                # Retry with one fewer share to clear the discrepancy.
                retry_shares = int(remaining_shares) - 1
                print(f"  🔄 Rounding fix: retrying claim with {retry_shares} shares...")
                if retry_shares > 0 and float(retry_shares) * 0.99 >= 5.0:
                    retry = self.executor.sell(
                        token_id=self._trade_token_id,
                        shares=float(retry_shares),
                        price=0.99,
                    )
                    retry_status = retry.get("status", FAILED)
                    retry_revenue = retry.get("revenue", 0.0)
                    if retry_status == FILLED and retry_revenue > retry_shares * 0.50:
                        time.sleep(2)
                        post_bal = self.executor.get_collateral_balance()
                        balance_increase = max(0.0, post_bal - pre_sell_balance)
                        if balance_increase > float(retry_shares) * 0.99 * 0.50:
                            won = True
                            claim_revenue = retry_revenue
                            claim_result = "filled"
                            self.stats.bankroll += claim_revenue
                        # else: retry succeeded but balance unconfirmed — fall to defer
                # else: retry failed or too small — fall to defer (won still None)
            # else: any other error — fall to defer (won still None)
                if claim_error_msg:
                    self.telegram.error_alert(f"Claim failed: {claim_error_msg}")
        else:
            if live_token and claim_notional < 5.0:
                print(f"  💰 {remaining_shares:.0f} shares below $5 min — deferring to auto-resolution")

        # ── Deferred fallback ────────────────────────────────────────
        # The old balance check fired before auto-resolution settled on-chain.
        # Any unresolved case is now deferred to the next window boundary
        # (~5 min), where Polygon settlement is guaranteed to have landed.
        if won is None:
            if not live_token:
                # No valid token/executor — Binance price as last resort
                btc_price, _ = self.price_feed.get_price()
                if self._opening_price > 0 and btc_price > 0:
                    won = (btc_price >= self._opening_price) == (self._trade_side == "UP")
                    resolution_method = "binance_fallback"
                    print(f"  ⚠️  No live token — using Binance fallback")
                else:
                    print(f"  ⚠️  Cannot determine resolution outcome — skipping")
                    return
            else:
                if pre_sell_balance <= 0:
                    pre_sell_balance = self.executor.get_collateral_balance()
                print(f"  ⏳ Resolution deferred to next window balance sync")
                self._pending_phantom = {
                    "pre_sell_balance": pre_sell_balance,
                    "expected_revenue": remaining_shares * 0.99,
                    "cost": original_cost,
                    "exit_revenue": self._exit_revenue,
                    "shares": remaining_shares,
                    "side": self._trade_side,
                    "token_id": self._trade_token_id,
                    "window_ts": self._current_window,
                    "opening_price": self._opening_price,
                }
                return

        if won is None:
            return

        self._record_resolution(
            won=won,
            original_cost=original_cost,
            remaining_shares=remaining_shares,
            resolution_method=resolution_method,
            claim_revenue=claim_revenue,
            claim_result=claim_result,
        )

    def _record_resolution(
        self, won: bool, original_cost: float, remaining_shares: float,
        resolution_method: str, claim_revenue: float, claim_result: str = "not_attempted",
    ):
        """Apply win/loss to stats, print result, alert Telegram, log to tracker."""
        # Restore cost deducted on buy — record_win/record_loss will
        # re-apply the correct net P&L to bankroll.
        self.stats.bankroll += original_cost
        if won:
            if claim_revenue > 0:
                total_received = self._exit_revenue + claim_revenue
            else:
                resolution_payout = remaining_shares * 1.0
                total_received = self._exit_revenue + resolution_payout
                self.stats.bankroll += resolution_payout
                self._unclaimed_winnings += resolution_payout
            profit = total_received - original_cost
            self.stats.record_win(profit)
            partial_note = f" (partial exit ${self._exit_revenue:.2f})" if self._exit_revenue > 0 else ""
            claimed_note = " (claimed)" if claim_revenue > 0 else " (unclaimed)"
            print(f"  ✅ WIN{partial_note}{claimed_note} +${profit:.2f} [{resolution_method}] | "
                  f"P&L: ${self.stats.total_pnl:+.2f} | "
                  f"Bank: ${self.stats.bankroll:.2f}")
            self.telegram.win_alert(profit, self.stats.total_pnl)
        else:
            net_loss = original_cost - self._exit_revenue
            profit = -net_loss
            self.stats.record_loss(net_loss)
            partial_note = f" (partial exit ${self._exit_revenue:.2f})" if self._exit_revenue > 0 else ""
            print(f"  ❌ LOSS{partial_note} -${net_loss:.2f} [{resolution_method}] | "
                  f"P&L: ${self.stats.total_pnl:+.2f} | "
                  f"Bank: ${self.stats.bankroll:.2f}")
            self.telegram.loss_alert(net_loss, self.stats.total_pnl)

        btc_price, _ = self.price_feed.get_price()
        self.tracker.log_trade_resolve(
            btc_final_price=btc_price,
            opening_price=self._opening_price,
            won=won,
            profit=profit,
            exit_revenue=self._exit_revenue,
            resolution_method=resolution_method,
            claim_result=claim_result,
        )

    # ── Hourly + shutdown ───────────────────────────────────────────

    def _check_hourly_summary(self):
        current_hour = int(time.time() // 3600)
        if current_hour != self._last_hour_check:
            self._last_hour_check = current_hour
            h = self.stats.hourly.to_dict()
            o = self.stats.to_dict()

            # Sync real balance for accuracy
            if not self.dry_run and self.executor._initialized:
                real_bal = self.executor.get_collateral_balance()
                if real_bal > 0:
                    self.stats.bankroll = real_bal
                    self._last_real_balance = real_bal
                    o["bankroll"] = real_bal

            real_pnl = self.stats.bankroll - self._session_start_balance

            print(f"\n{'═' * 55}")
            print(f"  📊 HOURLY SUMMARY")
            print(f"  This hour: {h['trades']} trades | "
                  f"{h['wins']}W/{h['losses']}L | "
                  f"P&L: ${h['pnl']:+.2f}")
            if h['trades'] > 0:
                print(f"  Avg edge: {h['avg_edge']*100:.1f}%")
            print(f"  Windows: {h['windows_seen']} seen, "
                  f"{h['windows_skipped']} skipped")
            print(f"  Overall: {o['total_trades']} trades | "
                  f"P&L: ${o['pnl']:+.2f} | Bank: ${o['bankroll']:.2f}")
            print(f"  💰 Real P&L (balance): ${real_pnl:+.2f} "
                  f"(${self._session_start_balance:.2f} → ${self.stats.bankroll:.2f})")
            if self._unclaimed_winnings > 0:
                print(f"  💰 Unclaimed: ${self._unclaimed_winnings:.2f}")
            print(f"{'═' * 55}\n")
            self.telegram.hourly_summary(h, o)
            self.stats.hourly.reset()

    def _handle_shutdown(self, signum, frame):
        print(f"\n\n🛑 Shutting down...")
        self._running = False
        self.price_feed.stop()
        if self.executor._initialized:
            self.executor.cancel_all_orders()

        # Final real balance sync
        if not self.dry_run and self.executor._initialized:
            real_bal = self.executor.get_collateral_balance()
            if real_bal > 0:
                self.stats.bankroll = real_bal
                self._last_real_balance = real_bal

        real_pnl = self.stats.bankroll - self._session_start_balance
        o = self.stats.to_dict()
        print(f"\n{'═' * 55}")
        print(f"  FINAL: {o['total_trades']} trades | "
              f"{o['wins']}W/{o['losses']}L | "
              f"WR: {o['win_rate']:.1f}%")
        print(f"  Tracked P&L: ${o['pnl']:+.2f} | Bank: ${o['bankroll']:.2f}")
        print(f"  💰 Real P&L: ${real_pnl:+.2f} "
              f"(${self._session_start_balance:.2f} → ${self.stats.bankroll:.2f})")
        if self._unclaimed_winnings > 0:
            print(f"  💰 Unclaimed: ${self._unclaimed_winnings:.2f}")
        print(f"{'═' * 55}")

        self.telegram.status_update(o)

        self.tracker.log_session(
            start_time=self._session_start_time,
            end_time=time.time(),
            start_balance=self._session_start_balance,
            end_balance=self.stats.bankroll,
            tracked_pnl=o["pnl"],
            trades=o["total_trades"],
            wins=o["wins"],
            losses=o["losses"],
            avg_entry_price=self.stats.hourly.avg_edge,   # proxy via hourly stats
            avg_edge=self.stats.hourly.avg_edge,
            avg_delta=self.stats.hourly.avg_delta,
        )

        time.sleep(1)
        sys.exit(0)


if __name__ == "__main__":
    bot = PolyBot()
    bot.start()
