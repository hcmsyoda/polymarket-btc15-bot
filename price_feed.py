"""Real-time BTC price feed from Binance WebSocket.

Subscribes to BTCUSDT trade stream for tick-by-tick price updates.
Falls back to REST API polling if WebSocket fails.
"""

import json
import time
import threading
import urllib.request
from dataclasses import dataclass, field
from typing import Optional, Callable


BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@trade"
BINANCE_REST_URL = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


@dataclass
class PriceState:
    """Thread-safe container for current BTC price."""
    price: float = 0.0
    timestamp: float = 0.0
    source: str = "none"
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, price: float, source: str = "ws"):
        with self._lock:
            self.price = price
            self.timestamp = time.time()
            self.source = source

    def get(self) -> tuple[float, float]:
        """Returns (price, age_in_seconds)."""
        with self._lock:
            return self.price, time.time() - self.timestamp

    @property
    def is_fresh(self) -> bool:
        """Price is considered fresh if < 5 seconds old."""
        _, age = self.get()
        return age < 5.0 and self.price > 0


class BinancePriceFeed:
    def __init__(self):
        self.state = PriceState()
        self._ws_thread: Optional[threading.Thread] = None
        self._running = False
        self._on_price: Optional[Callable] = None
        # Momentum tracking — store (timestamp, price) for last 120s
        self._price_history: list = []
        self._history_lock = threading.Lock()
        # Tick price buffer for composite signal (last 30 ticks at ~2s intervals)
        self._tick_prices: list = []
        self._tick_lock = threading.Lock()
        self._MAX_TICKS = 30
        # Kline cache — {window_ts: [[kline_data], ...]}
        self._kline_cache: dict = {}
        self._kline_lock = threading.Lock()

    def _record_price(self, price: float):
        """Record price with timestamp for momentum calculation and tick trend."""
        now = time.time()
        with self._history_lock:
            self._price_history.append((now, price))
            # Prune entries older than 120s
            cutoff = now - 120
            self._price_history = [(t, p) for t, p in self._price_history if t > cutoff]
        with self._tick_lock:
            self._tick_prices.append(price)
            if len(self._tick_prices) > self._MAX_TICKS:
                self._tick_prices = self._tick_prices[-self._MAX_TICKS:]

    def get_tick_prices(self) -> list:
        """Return recent tick prices for composite signal tick trend indicator."""
        with self._tick_lock:
            return list(self._tick_prices)

    def clear_tick_prices(self):
        """Clear tick buffer (call at start of each new window)."""
        with self._tick_lock:
            self._tick_prices.clear()

    def get_momentum(self, lookback_seconds: float = 60.0) -> dict:
        """Calculate Binance price momentum over lookback window.

        Returns dict with:
          - pct_change: price change as percentage (positive = BTC went up)
          - abs_change: absolute price change
          - start_price: price at start of window
          - end_price: current price
          - valid: True if enough data exists
        """
        now = time.time()
        cutoff = now - lookback_seconds
        with self._history_lock:
            window = [(t, p) for t, p in self._price_history if t > cutoff]

        if len(window) < 2:
            return {"pct_change": 0, "abs_change": 0, "start_price": 0,
                    "end_price": 0, "valid": False}

        start_price = window[0][1]
        end_price = window[-1][1]
        abs_change = end_price - start_price
        pct_change = (abs_change / start_price) * 100 if start_price > 0 else 0

        return {
            "pct_change": pct_change,
            "abs_change": abs_change,
            "start_price": start_price,
            "end_price": end_price,
            "valid": True,
        }

    def start(self, on_price: Callable = None):
        """Start the price feed. Tries WebSocket first, falls back to REST polling."""
        self._on_price = on_price
        self._running = True

        # Try WebSocket first
        self._ws_thread = threading.Thread(target=self._ws_loop, daemon=True)
        self._ws_thread.start()

        # Also start REST poller as backup
        threading.Thread(target=self._rest_poll_loop, daemon=True).start()

        print("[price] BTC price feed starting...")

    def stop(self):
        self._running = False

    def _ws_loop(self):
        """WebSocket connection to Binance for real-time trades."""
        try:
            import websockets
            import asyncio

            async def connect():
                while self._running:
                    try:
                        async with websockets.connect(BINANCE_WS_URL) as ws:
                            print("[price] WebSocket connected to Binance")
                            while self._running:
                                msg = await asyncio.wait_for(ws.recv(), timeout=30)
                                data = json.loads(msg)
                                price = float(data.get("p", 0))
                                if price > 0:
                                    self.state.update(price, source="ws")
                                    self._record_price(price)
                                    if self._on_price:
                                        self._on_price(price)
                    except Exception as e:
                        if self._running:
                            print(f"[price] WebSocket error: {e}, reconnecting in 3s...")
                            await asyncio.sleep(3)

            asyncio.run(connect())
        except ImportError:
            print("[price] websockets not available, using REST polling only")

    def _rest_poll_loop(self):
        """Fallback: poll Binance REST API every 2 seconds."""
        time.sleep(3)  # Give WebSocket a head start
        while self._running:
            try:
                # Only poll if WebSocket data is stale
                if not self.state.is_fresh:
                    req = urllib.request.Request(
                        BINANCE_REST_URL,
                        headers={"User-Agent": "PolyBot/1.0"},
                    )
                    resp = urllib.request.urlopen(req, timeout=5)
                    data = json.loads(resp.read().decode())
                    price = float(data.get("price", 0))
                    if price > 0:
                        self.state.update(price, source="rest")
                        self._record_price(price)
                        if self._on_price:
                            self._on_price(price)
            except Exception:
                pass
            time.sleep(2)

    def get_price(self) -> tuple[float, bool]:
        """Get current BTC price and whether it's fresh.
        
        Returns (price, is_fresh).
        """
        price, age = self.state.get()
        return price, self.state.is_fresh

    def wait_for_price(self, timeout: float = 30) -> float:
        """Block until we have a valid price. Returns price or 0 on timeout."""
        start = time.time()
        while time.time() - start < timeout:
            if self.state.is_fresh:
                return self.state.price
            time.sleep(0.1)
        return 0.0

    def fetch_klines(self, window_ts: int) -> list:
        """Fetch 1-minute klines from Binance for the current 5-min window.

        Returns list of klines: [[open_time, open, high, low, close, volume, ...], ...]
        Caches results per window_ts to avoid redundant API calls.
        """
        with self._kline_lock:
            if window_ts in self._kline_cache:
                return self._kline_cache[window_ts]

        # Fetch 1-min klines that overlap this 5-min window
        # Request 10 to ensure we have enough history for EMA-21 etc.
        try:
            start_ms = (window_ts - 600) * 1000  # 10 min before window start
            url = (f"{BINANCE_KLINES_URL}?symbol=BTCUSDT&interval=1m"
                   f"&startTime={start_ms}&limit=10")
            req = urllib.request.Request(url, headers={"User-Agent": "PolyBot/1.0"})
            resp = urllib.request.urlopen(req, timeout=5)
            data = json.loads(resp.read().decode())
            with self._kline_lock:
                self._kline_cache[window_ts] = data
            return data
        except Exception as e:
            print(f"[price] Kline fetch failed: {e}")
            return []

    def get_window_open_from_kline(self, window_ts: int) -> float:
        """Get the exact opening price from the Binance 1-min kline for this window.

        This is the price Polymarket uses for resolution (Binance candle open).
        More accurate than using the first tick price which may arrive late.
        """
        klines = self.fetch_klines(window_ts)
        window_ts_ms = window_ts * 1000
        for k in klines:
            if int(k[0]) == window_ts_ms:
                return float(k[1])  # open price of the 1-min candle at window start
        return 0.0

    def get_klines_for_composite(self, window_ts: int) -> list:
        """Get 1-min klines for composite signal analysis.

        Returns klines from the current window and preceding history
        (up to ~10 min) for EMA/RSI calculation.
        """
        return self.fetch_klines(window_ts)


if __name__ == "__main__":
    feed = BinancePriceFeed()
    feed.start()
    
    print("Waiting for first price...")
    price = feed.wait_for_price(timeout=15)
    if price:
        print(f"BTC price: ${price:,.2f} (source: {feed.state.source})")
    else:
        print("Timeout waiting for price")
    
    # Watch for 10 seconds
    for i in range(10):
        time.sleep(1)
        p, fresh = feed.get_price()
        print(f"  [{i+1}s] ${p:,.2f} {'✓' if fresh else '✗'}")
    
    feed.stop()
