"""Polymarket CLOB v2 order execution with deposit wallet (POLY_1271) support.

Features:
- CLOB v2 API via py-clob-client-v2 (supports POLY_1271 / signature_type=3)
- Deposit wallet support (ERC-1271 signatures)
- Fee estimation from market metadata
- Market tradability checks
- Dry-run mode (default)
- Collateral balance terminology
"""

import os
import time
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from decimal import Decimal

# py-clob-client-v2 imports (supports POLY_1271 / signature_type=3)
try:
    from py_clob_client_v2.client import ClobClient
    from py_clob_client_v2.clob_types import (
        ApiCreds, OrderArgsV2, MarketOrderArgs, BalanceAllowanceParams,
        AssetType,
    )
    from py_clob_client_v2.order_builder.constants import BUY, SELL
    from py_clob_client_v2.constants import POLYGON
except ImportError as e:
    print(f"[executor] py-clob-client-v2 not installed: {e}")
    ClobClient = None

# Constants
CLOB_HOST = "https://clob.polymarket.com"
CHAIN_ID = 137  # Polygon mainnet
MAX_BUY_PRICE = 0.90
MIN_ORDER_SIZE = 5.0

# Order statuses
FILLED = "FILLED"
PARTIAL = "PARTIAL"
FAILED = "FAILED"
PENDING = "PENDING"
UNVERIFIED_BUY = "UNVERIFIED_BUY"


@dataclass
class MarketMeta:
    """Cached market metadata from CLOB v2."""
    condition_id: str
    active: bool
    closed: bool
    fee_rate_bps: int = 0
    min_order_size: str = "5"
    min_tick_size: str = "0.01"
    tokens: list = field(default_factory=list)


@dataclass
class BuilderConfig:
    """Builder fee configuration for CLOB v2."""
    address: str = ""
    fee: str = "0"

    def to_dict(self) -> Optional[dict]:
        if self.address and self.fee:
            return {"builder": self.address, "fee": self.fee}
        return None


class Executor:
    """Polymarket CLOB v2 order executor."""

    def __init__(
        self,
        private_key: str = "",
        safe_address: str = "",
        dry_run: bool = True,
    ):
        self.private_key = private_key
        self.safe_address = safe_address
        self.dry_run = dry_run
        self._initialized = False
        self.client: Optional[ClobClient] = None
        self.builder_config = self._load_builder_config()
        self._market_meta_cache: Dict[str, MarketMeta] = {}
        self._fee_cache: Dict[str, int] = {}
        self._fee_cache_ts: Dict[str, float] = {}
        self._api_creds: Optional[ApiCreds] = None

    def _load_builder_config(self) -> BuilderConfig:
        """Load builder config from environment variables."""
        builder_address = os.getenv("BUILDER_ADDRESS", "").strip()
        builder_fee = os.getenv("BUILDER_FEE", "").strip()
        if builder_address and builder_fee:
            return BuilderConfig(address=builder_address, fee=builder_fee)
        return BuilderConfig()

    def initialize(self) -> bool:
        """Initialize the CLOB v2 client and derive API credentials."""
        if not ClobClient:
            print("[executor] py-clob-client not available.")
            return False

        if not self.private_key:
            print("[executor] No private key provided.")
            return False

        try:
            # Strip 0x prefix if present
            key = self.private_key[2:] if self.private_key.startswith("0x") else self.private_key

            if self.safe_address:
                sig_type = int(os.getenv("SIGNATURE_TYPE", "2"))
                print(f"[executor] Initializing with wallet: {self.safe_address}, signature_type={sig_type}")
                self.client = ClobClient(
                    host=CLOB_HOST,
                    key=key,
                    chain_id=CHAIN_ID,
                    funder=self.safe_address,
                    signature_type=sig_type,
                )
            else:
                print("[executor] Initializing with EOA wallet.")
                self.client = ClobClient(
                    host=CLOB_HOST,
                    key=key,
                    chain_id=CHAIN_ID,
                )

            # Derive API credentials
            # Check for explicit API credentials from .env
            explicit_key = os.getenv("CLOB_API_KEY")
            explicit_secret = os.getenv("CLOB_SECRET")
            explicit_passphrase = os.getenv("CLOB_PASSPHRASE")

            if explicit_key and explicit_secret and explicit_passphrase:
                print("[executor] Using explicit API credentials from .env")
                self._api_creds = ApiCreds(
                    api_key=explicit_key,
                    api_secret=explicit_secret,
                    api_passphrase=explicit_passphrase,
                )
            else:
                print("[executor] Deriving API credentials from private key")
                self._api_creds = self.client.create_or_derive_api_key()

            self.client.set_api_creds(self._api_creds)
            print("[executor] API credentials set.")

            self._initialized = True
            return True

        except Exception as e:
            print(f"[executor] Initialization failed: {e}")
            return False

    def get_collateral_balance(self) -> float:
        """Return collateral balance in dollars (6-decimal precision)."""
        if not self.client:
            return 100.0

        try:
            sig_type = int(os.getenv("SIGNATURE_TYPE", "3"))
            params = BalanceAllowanceParams(
                asset_type=AssetType.COLLATERAL,
                signature_type=sig_type,
            )
            result = self.client.get_balance_allowance(params)
            balance_raw = result.get("balance", "0")
            return float(balance_raw) / 1e6
        except Exception as e:
            print(f"[executor] Balance check failed: {e}")
            return 0.0

    def get_fee_rate_bps(self, condition_id: str) -> int:
        """Return cached taker fee rate in basis points for a condition."""
        now = time.time()
        if condition_id in self._fee_cache:
            if now - self._fee_cache_ts.get(condition_id, 0) < 300:
                return self._fee_cache[condition_id]

        meta = self.refresh_market_meta(condition_id)
        if meta:
            fee = meta.fee_rate_bps
            self._fee_cache[condition_id] = fee
            self._fee_cache_ts[condition_id] = now
            return fee
        return 0

    def refresh_market_meta(self, condition_id: str) -> Optional[MarketMeta]:
        """Fetch fresh market metadata from CLOB v2."""
        if self.dry_run or not self.client:
            return None

        try:
            # get_market returns a dict with metadata
            data = self.client.get_market(condition_id=condition_id)
            if not data:
                return None

            meta = MarketMeta(
                condition_id=condition_id,
                active=data.get("active", True),
                closed=data.get("closed", False),
                fee_rate_bps=int(data.get("fee_rate_bps", "0")),
                min_order_size=data.get("min_order_size", "5"),
                min_tick_size=data.get("min_tick_size", "0.01"),
                tokens=data.get("tokens", []),
            )
            self._market_meta_cache[condition_id] = meta
            return meta
        except Exception as e:
            print(f"[executor] Market meta fetch failed: {e}")
            return None

    def is_market_tradable(self, condition_id: str) -> bool:
        """Check if a market is active and not closed."""
        meta = self._market_meta_cache.get(condition_id)
        if not meta:
            meta = self.refresh_market_meta(condition_id)
        if meta:
            return meta.active and not meta.closed
        return True  # Default permissive if we can't check

    def get_market_price(self, token_id: str, side: str = "BUY", amount: float = 1.0,
                         retries: int = 5) -> float:
        """Probe market price for a token with retry. Always fetches live — no cached fallback."""
        if not self.client:
            print("[executor] No CLOB client — cannot fetch live price")
            return 0.0

        last_err = None
        for attempt in range(retries):
            try:
                book = self.client.get_order_book(token_id)
                # v2 client returns dicts; v1 returns objects. Handle both.
                asks = book.get("asks", []) if isinstance(book, dict) else getattr(book, "asks", []) or []
                bids = book.get("bids", []) if isinstance(book, dict) else getattr(book, "bids", []) or []
                def _ep(e):
                    """Extract price from order book entry (dict or object)."""
                    return float(e["price"]) if isinstance(e, dict) else float(e.price)
                if side.upper() == "BUY":
                    if asks:
                        return _ep(min(asks, key=_ep))
                else:
                    if bids:
                        return _ep(max(bids, key=_ep))
                # Empty book — try again
                last_err = Exception(f"Empty {'asks' if side.upper() == 'BUY' else 'bids'} on book")
            except Exception as e:
                last_err = e
            if attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1))  # 2s, 4s, 6s, 8s backoff

        print(f"[executor] Price probe failed after {retries} attempts: {last_err}")
        return 0.0

    def calculate_order_size(self, amount_usd: float, price: float, fee_rate_bps: int = 0) -> tuple[float, float]:
        """Calculate shares and cost including fees.

        Returns (shares, total_cost).
        """
        if price <= 0:
            return 0.0, 0.0

        shares = amount_usd / price
        fee_multiplier = 1 + (fee_rate_bps / 10_000)
        total_cost = amount_usd * fee_multiplier
        return shares, total_cost

    def apply_fee_to_cost(self, amount: float, fee_rate_bps: int) -> float:
        """Add taker fee to a cost amount."""
        return amount * (1 + fee_rate_bps / 10_000)

    def apply_fee_to_payout(self, amount: float, fee_rate_bps: int) -> float:
        """Subtract taker fee from a payout amount."""
        return amount * (1 - fee_rate_bps / 10_000)

    def buy(self, token_id: str, amount_usd: float, price: float, fee_rate_bps: int = 0) -> dict:
        """Execute a BUY order (or simulate in dry-run)."""
        if price > MAX_BUY_PRICE:
            return {"status": FAILED, "error": f"Price {price} exceeds max {MAX_BUY_PRICE}"}

        shares, total_cost = self.calculate_order_size(amount_usd, price, fee_rate_bps)

        if self.dry_run:
            print(f"[DRY-RUN] BUY {shares:.2f} shares @ {price:.2f} (cost=${total_cost:.2f})")
            return {
                "status": FILLED,
                "shares": shares,
                "price": price,
                "cost": total_cost,
                "dry_run": True,
            }

        if not self._initialized or not self.client:
            return {"status": FAILED, "error": "Executor not initialized"}

        try:
            rounded_price = round(price, 2)
            rounded_shares = float(int(shares))

            if rounded_shares < 1:
                return {"status": FAILED, "error": f"Order size too small: {rounded_shares}"}

            order_args = OrderArgsV2(
                token_id=token_id,
                price=rounded_price,
                size=rounded_shares,
                side=BUY,
            )

            resp = self.client.create_and_post_order(order_args)
            print(f"[executor] Order response: {resp}")
            if resp and resp.get("success"):
                return {
                    "status": FILLED,
                    "shares": rounded_shares,
                    "price": rounded_price,
                    "cost": total_cost,
                    "order_id": resp.get("orderID", ""),
                }
            else:
                return {"status": FAILED, "error": resp.get("error", "Unknown")}

        except Exception as e:
            print(f"[executor] Buy failed: {e}")
            return {"status": FAILED, "error": str(e)}

    def sell(self, token_id: str, shares: float, price: float, fee_rate_bps: int = 0, force: bool = False) -> dict:
        """Execute a SELL order (or simulate in dry-run).
        
        Args:
            force: If True, bypass minimum notional check (for stop-loss exits).
        """
        notional = shares * price
        if not force and notional < MIN_ORDER_SIZE:
            return {"status": FAILED, "error": f"Notional ${notional:.2f} below minimum ${MIN_ORDER_SIZE}"}

        if self.dry_run:
            print(f"[DRY-RUN] SELL {shares:.2f} shares @ {price:.2f}")
            return {
                "status": FILLED,
                "shares": shares,
                "price": price,
                "revenue": notional,
                "dry_run": True,
            }

        if not self._initialized or not self.client:
            return {"status": FAILED, "error": "Executor not initialized"}

        try:
            rounded_shares = float(int(shares))
            if rounded_shares < 1:
                return {"status": FAILED, "error": f"Sell size too small: {rounded_shares}"}

            order_args = OrderArgsV2(
                token_id=token_id,
                price=round(price, 2),
                size=rounded_shares,
                side=SELL,
            )

            resp = self.client.create_and_post_order(order_args)

            if resp and resp.get("success"):
                return {
                    "status": FILLED,
                    "shares": rounded_shares,
                    "price": price,
                    "revenue": notional,
                    "order_id": resp.get("orderID", ""),
                }
            else:
                return {"status": FAILED, "error": resp.get("error", "Unknown")}

        except Exception as e:
            print(f"[executor] Sell failed: {e}")
            return {"status": FAILED, "error": str(e)}

    def cancel_all_orders(self) -> dict:
        """Cancel all open orders."""
        if self.dry_run or not self.client:
            return {"cancelled": 0, "dry_run": True}

        try:
            resp = self.client.cancel_all()
            return {"cancelled": len(resp) if resp else 0}
        except Exception as e:
            print(f"[executor] Cancel all failed: {e}")
            return {"cancelled": 0, "error": str(e)}

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a specific order."""
        if self.dry_run or not self.client:
            return True

        try:
            self.client.cancel(order_id=order_id)
            return True
        except Exception as e:
            print(f"[executor] Cancel failed: {e}")
            return False
