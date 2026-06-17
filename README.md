# PolyBot — Polymarket BTC 5-Minute Up/Down Trading Bot

A CLOB v2-compatible trading bot for Polymarket's 5-minute BTC Up/Down binary markets.

## Features

- **CLOB v2 API** via `py-clob-client` ≥ 0.34.6
- **Safe/Proxy wallet support** (signature_type=2)
- **Builder fee configuration**
- **Fee-aware strategy** — taker fees subtracted from edge before entry
- **Dry-run mode** (default) — simulate trades without real orders
- **Brownian motion probability model** with calibrated vol=0.12
- **Quarter-Kelly position sizing** with $5–$25 bounds
- **70 automated tests** covering strategy, execution, and market discovery

## Architecture

| File | Role |
|------|------|
| `bot.py` | Main trading loop, position lifecycle, circuit breakers |
| `strategy.py` | Brownian motion probability + Kelly sizing + fee-aware filters |
| `executor.py` | Polymarket CLOB v2 order execution, Safe wallet, builder config |
| `market.py` | Gamma API market discovery, slug generation, token ID parsing |
| `price_feed.py` | Binance WebSocket real-time BTC price feed |
| `tracker.py` | Quant analytics logging (CSV) |
| `telegram_notifier.py` | Mobile alerts + hourly summaries |
| `proxy.py` | Tor proxy for CLOB API geo-restrictions |

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your wallet credentials
python bot.py
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PRIVATE_KEY` | EOA private key (without 0x) | — |
| `WALLET_ADDRESS` | EOA wallet address | — |
| `SAFE_ADDRESS` | Polymarket Safe address (optional) | — |
| `BUILDER_ADDRESS` | CLOB builder address (optional) | — |
| `BUILDER_FEE` | Builder fee (optional) | — |
| `DRY_RUN` | Simulate trades without sending orders | `true` |
| `MIN_EDGE` | Minimum edge (prob - price) | `0.05` |
| `MIN_PROB` | Minimum model probability | `0.80` |
| `MIN_BTC_DELTA` | Minimum BTC delta % to consider | `0.06` |
| `KELLY_FRACTION` | Kelly fraction (0.25 = quarter-Kelly) | `0.25` |
| `MIN_BET` / `MAX_BET` | Per-trade bounds | `5` / `25` |
| `DAILY_LOSS_LIMIT` | Halt trading after this loss | `30.0` |

## Safe Live Trading Checklist

Before switching `DRY_RUN=false`:

1. ✅ `.env` has correct `PRIVATE_KEY` and `WALLET_ADDRESS`
2. ✅ Wallet is funded with USDC on Polygon
3. ✅ `SAFE_ADDRESS` set if using Polymarket Safe
4. ✅ `BUILDER_ADDRESS` and `BUILDER_FEE` set if using a builder
5. ✅ `MIN_EDGE` and `MIN_PROB` calibrated for current market conditions
6. ✅ `DAILY_LOSS_LIMIT` set to your risk tolerance
7. ✅ Run `python -m pytest -q` and confirm all tests pass
8. ✅ Run `python -m py_compile bot.py strategy.py executor.py market.py` to verify syntax

## Testing

```bash
python -m pytest -q
```

## License

MIT
