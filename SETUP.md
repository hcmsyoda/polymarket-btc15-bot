# PolyBot Setup Guide

## Prerequisites

- Python 3.11+
- `pip` and `venv`
- Polygon mainnet USDC in your wallet
- (Optional) Tor for CLOB API geo-restrictions

## 1. Clone and Install

```bash
git clone https://github.com/yourusername/polymarket-bot.git
cd polymarket-bot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:

```
PRIVATE_KEY=your_private_key_without_0x
WALLET_ADDRESS=your_wallet_address
SAFE_ADDRESS=          # leave empty for EOA-only
BUILDER_ADDRESS=       # leave empty if not using builder
BUILDER_FEE=           # leave empty if not using builder
DRY_RUN=true
```

## 3. Wallet Setup

### EOA Wallet

If you don't have a wallet:

```python
from eth_account import Account
acct = Account.create()
print(f"Address: {acct.address}")
print(f"Private Key: {acct.key.hex()}")
```

### Polymarket Safe Wallet

1. Connect to [Polymarket](https://polymarket.com) with your EOA
2. Enable the Polymarket Safe (smart contract wallet)
3. Deposit USDC into the Safe
4. Copy the Safe address to `SAFE_ADDRESS`
5. Set `signature_type=2` in the executor (handled automatically when `SAFE_ADDRESS` is set)

### Funding

- Send USDC (Polygon PoS) to your wallet address
- The bot queries **collateral balance** (not just USDC)
- Minimum recommended: $100–$200

## 4. Builder Configuration (Optional)

Some setups use a builder for order flow:

```
BUILDER_ADDRESS=0xYourBuilderAddress
BUILDER_FEE=100
```

Leave empty if you don't have a builder.

## 5. Run Tests

```bash
python -m pytest -q
```

Expected: **70 passed**

## 6. Dry Run

```bash
python bot.py
```

In dry-run mode the bot:
- Connects to Binance WebSocket for BTC prices
- Queries Gamma API for active markets
- Runs the strategy and prints what it *would* trade
- Does **not** send any orders to Polymarket

## 7. Go Live

After extensive dry-run validation:

1. Set `DRY_RUN=false` in `.env`
2. Re-run `python bot.py`
3. The bot will:
   - Initialize the CLOB v2 client
   - Derive API credentials
   - Start trading real USDC

**Start small.** Even with a validated strategy, begin with minimal bankroll and monitor for several hours.

## 8. CLOB v2 API Notes

- `create_or_derive_api_creds()` returns `ApiCreds(api_key, api_secret, api_passphrase)`
- Credentials are set via `client.set_api_creds(creds)`
- `AssetType.COLLATERAL` is the string `"COLLATERAL"` (not an enum with `.value`)
- Balance API returns 6-decimal strings (`"5000000"` = $5.00)
- `use_server_time` and `retry_on_error` are **not** constructor parameters in the current pip version

## 9. Troubleshooting

### "CLOB API blocked"

The CLOB API blocks POST `/order` from datacenter/VPN IPs. Enable Tor in `proxy.py`:

```python
from proxy import ensure_tor, apply_proxy
proxy_url = ensure_tor()
apply_proxy(proxy_url)
```

### "invalid amounts, max accuracy of 4 decimals"

The executor rounds prices to 2 decimals and shares to integers before creating orders. This avoids float precision artifacts.

### "No active market found"

Polymarket 5-min markets may not exist between windows. The bot waits for the next window automatically.

## 10. Safety Defaults

| Setting | Value | Rationale |
|---------|-------|-----------|
| `DRY_RUN` | `true` | Must be explicitly disabled |
| `MIN_PROB` | `0.80` | 80% model confidence required |
| `MIN_EDGE` | `0.05` | 5% edge after fees |
| `KELLY_FRACTION` | `0.25` | Quarter-Kelly (conservative) |
| `MAX_BET` | `25.0` | Hard cap per trade |
| `DAILY_LOSS_LIMIT` | `30.0` | Halt after $30 loss |
