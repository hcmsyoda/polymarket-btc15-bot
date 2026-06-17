import csv
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import re

now = datetime(2026, 6, 15, 11, 12, 34, tzinfo=timezone.utc)
cutoff_8h = now - timedelta(hours=8)  # 03:12:34 UTC
dry_run_start = datetime(2026, 6, 12, 4, 15, 0, tzinfo=timezone.utc)

def parse_ts(ts_str):
    try:
        ts = float(ts_str.strip())
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except:
        return None

# === TRADES ===
with open("logs/trades.csv") as f:
    reader = csv.DictReader(f)
    all_trades = list(reader)

print(f"Total trades: {len(all_trades)}")

all_w = 0
all_l = 0
all_pnl = 0.0
last8h_w = 0
last8h_l = 0
last8h_pnl = 0.0
last8h_count = 0
last8h_hours = defaultdict(int)
last8h_hour_pnl = defaultdict(float)

for t in all_trades:
    ts = parse_ts(t.get("timestamp", ""))
    won_str = t.get("won_resolution", "").strip().lower()
    try:
        profit = float(t.get("profit", "0").strip())
    except:
        profit = 0.0
    
    won = won_str == "true"
    lost = won_str == "false"
    
    all_pnl += profit
    if won: all_w += 1
    elif lost: all_l += 1
    
    if ts and ts >= cutoff_8h:
        last8h_count += 1
        last8h_pnl += profit
        if won: last8h_w += 1
        elif lost: last8h_l += 1
        hour = ts.hour
        last8h_hours[hour] += 1
        last8h_hour_pnl[hour] += profit

# === SIGNALS ===
with open("logs/signals.csv") as f:
    reader = csv.DictReader(f)
    all_signals = list(reader)

print(f"Total signals: {len(all_signals)}")

skip_reasons = defaultdict(int)
total_skipped = 0
total_traded = 0
last8h_skipped = 0
last8h_traded = 0
total_signals_last8h = 0
windows_seen_unique = set()
last8h_windows_seen = set()

for s in all_signals:
    ts = parse_ts(s.get("timestamp", ""))
    action = s.get("action", "").strip().lower()
    skip_reason = s.get("skip_reason", "").strip()
    window_ts = s.get("window_ts", "").strip()
    
    if window_ts:
        windows_seen_unique.add(window_ts)
    
    if skip_reason:
        total_skipped += 1
        skip_reasons[skip_reason] += 1
    elif action == "buy":
        total_traded += 1
    
    if ts and ts >= cutoff_8h:
        total_signals_last8h += 1
        if window_ts:
            last8h_windows_seen.add(window_ts)
        if skip_reason:
            last8h_skipped += 1
        elif action == "buy":
            last8h_traded += 1

# === Bankroll from log ===
with open("dry_run_output.log") as f:
    lines = f.readlines()

bank = None
unclaimed = None
for line in reversed(lines):
    m = re.search(r'Bank:\s*\$?([\d.]+)', line)
    if m and bank is None:
        bank = float(m.group(1))
    m2 = re.search(r'Unclaimed:\s*\$?([\d.]+)', line)
    if m2 and unclaimed is None:
        unclaimed = float(m2.group(1))
    if bank is not None and unclaimed is not None:
        break

# Edge analysis
edges = []
for t in all_trades:
    try:
        edges.append(float(t.get("edge_at_entry", "0").strip()))
    except:
        pass

# Resolution methods
res_methods = defaultdict(int)
for t in all_trades:
    rm = t.get("resolution_method", "unknown").strip()
    res_methods[rm] += 1

# Hour analysis
all_hour_counts = defaultdict(int)
all_hour_wins = defaultdict(int)
all_hour_losses = defaultdict(int)
for t in all_trades:
    ts = parse_ts(t.get("timestamp", ""))
    if ts:
        h = ts.hour
        all_hour_counts[h] += 1
        won_str = t.get("won_resolution", "").strip().lower()
        if won_str == "true": all_hour_wins[h] += 1
        elif won_str == "false": all_hour_losses[h] += 1

# Errors from log
error_count = 0
error_lines = []
for line in lines[-200:]:
    if "error" in line.lower() or "timeout" in line.lower() or "failed" in line.lower():
        error_count += 1
        error_lines.append(line.strip()[:120])

runtime_hours = (now - dry_run_start).total_seconds() / 3600

print(f"\n=== RUNTIME ===")
print(f"Runtime: {runtime_hours:.1f}h / 96h ({runtime_hours/96*100:.0f}%)")

print(f"\n=== LAST 8H ===")
print(f"Trades: {last8h_count}")
print(f"W/L: {last8h_w}/{last8h_l}")
last8h_total = last8h_w + last8h_l
if last8h_total > 0:
    print(f"WR: {last8h_w/last8h_total*100:.1f}%")
else:
    print("WR: N/A")
print(f"P&L: ${last8h_pnl:+.2f}")
print(f"Signals: {total_signals_last8h}")
print(f"Skipped/Traded: {last8h_skipped}/{last8h_traded}")
print(f"Hours distribution: {dict(last8h_hours)}")
print(f"Hour P&L: {dict(last8h_hour_pnl)}")

print(f"\n=== CUMULATIVE ===")
print(f"Trades: {len(all_trades)}")
print(f"W/L: {all_w}/{all_l}")
all_total = all_w + all_l
if all_total > 0:
    print(f"WR: {all_w/all_total*100:.1f}%")
else:
    print("WR: N/A")
print(f"P&L: ${all_pnl:+.2f}")
print(f"Unique windows seen: {len(windows_seen_unique)}")
print(f"Signals skipped/traded: {total_skipped}/{total_traded}")
print(f"Skip reasons: {dict(skip_reasons)}")

print(f"\n=== EDGE ===")
if edges:
    print(f"Avg edge: {sum(edges)/len(edges):.4f}")
    print(f"Min edge: {min(edges):.4f}")
    print(f"Max edge: {max(edges):.4f}")

print(f"\n=== RESOLUTION METHODS ===")
print(dict(res_methods))

print(f"\n=== HOUR ANALYSIS (ALL TRADES) ===")
for h in sorted(all_hour_counts.keys()):
    total = all_hour_wins[h] + all_hour_losses[h]
    wr = all_hour_wins[h]/total*100 if total > 0 else 0
    print(f"  {h:02d}:00 UTC: {all_hour_counts[h]} trades, {all_hour_wins[h]}W/{all_hour_losses[h]}L, WR: {wr:.0f}%")

print(f"\n=== 07-16 UTC vs OTHER ===")
in_range_trades = sum(all_hour_counts[h] for h in range(7,17))
in_range_wins = sum(all_hour_wins[h] for h in range(7,17))
in_range_losses = sum(all_hour_losses[h] for h in range(7,17))
other_trades = len(all_trades) - in_range_trades
other_wins = all_w - in_range_wins
other_losses = all_l - in_range_losses
if in_range_trades > 0:
    print(f"  07-16 UTC: {in_range_trades} trades, {in_range_wins}W/{in_range_losses}L, WR: {in_range_wins/in_range_trades*100:.0f}%")
else:
    print("  07-16 UTC: 0 trades")
if other_trades > 0:
    print(f"  Other hours: {other_trades} trades, {other_wins}W/{other_losses}L, WR: {other_wins/other_trades*100:.0f}%")
else:
    print("  Other hours: 0 trades")

print(f"\n=== BANKROLL ===")
if bank:
    print(f"Bank: ${bank:.2f}")
else:
    print("Bank: unknown")
if unclaimed:
    print(f"Unclaimed: ${unclaimed:.2f}")
else:
    print("Unclaimed: unknown")
if bank:
    realized_pnl = bank - 101.73
    print(f"Realized P&L (from $101.73): ${realized_pnl:+.2f}")

print(f"\n=== ERRORS (last 200 log lines) ===")
print(f"Error count: {error_count}")
for e in error_lines[-5:]:
    print(f"  {e}")

# Last 8h best hours
print(f"\n=== LAST 8H HOUR PERFORMANCE ===")
for h in sorted(last8h_hour_pnl.keys()):
    print(f"  {h:02d}:00 UTC: {last8h_hours[h]} trades, P&L: ${last8h_hour_pnl[h]:+.2f}")
