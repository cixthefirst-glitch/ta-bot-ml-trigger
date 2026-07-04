"""
Backtest the scoring strategy against historical MEXC data.
Walks through N days of 1h klines for the top coins by recent volatility,
applies the same scoring+SL/TP rules as bot.py, and reports win rate, avg R, drawdown.
Sends a summary to Telegram.
"""
import os
import json
import time
import requests
from datetime import datetime, timezone, timedelta

MEXC_BASE = "https://api.mexc.com"
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "")
ADMIN_CHAT = os.environ.get("ADMIN_CHAT_ID", "")

# Same as bot.py
VOLATILITY_THRESHOLD = 5.0      # % change in last 24h to consider
BACKTEST_DAYS = 14               # how many days to walk
SCORE_MIN = 0.4
KLINE_INTERVAL = "1h"
KLINE_LIMIT = 500                # ~20 days of 1h candles

# ===== Indicators (mirrors bot.py) =====
def rsi(closes, period=14):
    if len(closes) < period + 1: return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i-1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def ema(values, period):
    if not values: return None
    k = 2 / (period + 1)
    ema_val = values[0]
    for v in values[1:]:
        ema_val = v * k + ema_val * (1 - k)
    return ema_val

def bollinger(closes, period=20, std_mult=2):
    if len(closes) < period: return None, None, None
    sma = sum(closes[-period:]) / period
    variance = sum((c - sma) ** 2 for c in closes[-period:]) / period
    std = variance ** 0.5
    return sma + std_mult * std, sma, sma - std_mult * std

def atr(highs, lows, closes, period=14):
    if len(closes) < period + 1: return None
    trs = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        trs.append(tr)
    return sum(trs[-period:]) / period

def volume_spike(volumes, period=20):
    if len(volumes) < period: return 1.0
    avg = sum(volumes[-period:-1]) / (period - 1)
    return volumes[-1] / avg if avg > 0 else 1.0

def score_setup(indicators):
    score = 0.0
    rsi_v = indicators["rsi"]
    if rsi_v is not None:
        if rsi_v < 25: score += 0.35
        elif rsi_v < 35: score += 0.2
        elif rsi_v > 75: score += 0.35
        elif rsi_v > 65: score += 0.2
    if indicators["ema_trend"] in ("up", "down"): score += 0.15
    if indicators["bb_position"] in ("below_lower", "above_upper"): score += 0.2
    if indicators["volume_ratio"] > 2.0: score += 0.15
    if abs(indicators["momentum_1h"]) > 2: score += 0.1
    return score

def compute_indicators_at(closes, highs, lows, volumes, idx):
    """Slice klines up to idx, compute indicators as if 'now' is at idx."""
    if idx < 60: return None
    sub_c = closes[:idx+1]
    sub_h = highs[:idx+1]
    sub_l = lows[:idx+1]
    sub_v = volumes[:idx+1]
    last_close = sub_c[-1]
    rsi_v = rsi(sub_c)
    ema9 = ema(sub_c[-30:], 9)
    ema21 = ema(sub_c[-60:], 21)
    ema_trend = "up" if ema9 > ema21 else "down"
    bb_u, bb_m, bb_l = bollinger(sub_c)
    bb_position = "above_upper" if last_close > bb_u else "below_lower" if last_close < bb_l else "middle"
    atr_v = atr(sub_h, sub_l, sub_c) or (last_close * 0.02)
    vol_ratio = volume_spike(sub_v)
    mom_1h = ((sub_c[-1] - sub_c[-2]) / sub_c[-2] * 100) if len(sub_c) > 1 else 0
    return {
        "rsi": rsi_v or 50,
        "ema_trend": ema_trend,
        "bb_position": bb_position,
        "volume_ratio": vol_ratio,
        "momentum_1h": mom_1h,
        "atr": atr_v,
        "last_close": last_close,
    }

def evaluate_trade(side, entry, sl, tp1, tp2, tp3, future_highs, future_lows):
    """Walk forward through future candles, return outcome string."""
    for h, l in zip(future_highs, future_lows):
        if side == "LONG":
            if l <= sl: return "LOSS", -1.0
            if h >= tp3: return "WIN_TP3", 3.0
            if h >= tp2: return "WIN_TP2", 2.0
            if h >= tp1: return "WIN_TP1", 1.0
        else:  # SHORT
            if h >= sl: return "LOSS", -1.0
            if l <= tp3: return "WIN_TP3", 3.0
            if l <= tp2: return "WIN_TP2", 2.0
            if l <= tp1: return "WIN_TP1", 1.0
    return "OPEN_AT_END", 0.0

# ===== MEXC data =====
def get_top_volatile_coins(limit=30):
    try:
        r = requests.get(f"{MEXC_BASE}/api/v3/ticker/24hr", timeout=20)
        data = r.json()
        candidates = [
            t for t in data
            if t.get("symbol", "").endswith("USDT")
            and abs(float(t.get("priceChangePercent", 0))) >= VOLATILITY_THRESHOLD
        ]
        # If too few, relax to top-volume
        if len(candidates) < 10:
            usdt = [t for t in data if t.get("symbol", "").endswith("USDT")]
            usdt.sort(key=lambda t: float(t.get("quoteVolume", 0)), reverse=True)
            return [t["symbol"] for t in usdt[:limit]]
        candidates.sort(key=lambda t: abs(float(t.get("priceChangePercent", 0))), reverse=True)
        return [t["symbol"] for t in candidates[:limit]]
    except Exception as e:
        print(f"Ticker error: {e}")
        return []

def get_klines(symbol, limit=KLINE_LIMIT):
    try:
        r = requests.get(
            f"{MEXC_BASE}/api/v3/klines",
            params={"symbol": symbol, "interval": KLINE_INTERVAL, "limit": limit},
            timeout=20,
        )
        return r.json()
    except Exception as e:
        print(f"Klines error {symbol}: {e}")
        return []

# ===== Telegram =====
def tg_send(text, to_admin=False):
    if not TG_TOKEN: return
    target = ADMIN_CHAT if to_admin else TG_CHAT
    if not target: return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": target, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}, timeout=15)
    except Exception as e:
        print(f"TG error: {e}")

# ===== Main backtest =====
def run_backtest():
    print(f"[{datetime.now(timezone.utc).isoformat()}] Starting backtest ({BACKTEST_DAYS}d, score>={SCORE_MIN})...")
    symbols = get_top_volatile_coins(limit=30)
    print(f"Scanning {len(symbols)} symbols")

    trades = []
    start_time = time.time()
    for sym in symbols:
        klines = get_klines(sym)
        if len(klines) < 100: continue
        closes = [float(k[4]) for k in klines]
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        volumes = [float(k[5]) for k in klines]

        # Walk: at every hour between now-BACKTEST_DAYS and now-1h, score and simulate
        hours_total = len(klines)
        walk_hours = min(BACKTEST_DAYS * 24, hours_total - 70)  # leave 10 future bars for outcome
        # Start at idx=60 (need history), step every 4h to avoid redundancy
        for idx in range(60, hours_total - 10, 4):
            ind = compute_indicators_at(closes, highs, lows, volumes, idx)
            if ind is None: continue
            sc = score_setup(ind)
            if sc < SCORE_MIN: continue
            side = "LONG" if (ind["rsi"] < 50 or ind["ema_trend"] == "up") else "SHORT"
            entry = ind["last_close"]
            atr_v = ind["atr"]
            if side == "LONG":
                sl = entry - atr_v * 1.5
                tp1, tp2, tp3 = entry + atr_v, entry + atr_v * 2, entry + atr_v * 3
            else:
                sl = entry + atr_v * 1.5
                tp1, tp2, tp3 = entry - atr_v, entry - atr_v * 2, entry - atr_v * 3
            fut_h = highs[idx+1:idx+11]
            fut_l = lows[idx+1:idx+11]
            outcome, r_mult = evaluate_trade(side, entry, sl, tp1, tp2, tp3, fut_h, fut_l)
            if outcome == "OPEN_AT_END": continue
            trades.append({
                "symbol": sym, "side": side, "entry": entry, "outcome": outcome,
                "r": r_mult, "score": sc, "ts": idx,
            })
        if time.time() - start_time > 480:  # 8min cap
            print("Time cap hit, stopping")
            break

    if not trades:
        msg = "🧪 <b>Backtest</b>: no qualifying trades (market too quiet)"
        print(msg)
        tg_send(msg, to_admin=True)
        return

    wins = [t for t in trades if t["outcome"].startswith("WIN")]
    losses = [t for t in trades if t["outcome"] == "LOSS"]
    win_rate = len(wins) / len(trades) * 100
    avg_r = sum(t["r"] for t in trades) / len(trades)
    total_r = sum(t["r"] for t in trades)

    # Max drawdown (cumulative R curve)
    cum_r = 0
    peak = 0
    max_dd = 0
    for t in trades:
        cum_r += t["r"]
        peak = max(peak, cum_r)
        dd = peak - cum_r
        max_dd = max(max_dd, dd)

    # Per-coin breakdown
    per_coin = {}
    for t in trades:
        per_coin.setdefault(t["symbol"], {"wins": 0, "losses": 0, "r": 0})
        if t["outcome"].startswith("WIN"): per_coin[t["symbol"]]["wins"] += 1
        else: per_coin[t["symbol"]]["losses"] += 1
        per_coin[t["symbol"]]["r"] += t["r"]

    top_coins = sorted(per_coin.items(), key=lambda x: x[1]["r"], reverse=True)[:5]
    worst_coins = sorted(per_coin.items(), key=lambda x: x[1]["r"])[:5]

    # By TP level
    tp1_wins = len([t for t in wins if t["outcome"] == "WIN_TP1"])
    tp2_wins = len([t for t in wins if t["outcome"] == "WIN_TP2"])
    tp3_wins = len([t for t in wins if t["outcome"] == "WIN_TP3"])

    summary = f"""🧪 <b>Backtest — last {BACKTEST_DAYS} days</b>

Trades: <b>{len(trades)}</b>
Wins: {len(wins)} ({win_rate:.1f}%)
Losses: {len(losses)}
Avg R: <b>{avg_r:+.2f}R</b>
Total R: <b>{total_r:+.1f}R</b>
Max drawdown: {max_dd:.1f}R

TP breakdown:
  TP1: {tp1_wins}
  TP2: {tp2_wins}
  TP3: {tp3_wins}

🏆 Top coins:
""" + "\n".join(f"  {s}: {v['wins']}W/{v['losses']}L, {v['r']:+.1f}R" for s, v in top_coins) + "\n\n💀 Worst coins:\n" + "\n".join(f"  {s}: {v['wins']}W/{v['losses']}L, {v['r']:+.1f}R" for s, v in worst_coins)

    print(summary)
    tg_send(summary, to_admin=True)

    # Save full results
    os.makedirs("data", exist_ok=True)
    with open("data/backtest_results.json", "w") as f:
        json.dump({
            "ts": datetime.now(timezone.utc).isoformat(),
            "trades_count": len(trades),
            "win_rate": win_rate,
            "avg_r": avg_r,
            "total_r": total_r,
            "max_dd": max_dd,
            "trades": trades,
        }, f, indent=2)
    print("Saved to data/backtest_results.json")

if __name__ == "__main__":
    run_backtest()
