import os
import json
import time
import hmac
import hashlib
import re
import requests
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import subprocess

# ===== Config =====
MEXC_BASE = "https://api.mexc.com"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"
CG_BASE = "https://api.coingecko.com/api/v3"
STATE_FILE = "data/signals.json"
MODEL_FILE = "data/model.pkl"

GEMINI_KEY = os.environ["GEMINI_API_KEY"]
TG_TOKEN = "8766961406:AAEikTWIpdxMjjUEfd6qW-79o2zgz_95gvw"
TG_CHAT = os.environ["TELEGRAM_CHAT_ID"]
ADMIN_CHAT = os.environ.get("ADMIN_CHAT_ID", "")
MEXC_SECRET = os.environ.get("MEXC_SECRET_KEY", "")
MEXC_ACCESS = os.environ.get("MEXC_ACCESS_KEY", "")
CG_KEY = os.environ.get("COINGECKO_API_KEY", "")

TOP_N_BY_VOLUME = 1000
MAX_WORKERS = 10

# Gemini free-tier quota: limit: 0 per minute is effectively ~15/min and ~1500/day.
GEMINI_DAILY_LIMIT = 30
GEMINI_TOP_N_PER_RUN = 5
GEMINI_USAGE_FILE = "data/gemini_usage.json"

# Rule-stats learning: per-rule win/loss tracking. After every CLOSED signal,
# the bot counts which 'reasons' (e.g. 'RSI overbought', 'EMA up') led to a WIN
# or LOSS. Each rule's score weight is then nudged by its empirical win-rate
# (bounded ±0.10). Lightweight, transparent learning loop — no sklearn, no
# separate training step. Stats are persisted to data/rule_stats.json and
# auto-committed by the bot.
RULE_STATS_FILE = "data/rule_stats.json"
RULE_ADJUSTMENT_MAX = 0.10
RULE_MIN_SAMPLES = 3

# ===== Telegram =====
def tg_send(text, to_admin=False):
    target = ADMIN_CHAT if to_admin else TG_CHAT
    if not target: return False
    url = TELEGRAM_API.format(token=TG_TOKEN, method="sendMessage")
    try:
        r = requests.post(url, json={"chat_id": target, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}, timeout=15)
        return r.json().get("ok", False)
    except Exception as e:
        print(f"TG error: {e}")
        return False

# ===== Gemini quota tracking =====
def gemini_quota_today():
    if not os.path.exists(GEMINI_USAGE_FILE): return 0
    try:
        with open(GEMINI_USAGE_FILE) as f:
            data = json.load(f)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if data.get("date") != today: return 0
        return int(data.get("count", 0))
    except Exception: return 0

def gemini_quota_bump(n=1):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    current = 0
    if os.path.exists(GEMINI_USAGE_FILE):
        try:
            with open(GEMINI_USAGE_FILE) as f:
                data = json.load(f)
            if data.get("date") == today: current = int(data.get("count", 0))
        except Exception: pass
    os.makedirs("data", exist_ok=True)
    with open(GEMINI_USAGE_FILE, "w") as f:
        json.dump({"date": today, "count": current + n}, f)

# ===== Rule-stats learning =====
def _normalize_reason(reason):
    """Bucket dynamic reasons: 'Momentum +3.2%' -> 'Momentum'."""
    r = reason.strip()
    if r.startswith("Momentum"): return "Momentum"
    if r.startswith("Volume"): return "Volume"
    if r.startswith("BB"): return "Bollinger"
    if r.startswith("EMA"): return "EMA"
    if r.startswith("RSI"): return "RSI"
    if r.startswith("BTC"): return "BTC"
    return r

def load_rule_stats():
    if not os.path.exists(RULE_STATS_FILE): return {}
    try:
        with open(RULE_STATS_FILE) as f: return json.load(f)
    except Exception: return {}

def save_rule_stats(stats):
    os.makedirs("data", exist_ok=True)
    with open(RULE_STATS_FILE, "w") as f: json.dump(stats, f, indent=2)

def learn_from_outcomes(signals):
    """Walk all closed signals, count wins/losses per rule, persist stats."""
    stats = load_rule_stats()
    new_outcomes = 0
    for sig in signals:
        if sig.get("status") in (None, "OPEN"): continue
        if sig.get("OUTCOME_LOGGED"): continue
        is_win = "WIN" in sig.get("status", "")
        for r in sig.get("reasons", []):
            if r == "base": continue
            key = _normalize_reason(r)
            entry = stats.get(key, {"wins": 0, "losses": 0, "samples": 0})
            entry["wins" if is_win else "losses"] += 1
            entry["samples"] += 1
            stats[key] = entry
        sig["OUTCOME_LOGGED"] = True
        new_outcomes += 1
    if new_outcomes > 0:
        save_rule_stats(stats)
    return new_outcomes, stats

def rule_weight_adjustments():
    """Return {normalized_rule: delta_weight} based on empirical win-rate.
    Bounded ±RULE_ADJUSTMENT_MAX. Returns empty for rules with <MIN_SAMPLES."""
    stats = load_rule_stats()
    adj = {}
    for rule, s in stats.items():
        if s["samples"] < RULE_MIN_SAMPLES: continue
        win_rate = s["wins"] / s["samples"]
        delta = (win_rate - 0.5) * 2 * RULE_ADJUSTMENT_MAX
        adj[rule] = round(delta, 4)
    return adj

# ===== MEXC signed requests =====
def mexc_signed_request(method, endpoint, params=None):
    if not MEXC_SECRET or not MEXC_ACCESS: return None
    params = params or {}
    params["timestamp"] = int(time.time() * 1000)
    params["recvWindow"] = 5000
    query_string = "&".join(f"{k}={params[k]}" for k in sorted(params.keys()))
    signature = hmac.new(MEXC_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    headers = {"X-MEXC-APIKEY": MEXC_ACCESS, "Content-Type": "application/json"}
    try:
        url = f"{MEXC_BASE}{endpoint}?{query_string}&signature={signature}"
        if method == "GET": r = requests.get(url, headers=headers, timeout=15)
        else: r = requests.post(url, headers=headers, timeout=15)
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        print(f"MEXC signed error: {e}")
        return None

def get_mexc_server_time():
    try:
        r = requests.get(f"{MEXC_BASE}/api/v3/time", timeout=10)
        return r.json().get("serverTime", int(time.time() * 1000))
    except Exception: return int(time.time() * 1000)

# ===== MEXC public data =====
def get_top_usdt_tickers(limit=TOP_N_BY_VOLUME):
    try:
        r = requests.get(f"{MEXC_BASE}/api/v3/ticker/24hr", timeout=20)
        all_t = r.json()
    except Exception as e:
        print(f"MEXC ticker error: {e}"); return []
    usdt = []
    for t in all_t:
        if not t.get("symbol", "").endswith("USDT"): continue
        try:
            vol = float(t.get("quoteVolume", 0) or 0)
            ch24 = float(t.get("priceChangePercent", 0) or 0)
        except: continue
        if vol <= 0: continue
        usdt.append({"t": t, "vol": vol, "ch24": ch24})
    usdt.sort(key=lambda x: x["vol"], reverse=True)
    return usdt[:limit]

def get_1h_change(symbol):
    try:
        r = requests.get(f"{MEXC_BASE}/api/v3/klines",
                         params={"symbol": symbol, "interval": "60m", "limit": 2}, timeout=10)
        k = r.json()
        if len(k) >= 2:
            prev_close = float(k[0][4]); last_close = float(k[1][4])
            if prev_close > 0: return (last_close - prev_close) / prev_close * 100
    except Exception: pass
    return None

def get_tickers():
    top = get_top_usdt_tickers(TOP_N_BY_VOLUME)
    if not top: return []
    FLOOR = 0.03
    candidates_24h = [x for x in top if abs(x["ch24"]) >= FLOOR]
    print(f"  Top {len(top)} by volume, {len(candidates_24h)} pass 24h >= {FLOOR}%")
    return [(x["t"], x["ch24"], None) for x in candidates_24h]

def get_klines(symbol, interval="60m", limit=100):
    try:
        r = requests.get(f"{MEXC_BASE}/api/v3/klines", params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=15)
        return r.json()
    except Exception as e:
        print(f"Klines error {symbol}: {e}"); return []

# ===== CoinGecko =====
CG_CACHE = {}
CG_RATE_LIMITED_UNTIL = 0

def get_coin_context(symbol):
    global CG_CACHE, CG_RATE_LIMITED_UNTIL
    if CG_RATE_LIMITED_UNTIL > time.time(): return None
    if symbol in CG_CACHE: return CG_CACHE[symbol]
    try:
        r = requests.get(f"{CG_BASE}/coins/markets",
                         params={"vs_currency": "usd", "ids": symbol.lower(),
                                 "order": "market_cap_desc", "per_page": 1, "page": 1, "sparkline": False},
                         timeout=10)
        data = r.json()
        if not data: return None
        coin = data[0]
        CG_CACHE[symbol] = {"change_30d": coin.get("price_change_percentage_30d", 0), "rank": coin.get("market_cap_rank", None)}
        return CG_CACHE[symbol]
    except Exception as e:
        print(f"CoinGecko error {symbol}: {e}"); return None

# ===== Indicators =====
def rsi(closes, period=14):
    if len(closes) < period + 1: return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    ups = sum(d for d in deltas if d > 0); downs = sum(-d for d in deltas if d < 0)
    rs = ups / downs if downs != 0 else 0
    return 100 - (100 / (1 + rs))

def ema(data, period=9):
    if len(data) < period: return None
    k = 2 / (period + 1)
    ema_val = sum(data[:period]) / period
    for price in data[period:]: ema_val = price * k + ema_val * (1 - k)
    return ema_val

def bollinger(closes, period=20, std_dev=2):
    if len(closes) < period: return None, None, None
    sma = ema(closes, period)
    if sma is None: return None, None, None
    std = (sum((x - sma) ** 2 for x in closes[-period:]) / period) ** 0.5
    return sma + std_dev * std, sma, sma - std_dev * std

def atr(highs, lows, closes, period=14):
    if len(highs) < period: return None
    trs = []
    for i in range(1, len(highs)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        trs.append(max(hl, hc, lc))
    if not trs: return None
    return sum(trs[-period:]) / period

def volume_spike(volumes):
    if len(volumes) < 20: return 1.0
    avg = sum(volumes[-20:]) / 20
    if avg == 0: return 1.0
    return volumes[-1] / avg

def score_setup(indicators):
    """Score a coin setup (0.0–1.0). Base 0.20; weights are nudged by historical
    win-rate per rule (see rule_weight_adjustments)."""
    adj = rule_weight_adjustments()
    score = 0.20; reasons = ["base"]

    rsi_v = indicators.get("rsi", 50)
    rsi_w = 0.20 + adj.get("RSI", 0.0)
    if rsi_v < 30: score += rsi_w; reasons.append("RSI oversold")
    elif rsi_v > 70: score += rsi_w; reasons.append("RSI overbought")

    ema_trend = indicators.get("ema_trend", "neutral")
    ema_w = 0.15 + adj.get("EMA", 0.0)
    if ema_trend == "up": score += ema_w; reasons.append("EMA up")
    elif ema_trend == "down": score -= ema_w; reasons.append("EMA down")

    bb_pos = indicators.get("bb_position", "middle")
    bb_w = 0.10 + adj.get("Bollinger", 0.0)
    if bb_pos == "above_upper": score += bb_w; reasons.append("BB above upper")
    elif bb_pos == "below_lower": score += bb_w; reasons.append("BB below lower")

    vol_ratio = indicators.get("volume_ratio", 1.0)
    vol_w = 0.10 + adj.get("Volume", 0.0)
    if vol_ratio >= 2.0: score += vol_w + 0.05; reasons.append("Volume spike")
    elif vol_ratio >= 1.5: score += vol_w; reasons.append("Volume high")

    mom_1h = indicators.get("momentum_1h", 0)
    if abs(mom_1h) >= 1.0: score += 0.10; reasons.append(f"Momentum {mom_1h:+.1f}%")

    btc_24h = indicators.get("btc_24h", 0)
    btc_w = 0.10 + adj.get("BTC", 0.0)
    if btc_24h < -2.0: score -= btc_w; reasons.append("BTC down >2%")
    elif btc_24h > 2.0: score += btc_w; reasons.append("BTC up >2%")

    return max(0.0, min(1.0, score)), reasons

def market_allows_side(side, market_ctx):
    btc_24h = market_ctx.get("btc_24h", 0)
    if side == "LONG" and btc_24h < -2.0: return False, "BTC down >2% (LONG not allowed)"
    if side == "SHORT" and btc_24h > 2.0: return False, "BTC up >2% (SHORT not allowed)"
    return True, "OK"

def gemini_decide(symbol, side, indicators, market_ctx):
    prompt = f"""Given the following crypto setup, should we take a {side} position on {symbol}? Answer YES or NO.

Indicators:
- RSI: {indicators.get('rsi', 50)}
- EMA trend: {indicators.get('ema_trend', 'neutral')}
- Bollinger position: {indicators.get('bb_position', 'middle')}
- Volume ratio: {indicators.get('volume_ratio', 1.0)}
- 1h momentum: {indicators.get('momentum_1h', 0)}%
- BTC 24h: {market_ctx.get('btc_24h', 0)}%

Answer YES or NO."""
    try:
        r = requests.post(f"{GEMINI_URL}?key={GEMINI_KEY}",
                          json={"contents": [{"parts": [{"text": prompt}]}]},
                          timeout=20)
        data = r.json()
        if "candidates" not in data or not data["candidates"]:
            print(f"  Gemini advisory unavailable: {data.get('error', data) if isinstance(data, dict) else 'no candidates'}")
            return True, "advisory_unavailable"
        gemini_quota_bump(1)
        reply = data["candidates"][0]["content"]["parts"][0]["text"].strip().upper()
        return ("YES" in reply), reply
    except Exception as e:
        print(f"  Gemini error: {e}")
        return True, f"error:{e}"

# ===== Main scan =====
def scan_market():
    print(f"[{datetime.now(timezone.utc).isoformat()}] Scanning MEXC (top {TOP_N_BY_VOLUME} by volume)...")
    candidates = get_tickers()
    FLOOR = 0.03
    print(f"Found {len(candidates)} volatile coins (24h>={FLOOR}%)")

    market_ctx = get_btc_eth_context()
    print(f"BTC 24h: {market_ctx.get('btc_24h', 0):+.2f}%, trend: {market_ctx.get('btc_trend')}")
    print(f"Gemini quota used today: {gemini_quota_today()}/{GEMINI_DAILY_LIMIT}")

    # Print current learned adjustments
    adj = rule_weight_adjustments()
    if adj:
        print("Learned weight adjustments:")
        for k, v in adj.items(): print(f"  {k:12s} delta={v:+.3f}")
    else:
        print("No learned adjustments yet (need >=3 samples per rule).")

    scored = []
    cg_calls = 0
    SCORE_FLOOR = 0.05

    for t, ch24, ch1h in candidates:
        symbol = t["symbol"]
        klines = get_klines(symbol)
        if len(klines) < 50: continue
        closes = [float(k[4]) for k in klines]
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        volumes = [float(k[5]) for k in klines]
        last_close = closes[-1]
        rsi_v = rsi(closes) or 50
        ema9 = ema(closes[-30:], 9)
        ema21 = ema(closes[-60:], 21)
        ema_trend = "up" if ema9 > ema21 else "down"
        bb_u, bb_m, bb_l = bollinger(closes)
        bb_position = "above_upper" if last_close > bb_u else "below_lower" if last_close < bb_l else "middle"
        atr_v = atr(highs, lows, closes) or (last_close * 0.02)
        vol_ratio = volume_spike(volumes)
        mom_1h = ((closes[-1] - closes[-2]) / closes[-2] * 100) if len(closes) > 1 else 0
        indicators = {
            "rsi": rsi_v, "ema_trend": ema_trend, "bb_position": bb_position,
            "volume_ratio": vol_ratio, "momentum_1h": mom_1h,
            "btc_24h": market_ctx.get("btc_24h", 0),
            "cg_30d": None, "cg_mcap_rank": None,
        }
        if cg_calls < 5:
            cg_data = get_coin_context(symbol)
            if cg_data:
                indicators["cg_30d"] = cg_data.get("change_30d")
                indicators["cg_mcap_rank"] = cg_data.get("rank")
                cg_calls += 1
        score, reasons = score_setup(indicators)
        if score < SCORE_FLOOR: continue
        side = "LONG" if (rsi_v < 50 or ema_trend == "up") else "SHORT"
        allowed, block_reason = market_allows_side(side, market_ctx)
        if not allowed: continue
        scored.append({
            "score": score, "side": side, "symbol": symbol,
            "last_close": last_close, "atr_v": atr_v,
            "ch24": ch24, "ch1h": ch1h,
            "indicators": indicators, "reasons": reasons,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:GEMINI_TOP_N_PER_RUN]
    print(f"  {len(scored)} coins pass score floor; consulting Gemini on top {len(top)}")

    quota_used = gemini_quota_today()
    quota_left = max(0, GEMINI_DAILY_LIMIT - quota_used)
    gemini_slots = min(len(top), quota_left)

    signals = []
    gemini_calls_made = 0
    for i, c in enumerate(top):
        if i < gemini_slots:
            gem_ok, gem_reply = gemini_decide(c["symbol"], c["side"], c["indicators"], market_ctx)
            gemini_calls_made += 1
            c["gemini_reply"] = gem_reply
            if not gem_ok:
                print(f"  {c['symbol']} {c['side']} (score {c['score']:.2f}) -> Gemini NO: {gem_reply[:50]}")
                continue
        else:
            c["gemini_reply"] = "skipped:quota"

        side = c["side"]; last_close = c["last_close"]; atr_v = c["atr_v"]
        if side == "LONG":
            sl = last_close - atr_v * 1.5
            tp1 = last_close + atr_v * 1.0; tp2 = last_close + atr_v * 2.0; tp3 = last_close + atr_v * 3.0
        else:
            sl = last_close + atr_v * 1.5
            tp1 = last_close - atr_v * 1.0; tp2 = last_close - atr_v * 2.0; tp3 = last_close - atr_v * 3.0
        sig = {
            "id": f"{c['symbol']}_{int(time.time())}",
            "ts": datetime.now(timezone.utc).isoformat(),
            "symbol": c["symbol"], "side": side, "entry": last_close,
            "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
            "score": c["score"], "reasons": c["reasons"],
            "indicators": {k: v for k, v in c["indicators"].items() if k != "cg_30d" or v is not None},
            "btc_context": {"btc_24h": market_ctx.get("btc_24h"), "btc_trend": market_ctx.get("btc_trend")},
            "trigger": "1h" if c["ch1h"] is not None else "24h",
            "ch24": c["ch24"], "ch1h": c["ch1h"],
            "gemini": c["gemini_reply"],
            "status": "OPEN",
        }
        signals.append(sig)
        print(f"  ✓ {side} {c['symbol']} score={c['score']:.2f} entry={last_close:.6g} (gemini: {c['gemini_reply'][:30]})")

    print(f"Generated {len(signals)} signals (CG calls: {cg_calls}, Gemini calls: {gemini_calls_made}/{GEMINI_TOP_N_PER_RUN})")
    return signals

def check_outcomes(open_signals):
    updated = []
    for sig in open_signals:
        symbol = sig["symbol"]; old_status = sig.get("status")
        if old_status in ("CLOSED_WIN_TP1", "CLOSED_WIN_TP2", "CLOSED_WIN_TP3", "CLOSED_LOSS", "EXPIRED"):
            continue
        side = sig["side"]; sl = sig["sl"]; tp1 = sig["tp1"]; tp2 = sig["tp2"]; tp3 = sig["tp3"]
        klines = get_klines(symbol, interval="60m", limit=2)
        if len(klines) < 2: continue
        h = float(klines[-1][2]); l = float(klines[-1][3]); c = float(klines[-1][4])
        new_status = None
        if side == "LONG":
            if h >= tp3: new_status = "CLOSED_WIN_TP3"
            elif h >= tp2: new_status = "CLOSED_WIN_TP2"
            elif h >= tp1: new_status = "CLOSED_WIN_TP1"
            elif l <= sl: new_status = "CLOSED_LOSS"
        else:
            if l <= tp3: new_status = "CLOSED_WIN_TP3"
            elif l <= tp2: new_status = "CLOSED_WIN_TP2"
            elif l <= tp1: new_status = "CLOSED_WIN_TP1"
            elif h >= sl: new_status = "CLOSED_LOSS"
        if new_status:
            sig["status"] = new_status
            sig["close_ts"] = datetime.now(timezone.utc).isoformat()
            sig["close_price"] = c
            updated.append(sig)
    return updated

def load_signals():
    if not os.path.exists(STATE_FILE): return []
    try:
        with open(STATE_FILE) as f: return json.load(f)
    except: return []

def save_signals(sigs):
    os.makedirs("data", exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(sigs, f, indent=2, default=str)

def broadcast_signals(signals, to_admin=False):
    if not signals:
        if to_admin: tg_send("0 signals updated this run.", to_admin=True)
        return
    text = "<b>🚀 TA-BOT ML SIGNALS</b>\n\n"
    for sig in signals:
        if "close_price" in sig:
            emoji = "✅" if "WIN" in sig["status"] else "❌"
            text += f"{emoji} <b>{sig['side']} {sig['symbol']}</b> → {sig['status']}\n"
            text += f"Entry: {sig['entry']:.6g} → Close: {sig['close_price']:.6g}\n\n"
        else:
            emoji = "🟢" if sig["side"] == "LONG" else "🔴"
            text += f"{emoji} <b>{sig['side']} {sig['symbol']}</b> ({sig['trigger']}: {sig.get('ch24', 0) or sig.get('ch1h', 0):+.1f}%)\n"
            text += f"Entry: {sig['entry']:.6g} | SL: {sig['sl']:.6g}\n"
            text += f"TP1: {sig['tp1']:.6g} | TP2: {sig['tp2']:.6g} | TP3: {sig['tp3']:.6g}\n"
            text += f"Score: {sig['score']:.2f} ({', '.join(sig['reasons'])})\n\n"
    tg_send(text, to_admin=to_admin)

def get_btc_eth_context():
    if not CG_KEY: return {"btc_24h": 0, "btc_trend": "neutral"}
    try:
        r = requests.get(f"{CG_BASE}/simple/price",
                         params={"ids": "bitcoin,ethereum", "vs_currencies": "usd", "include_24hr_change": "true"},
                         headers={"x-cg-demo-api-key": CG_KEY}, timeout=10)
        data = r.json()
        btc_24h = data.get("bitcoin", {}).get("usd_24h_change", 0)
        btc_trend = "up" if btc_24h > 0 else "down" if btc_24h < 0 else "neutral"
        return {"btc_24h": btc_24h, "btc_trend": btc_trend}
    except: return {"btc_24h": 0, "btc_trend": "neutral"}

if __name__ == "__main__":
    signals = scan_market()
    if signals:
        broadcast_signals(signals)
        save_signals(signals)
    open_signals = load_signals()
    updated = check_outcomes(open_signals)
    if updated:
        broadcast_signals(updated, to_admin=True)
        save_signals(updated)
        print(f"Updated {len(updated)} signals")
    # Learning loop: log rule outcomes from ALL closed signals
    all_signals = load_signals()
    new_outcomes, stats = learn_from_outcomes(all_signals)
    if new_outcomes:
        print(f"Learned from {new_outcomes} new closed signal(s). Rule stats:")
        cur_adj = rule_weight_adjustments()
        for rule, s in sorted(stats.items(), key=lambda x: -x[1]["samples"]):
            wr = s["wins"] / s["samples"] * 100 if s["samples"] else 0
            print(f"  {rule:12s} {s['wins']:3d}W / {s['losses']:3d}L  ({wr:5.1f}% win)  delta={cur_adj.get(rule, 0):+.3f}")
    else:
        print("No new closed signals to learn from yet.")
    print("Done.")
