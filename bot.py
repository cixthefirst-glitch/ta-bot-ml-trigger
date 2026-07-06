import os
import json
import time
import hmac
import hashlib
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
ADMIN_CHAT = os.environ.get("ADMIN_CHAT_ID", "")  # your personal chat ID for diagnostics
MEXC_SECRET = os.environ.get("MEXC_SECRET_KEY", "")
MEXC_ACCESS = os.environ.get("MEXC_ACCESS_KEY", "")
CG_KEY = os.environ.get("COINGECKO_API_KEY", "")

# Only consider the top N coins by 24h quote volume to keep API calls manageable
TOP_N_BY_VOLUME = 1000
MAX_WORKERS = 10  # parallel 1h change fetches

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
    except Exception:
        return int(time.time() * 1000)

# ===== MEXC public data =====
def get_top_usdt_tickers(limit=TOP_N_BY_VOLUME):
    """Return top USDT pairs by 24h quote volume."""
    try:
        r = requests.get(f"{MEXC_BASE}/api/v3/ticker/24hr", timeout=20)
        all_t = r.json()
    except Exception as e:
        print(f"MEXC ticker error: {e}")
        return []
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
    """Return 1h price change % (last close vs previous close)."""
    try:
        r = requests.get(f"{MEXC_BASE}/api/v3/klines",
                         params={"symbol": symbol, "interval": "1h", "limit": 2}, timeout=10)
        k = r.json()
        if len(k) >= 2:
            prev_close = float(k[0][4])
            last_close = float(k[1][4])
            if prev_close > 0:
                return (last_close - prev_close) / prev_close * 100
    except Exception:
        pass
    return None

def get_tickers():
    """Return top-volume USDT pairs passing 24h >= 0.03% volatility.
    (Lowered from 0.1% to 0.03% to catch more signals.)"""
    top = get_top_usdt_tickers(TOP_N_BY_VOLUME)
    if not top: return []

    FLOOR = 0.03   # minimum movement to consider a coin (24h) — keeps out dead coins
    candidates_24h = [x for x in top if abs(x["ch24"]) >= FLOOR]
    print(f"  Top {len(top)} by volume, {len(candidates_24h)} pass 24h >= {FLOOR}%")

    # Combine
    out = [(x["t"], x["ch24"], None) for x in candidates_24h]
    return out

def get_klines(symbol, interval="1h", limit=100):
    try:
        r = requests.get(f"{MEXC_BASE}/api/v3/klines", params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=15)
        return r.json()
    except Exception as e:
        print(f"Klines error {symbol}: {e}")
        return []

# ===== CoinGecko =====
CG_CACHE = {}
CG_RATE_LIMITED_UNTIL = 0

# ===== CoinGecko =====
def get_coin_context(symbol):
    """Return CoinGecko context for a symbol: 30d change and market cap rank."""
    global CG_CACHE, CG_RATE_LIMITED_UNTIL
    if CG_RATE_LIMITED_UNTIL > time.time():
        return None
    if symbol in CG_CACHE:
        return CG_CACHE[symbol]
    try:
        if CG_KEY:
            r = requests.get(f"{CG_BASE}/coins/markets", params={"vs_currency": "usd", "ids": symbol.lower(), "order": "market_cap_desc", "per_page": 1, "page": 1, "sparkline": False}, timeout=10)
        else:
            r = requests.get(f"{CG_BASE}/coins/markets", params={"vs_currency": "usd", "ids": symbol.lower(), "order": "market_cap_desc", "per_page": 1, "page": 1, "sparkline": False}, timeout=10)
        data = r.json()
        if not data:
            return None
        coin = data[0]
        CG_CACHE[symbol] = {
            "change_30d": coin.get("price_change_percentage_30d", 0),
            "rank": coin.get("market_cap_rank", None),
        }
        return CG_CACHE[symbol]
    except Exception as e:
        print(f"CoinGecko error {symbol}: {e}")
        return None

# ===== Indicators =====
def rsi(closes, period=14):
    """RSI indicator."""
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    ups = sum(d for d in deltas if d > 0)
    downs = sum(-d for d in deltas if d < 0)
    rs = ups / downs if downs != 0 else 0
    return 100 - (100 / (1 + rs))

def ema(data, period=9):
    """Exponential Moving Average."""
    if len(data) < period:
        return None
    k = 2 / (period + 1)
    ema_val = data[:period].mean()
    for price in data[period:]:
        ema_val = price * k + ema_val * (1 - k)
    return ema_val

def bollinger(closes, period=20, std_dev=2):
    """Bollinger Bands."""
    if len(closes) < period:
        return None, None, None
    sma = ema(closes, period)
    if sma is None:
        return None, None, None
    std = (sum((x - sma) ** 2 for x in closes[-period:]) / period) ** 0.5
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower

def atr(highs, lows, closes, period=14):
    """Average True Range."""
    if len(highs) < period:
        return None
    trs = []
    for i in range(1, len(highs)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr = max(hl, hc, lc)
        trs.append(tr)
    if not trs:
        return None
    return sum(trs[-period:]) / period

def volume_spike(volumes):
    """Volume spike detection: 2x average of last 20."""
    if len(volumes) < 20:
        return 1.0
    avg = sum(volumes[-20:]) / 20
    if avg == 0:
        return 1.0
    return volumes[-1] / avg

def score_setup(indicators):
    """Score a coin setup (0.0–1.0)."""
    score = 0.0
    reasons = []

    # RSI: oversold (<30) or overbought (>70) is good for reversal
    rsi_v = indicators.get("rsi", 50)
    if rsi_v < 30:
        score += 0.20
        reasons.append("RSI oversold")
    elif rsi_v > 70:
        score += 0.20
        reasons.append("RSI overbought")

    # EMA trend: up is good for LONG, down is good for SHORT
    ema_trend = indicators.get("ema_trend", "neutral")
    if ema_trend == "up":
        score += 0.15
        reasons.append("EMA up")
    elif ema_trend == "down":
        score -= 0.15
        reasons.append("EMA down")

    # Bollinger position: middle is neutral, above/below extremes is good
    bb_pos = indicators.get("bb_position", "middle")
    if bb_pos == "above_upper":
        score += 0.10
        reasons.append("BB above upper")
    elif bb_pos == "below_lower":
        score += 0.10
        reasons.append("BB below lower")

    # Volume spike: 2x+ average is good
    vol_ratio = indicators.get("volume_ratio", 1.0)
    if vol_ratio >= 2.0:
        score += 0.15
        reasons.append("Volume spike")
    elif vol_ratio >= 1.5:
        score += 0.10
        reasons.append("Volume high")

    # Momentum 1h: strong move is good
    mom_1h = indicators.get("momentum_1h", 0)
    if abs(mom_1h) >= 1.0:
        score += 0.10
        reasons.append(f"Momentum {mom_1h:+.1f}%")

    # BTC context: if BTC is crashing, be cautious
    btc_24h = indicators.get("btc_24h", 0)
    if btc_24h < -2.0:
        score -= 0.10
        reasons.append("BTC down >2%")
    elif btc_24h > 2.0:
        score += 0.10
        reasons.append("BTC up >2%")

    # Cap at 1.0, floor at 0.0
    score = max(0.0, min(1.0, score))
    return score, reasons

def market_allows_side(side, market_ctx):
    """Check if market allows LONG or SHORT (no BTC crash for LONG, no BTC pump for SHORT)."""
    btc_24h = market_ctx.get("btc_24h", 0)
    if side == "LONG" and btc_24h < -2.0:
        return False, "BTC down >2% (LONG not allowed)"
    if side == "SHORT" and btc_24h > 2.0:
        return False, "BTC up >2% (SHORT not allowed)"
    return True, "OK"

def gemini_decide(symbol, side, indicators, market_ctx):
    """Ask Gemini 2.0 Flash for YES/NO on a signal."""
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
        r = requests.post(f"{GEMINI_URL}?key={GEMINI_KEY}", json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=20)
        reply = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip().upper()
        return "YES" in reply
    except Exception as e:
        print(f"Gemini error: {e}")
        return False

# ===== Main scan =====
def scan_market():
    print(f"[{datetime.now(timezone.utc).isoformat()}] Scanning MEXC (top {TOP_N_BY_VOLUME} by volume)...")
    candidates = get_tickers()
    FLOOR = 0.03
    print(f"Found {len(candidates)} volatile coins (24h>={FLOOR}%)")

    market_ctx = get_btc_eth_context()
    print(f"BTC 24h: {market_ctx.get('btc_24h', 0):+.2f}%, trend: {market_ctx.get('btc_trend')}")

    signals = []; cg_calls = 0
    # The main loop handles the <50 klines check inline; no separate pre-filter
    # (the old pre-filter hammered MEXC with 300+ sequential kline calls, hit
    # rate limit, returned empty arrays for everyone, and dropped all 302 candidates)
    for t, ch24, ch1h in candidates:
        symbol = t["symbol"]
        klines = get_klines(symbol)
        if len(klines) < 50: print(f"  {symbol} -> skipped: <50 klines"); continue
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
            "cg_30d": None, "cg_mcap_rank": None,
        }
        if cg_calls < 5:
            cg_data = get_coin_context(symbol)
            if cg_data:
                indicators["cg_30d"] = cg_data.get("change_30d")
                indicators["cg_mcap_rank"] = cg_data.get("rank")
                cg_calls += 1
        score, reasons = score_setup(indicators)
        if score < 0.05: print(f"  {symbol} -> score {score:.2f} (need 0.05+): {reasons}"); continue  # lowered 0.10->0.05 on 2026-07-06 for more volume; was getting 0 signals
        side = "LONG" if (rsi_v < 50 or ema_trend == "up") else "SHORT"
        allowed, block_reason = market_allows_side(side, market_ctx)
        if not allowed: print(f"  {symbol} {side} -> {block_reason}"); continue
        if not gemini_decide(symbol, side, indicators, market_ctx): print(f"  {symbol} {side} -> Gemini rejected"); continue
        if side == "LONG":
            sl = last_close - atr_v * 1.5
            tp1 = last_close + atr_v * 1.0; tp2 = last_close + atr_v * 2.0; tp3 = last_close + atr_v * 3.0
        else:
            sl = last_close + atr_v * 1.5
            tp1 = last_close - atr_v * 1.0; tp2 = last_close - atr_v * 2.0; tp3 = last_close - atr_v * 3.0
        relevant_ch = ch1h if ch1h is not None else ch24
        sig = {
            "id": f"{symbol}_{int(time.time())}",
            "ts": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol, "side": side, "entry": last_close,
            "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
            "score": score, "reasons": reasons,
            "indicators": {k: v for k, v in indicators.items() if k != "cg_30d" or v is not None},
            "btc_context": {"btc_24h": market_ctx.get("btc_24h"), "btc_trend": market_ctx.get("btc_trend")},
            "trigger": "1h" if ch1h is not None else "24h",
            "ch24": ch24, "ch1h": ch1h,
            "status": "OPEN",
        }
        signals.append(sig)
    print(f"Generated {len(signals)} signals (CG calls used: {cg_calls})")
    return signals

def check_outcomes(open_signals):
    updated = []
    for sig in open_signals:
        symbol = sig["symbol"]
        old_status = sig.get("status")
        # already closed
        if old_status in ("CLOSED_WIN_TP1", "CLOSED_WIN_TP2", "CLOSED_WIN_TP3", "CLOSED_LOSS", "EXPIRED"):
            continue
        side = sig["side"]
        entry = sig["entry"]
        sl = sig["sl"]; tp1 = sig["tp1"]; tp2 = sig["tp2"]; tp3 = sig["tp3"]
        klines = get_klines(symbol, interval="1h", limit=2)
        if len(klines) < 2: continue
        # last closed candle
        h = float(klines[-1][2]); l = float(klines[-1][3]); c = float(klines[-1][4])
        new_status = None
        if side == "LONG":
            if h >= tp3: new_status = "CLOSED_WIN_TP3"
            elif h >= tp2: new_status = "CLOSED_WIN_TP2"
            elif h >= tp1: new_status = "CLOSED_WIN_TP1"
            elif l <= sl: new_status = "CLOSED_LOSS"
        else:  # SHORT
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
    """Broadcast signals to Telegram."""
    if not signals:
        if to_admin: tg_send("0 signals updated this run.", to_admin=True)
        return
    text = "<b>🚀 TA-BOT ML SIGNALS</b>\n\n"
    for sig in signals:
        if "close_price" in sig:  # outcome update
            emoji = "✅" if "WIN" in sig["status"] else "❌"
            text += f"{emoji} <b>{sig['side']} {sig['symbol']}</b> → {sig['status']}\n"
            text += f"Entry: {sig['entry']:.6g} → Close: {sig['close_price']:.6g}\n\n"
        else:  # new signal
            emoji = "🟢" if sig["side"] == "LONG" else "🔴"
            text += f"{emoji} <b>{sig['side']} {sig['symbol']}</b> ({sig['trigger']}: {sig.get('ch24', 0) or sig.get('ch1h', 0):+.1f}%)\n"
            text += f"Entry: {sig['entry']:.6g} | SL: {sig['sl']:.6g}\n"
            text += f"TP1: {sig['tp1']:.6g} | TP2: {sig['tp2']:.6g} | TP3: {sig['tp3']:.6g}\n"
            text += f"Score: {sig['score']:.2f} ({', '.join(sig['reasons'])})\n\n"
    tg_send(text, to_admin=to_admin)

def get_btc_eth_context():
    """Get BTC and ETH 24h change from CoinGecko."""
    if not CG_KEY: return {"btc_24h": 0, "eth_24h": 0, "btc_trend": "neutral"}
    try:
        r = requests.get(f"{CG_BASE}/simple/price",
                         params={"ids": "bitcoin,ethereum", "vs_currencies": "usd", "include_24hr_change": "true"},
                         headers={"x-cg-demo-api-key": CG_KEY}, timeout=10)
        data = r.json()
        btc_24h = data.get("bitcoin", {}).get("usd_24h_change", 0)
        btc_trend = "up" if btc_24h > 0 else "down" if btc_24h < 0 else "neutral"
        return {"btc_24h": btc_24h, "btc_trend": btc_trend}
    except:
        return {"btc_24h": 0, "btc_trend": "neutral"}

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
    print("Done.")
