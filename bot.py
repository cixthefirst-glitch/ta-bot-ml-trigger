import os
import json
import time
import hmac
import hashlib
import requests
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor

# ===== Config =====
MEXC_BASE = "https://api.mexc.com"  # spot (kept for server time)
FUTURES_BASE = "https://contract.mexc.com"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"
CG_BASE = "https://api.coingecko.com/api/v3"
STATE_FILE = "data/signals.json"  # active futures signals
SPOT_ARCHIVE_FILE = "data/signals_spot_archive.json"  # archived spot signals
MODEL_FILE = "data/model.pkl"
COEFS_FILE = "data/model_coefs.json"
ML_INFLUENCE = 0.25

GEMINI_KEY = os.environ["GEMINI_API_KEY"]
TG_TOKEN = "8766961406:AAEikTWIpdxMjjUEfd6qW-79o2zgz_95gvw"
TG_CHAT = os.environ["TELEGRAM_CHAT_ID"]
ADMIN_CHAT = os.environ.get("ADMIN_CHAT_ID", "")
MEXC_SECRET = os.environ.get("MEXC_SECRET_KEY", "")
MEXC_ACCESS = os.environ.get("MEXC_ACCESS_KEY", "")
CG_KEY = os.environ.get("COINGECKO_API_KEY", "")

TOP_N_BY_VOLUME = 1000
MAX_WORKERS = 12

GEMINI_DAILY_LIMIT = 30
GEMINI_TOP_N_PER_RUN = 5
GEMINI_USAGE_FILE = "data/gemini_usage.json"

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
        print(f"TG error: {e}"); return False

# ===== Gemini quota tracking =====
def gemini_quota_today():
    if not os.path.exists(GEMINI_USAGE_FILE): return 0
    try:
        with open(GEMINI_USAGE_FILE) as f: data = json.load(f)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if data.get("date") != today: return 0
        return int(data.get("count", 0))
    except Exception: return 0

def gemini_quota_bump(n=1):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    current = 0
    if os.path.exists(GEMINI_USAGE_FILE):
        try:
            with open(GEMINI_USAGE_FILE) as f: data = json.load(f)
            if data.get("date") == today: current = int(data.get("count", 0))
        except Exception: pass
    os.makedirs("data", exist_ok=True)
    with open(GEMINI_USAGE_FILE, "w") as f:
        json.dump({"date": today, "count": current + n}, f)

# ===== Rule-stats learning =====
def _normalize_reason(reason):
    r = reason.strip()
    if r.startswith("Momentum"): return "Momentum"
    if r.startswith("Volume"): return "Volume"
    if r.startswith("BB"): return "Bollinger"
    if r.startswith("EMA"): return "EMA"
    if r.startswith("RSI"): return "RSI"
    if r.startswith("BTC"): return "BTC"
    if r.startswith("funding"): return "Funding"
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
    if new_outcomes > 0: save_rule_stats(stats)
    return new_outcomes, stats

def rule_weight_adjustments():
    stats = load_rule_stats()
    adj = {}
    for rule, s in stats.items():
        if s["samples"] < RULE_MIN_SAMPLES: continue
        win_rate = s["wins"] / s["samples"]
        delta = (win_rate - 0.5) * 2 * RULE_ADJUSTMENT_MAX
        adj[rule] = round(delta, 4)
    return adj

# ===== MEXC signed requests (kept for future use) =====
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
        print(f"MEXC signed error: {e}"); return None

def get_mexc_server_time():
    try:
        r = requests.get(f"{MEXC_BASE}/api/v3/time", timeout=10)
        return r.json().get("serverTime", int(time.time() * 1000))
    except Exception: return int(time.time() * 1000)

# ===== Futures market data =====
_PERP_CACHE = {"ts": 0, "symbols": []}
PERP_CACHE_TTL = 3600

def get_perp_symbols():
    now = time.time()
    if _PERP_CACHE["symbols"] and (now - _PERP_CACHE["ts"]) < PERP_CACHE_TTL:
        return _PERP_CACHE["symbols"]
    try:
        r = requests.get(f"{FUTURES_BASE}/api/v1/contract/detail", timeout=15)
        data = r.json().get("data", [])
        perps = []
        for c in data:
            sym = c.get("symbol", "")
            if not sym.endswith("_USDT"): continue
            if c.get("settleCoin") != "USDT": continue
            dd = c.get("deliveryDate", 0) or 0
            if dd > 0: continue
            perps.append(sym)
        _PERP_CACHE["symbols"] = perps
        _PERP_CACHE["ts"] = now
        return perps
    except Exception as e:
        print(f"Perp list error: {e}"); return _PERP_CACHE.get("symbols", [])

def get_top_futures_tickers(limit=TOP_N_BY_VOLUME):
    perps = get_perp_symbols()
    if not perps: return []
    try:
        r = requests.get(f"{FUTURES_BASE}/api/v1/contract/ticker", timeout=20)
        all_t = r.json().get("data", [])
    except Exception as e:
        print(f"Futures ticker error: {e}"); return []
    out = []
    for t in all_t:
        sym = t.get("symbol", "")
        if sym not in perps: continue
        try:
            vol = float(t.get("volume24", 0) or 0)
            ch24 = float(t.get("riseFallRate", 0) or 0) * 100
            last = float(t.get("lastPrice", 0) or 0)
        except: continue
        if vol <= 0: continue
        out.append({"symbol": sym, "vol": vol, "ch24": ch24, "last": last, "t": t})
    out.sort(key=lambda x: x["vol"], reverse=True)
    return out[:limit]

# Mapping spot-style interval names to futures-style
_FUTURES_INTERVAL = {
    "1m": "Min1", "5m": "Min5", "15m": "Min15", "30m": "Min30",
    "60m": "Min60", "1h": "Hour1", "4h": "Hour4", "1d": "Day1",
}

def get_klines(symbol, interval="60m", limit=100):
    """Fetch klines from MEXC FUTURES. Returns list-of-lists in spot-compatible format:
    [openTime_ms, open, high, low, close, volume]."""
    fut_interval = _FUTURES_INTERVAL.get(interval, "Min60")
    try:
        r = requests.get(f"{FUTURES_BASE}/api/v1/contract/kline/{symbol}",
                         params={"interval": fut_interval, "limit": limit}, timeout=15)
        d = r.json()
        if not d.get("success"): return []
        k = d.get("data", {})
        if not k or not k.get("time"): return []
        result = []
        n = len(k["time"])
        for i in range(n):
            result.append([
                k["time"][i] * 1000,
                float(k["open"][i]),
                float(k["high"][i]),
                float(k["low"][i]),
                float(k["close"][i]),
                float(k["vol"][i]),
            ])
        return result
    except Exception as e:
        print(f"Futures klines error {symbol}: {e}"); return []

def get_tickers():
    top = get_top_futures_tickers(TOP_N_BY_VOLUME)
    if not top: return []
    FLOOR = 0.03
    candidates_24h = [x for x in top if abs(x["ch24"]) >= FLOOR]
    print(f"  Top {len(top)} USDT-M perps by 24h vol, {len(candidates_24h)} pass 24h >= {FLOOR}%")
    return [(x["symbol"], x["ch24"], None) for x in candidates_24h]

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

# ===== Futures context (funding rate) =====
def get_futures_context(symbol):
    """Return {funding_rate, fair_price, idx_price}. Funding rate is 8h decimal
    (e.g. 0.0001 = 0.01%). Used as a contrarian indicator."""
    try:
        r = requests.get(f"{FUTURES_BASE}/api/v1/contract/funding_rate/{symbol}", timeout=8)
        d = r.json()
        if d.get("success") and d.get("data"):
            data = d["data"]
            return {
                "funding_rate": float(data.get("fundingRate", 0) or 0),
                "fair_price": float(data.get("fairPrice", 0) or 0),
                "idx_price": float(data.get("idxPrice", 0) or 0),
            }
    except Exception: pass
    return {"funding_rate": 0, "fair_price": 0, "idx_price": 0}

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

def load_model_coefs():
    if not os.path.exists(COEFS_FILE): return {}
    try:
        with open(COEFS_FILE) as f:
            data = json.load(f)
        if not isinstance(data, dict) or "coefs" not in data: return {}
        return data
    except Exception as e:
        print(f"ML coefs load error: {e}"); return {}

def _sigmoid(z):
    if z >= 0: return 1.0 / (1.0 + (2.718281828 ** -z))
    ez = 2.718281828 ** z
    return ez / (1.0 + ez)

def ml_win_probability(indicators, side, pre_ml_score, coefs_data):
    if not coefs_data: return None
    coefs = coefs_data.get("coefs", {})
    intercept = coefs_data.get("intercept", 0.0)
    if not coefs: return None
    ind = indicators or {}
    features = {
        "rsi": ind.get("rsi", 50) or 50,
        "volume_ratio": ind.get("volume_ratio", 1.0) or 1.0,
        "momentum_1h": ind.get("momentum_1h", 0) or 0,
        "score": pre_ml_score or 0.5,
        "btc_24h": ind.get("btc_24h", 0) or 0,
        "side_long": 1.0 if side == "LONG" else 0.0,
    }
    z = intercept
    for k, v in features.items():
        c = coefs.get(k)
        if c is not None: z += c * v
    return max(0.0, min(1.0, _sigmoid(z)))

def apply_ml_adjustment(score, indicators, side):
    coefs_data = load_model_coefs()
    if not coefs_data: return score
    p = ml_win_probability(indicators, side, score, coefs_data)
    if p is None: return score
    shift = (p - 0.5) * 2 * ML_INFLUENCE
    return max(0.0, min(1.0, score + shift))

def score_setup(indicators, side="LONG"):
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

    # Funding rate (futures only): extreme funding = market over-leveraged
    fund = indicators.get("funding_rate", 0) or 0
    if fund > 0.0005: score -= 0.05; reasons.append("funding very high")
    elif fund < -0.0005: score += 0.05; reasons.append("funding very low")

    return apply_ml_adjustment(max(0.0, min(1.0, score)), indicators, side), reasons

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
- Funding rate (8h): {indicators.get('funding_rate', 0)*100:.4f}%

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
    print(f"[{datetime.now(timezone.utc).isoformat()}] Scanning MEXC USDT-M PERPETUALS (top {TOP_N_BY_VOLUME} by volume)...")
    candidates = get_tickers()
    FLOOR = 0.03
    print(f"Found {len(candidates)} volatile perps (24h>={FLOOR}%)")

    market_ctx = get_btc_eth_context()
    print(f"BTC 24h: {market_ctx.get('btc_24h', 0):+.2f}%, trend: {market_ctx.get('btc_trend')}")
    print(f"Gemini quota used today: {gemini_quota_today()}/{GEMINI_DAILY_LIMIT}")

    adj = rule_weight_adjustments()
    if adj:
        print("Learned weight adjustments:")
        for k, v in adj.items(): print(f"  {k:12s} delta={v:+.3f}")
    else:
        print("No learned adjustments yet (need >=3 samples per rule).")

    scored = []
    cg_calls = 0
    SCORE_FLOOR = 0.05

    # Pre-fetch klines for all candidates in parallel (avoids timeout on slow MEXC responses)
    def fetch_one(item):
        sym, ch24, ch1h = item
        klines = get_klines(sym)
        return (sym, ch24, ch1h, klines)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fetched = list(ex.map(fetch_one, candidates))

    for sym, ch24, ch1h, klines in fetched:
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
            "funding_rate": 0,  # filled below for top candidates
            "cg_30d": None, "cg_mcap_rank": None,
        }
        side = "LONG" if (rsi_v < 50 or ema_trend == "up") else "SHORT"
        score, reasons = score_setup(indicators, side)
        if score < SCORE_FLOOR: continue
        allowed, block_reason = market_allows_side(side, market_ctx)
        if not allowed: continue
        scored.append({
            "score": score, "side": side, "symbol": sym,
            "last_close": last_close, "atr_v": atr_v,
            "ch24": ch24, "ch1h": ch1h,
            "indicators": indicators, "reasons": reasons,
        })

    # Fetch funding only for the top N that will go to Gemini (saves API calls)
    for c in scored[:GEMINI_TOP_N_PER_RUN]:
        ctx = get_futures_context(c["symbol"])
        c["indicators"]["funding_rate"] = ctx["funding_rate"]
    # Re-score with funding context now in hand
    for c in scored:
        new_score, new_reasons = score_setup(c["indicators"], c.get("side", "LONG"))
        c["score"] = new_score; c["reasons"] = new_reasons

    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:GEMINI_TOP_N_PER_RUN]
    print(f"  {len(scored)} perps pass score floor; consulting Gemini on top {len(top)}")

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

    print(f"Generated {len(signals)} signals (Gemini calls: {gemini_calls_made}/{GEMINI_TOP_N_PER_RUN})")
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
    text = "<b>🚀 TA-BOT ML SIGNALS (Futures)</b>\n\n"
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

def archive_spot_signals_if_needed():
    """One-time migration: if data/signals.json has spot-format symbols
    (no underscore) and SPOT_ARCHIVE_FILE doesn't exist yet, archive them."""
    if not os.path.exists(STATE_FILE): return
    if os.path.exists(SPOT_ARCHIVE_FILE): return
    try:
        with open(STATE_FILE) as f: sigs = json.load(f)
        if not isinstance(sigs, list): return
        # Spot symbols are like "XRBUSDT" (no underscore); futures are "XRB_USDT"
        has_spot = any("_" not in s.get("symbol", "") for s in sigs)
        if has_spot and sigs:
            with open(SPOT_ARCHIVE_FILE, "w") as f:
                json.dump(sigs, f, indent=2, default=str)
            print(f"Archived {len(sigs)} spot signals to {SPOT_ARCHIVE_FILE}")
            with open(STATE_FILE, "w") as f: json.dump([], f)
    except Exception as e:
        print(f"Archive error: {e}")

if __name__ == "__main__":
    archive_spot_signals_if_needed()
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
