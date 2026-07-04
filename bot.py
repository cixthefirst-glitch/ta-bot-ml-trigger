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
TG_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TG_CHAT = os.environ["TELEGRAM_CHAT_ID"]
ADMIN_CHAT = os.environ.get("ADMIN_CHAT_ID", "")  # your personal chat ID for diagnostics
MEXC_SECRET = os.environ.get("MEXC_SECRET_KEY", "")
MEXC_ACCESS = os.environ.get("MEXC_ACCESS_KEY", "")
CG_KEY = os.environ.get("COINGECKO_API_KEY", "")

# ===== Volatility thresholds (multi-armed bandit) =====
# The bandit learns which volatility threshold is actually profitable
# by tracking outcomes of every signal. Weights are persisted to the
# repo so they survive GitHub Actions' ephemeral runner restarts.
THRESHOLDS = [0.5, 1.0, 2.0, 3.0, 5.0]   # the 5 arms of the bandit
# Prior bias: 5% starts favored (user's trusted setting), low thresholds underweighted
# until the data proves otherwise.
PRIOR_WEIGHTS = {"0.5": 0.4, "1.0": 0.6, "2.0": 0.8, "3.0": 1.2, "5.0": 2.0}
# Multiplicative update on each resolved signal
LEARN_TP1 = 1.15
LEARN_TP2 = 1.25
LEARN_TP3 = 1.40
LEARN_LOSS = 0.75
LEARN_TIMEOUT = 0.92
WEIGHTS_FILE = "data/threshold_weights.json"
# Only consider the top N coins by 24h quote volume to keep API calls manageable
TOP_N_BY_VOLUME = 400  # bumped from 200 for more candidates / bandit learning data
MAX_WORKERS = 10  # parallel 1h change fetches

# ===== Bandit: load/save/select/update =====
def _load_weights_raw():
    """Load weights from local file or return prior defaults."""
    if not os.path.exists(WEIGHTS_FILE):
        return dict(PRIOR_WEIGHTS)
    try:
        with open(WEIGHTS_FILE) as f:
            data = json.load(f)
        w = data.get("weights", {})
        for t in THRESHOLDS:
            w.setdefault(str(t), PRIOR_WEIGHTS[str(t)])
        return w
    except Exception as e:
        print(f"[BANDIT] load error: {e}; using prior")
        return dict(PRIOR_WEIGHTS)


def _renormalize(weights):
    """Keep weights bounded so one runaway winner doesn't kill exploration."""
    mx = max(weights.values())
    mn = min(weights.values())
    if mx > 50.0 or mn < 0.01:
        scale = 10.0 / mx if mx > 0 else 1.0
        return {k: max(0.05, v * scale) for k, v in weights.items()}
    return weights


def select_threshold(coin_ch24):
    """Pick a threshold arm using epsilon-greedy on learned weights.
    90% exploit (highest weight), 10% explore (random arm)."""
    weights = _load_weights_raw()
    if random.random() < 0.20:   # bumped 0.10 -> 0.20 on 2026-07-03 (more exploration while bandit is young)
        return random.choice(THRESHOLDS), weights
    best_t = max(THRESHOLDS, key=lambda t: weights.get(str(t), 1.0))
    return best_t, weights


def update_bandit(threshold_fired, outcome):
    """Update the weight for one arm based on a resolved signal's outcome."""
    weights = _load_weights_raw()
    t = str(threshold_fired)
    if t not in weights:
        weights[t] = PRIOR_WEIGHTS.get(t, 1.0)
    if outcome == "CLOSED_WIN_TP1":
        weights[t] *= LEARN_TP1
    elif outcome == "CLOSED_WIN_TP2":
        weights[t] *= LEARN_TP2
    elif outcome == "CLOSED_WIN_TP3":
        weights[t] *= LEARN_TP3
    elif outcome == "CLOSED_LOSS":
        weights[t] *= LEARN_LOSS
    elif outcome == "TIMEOUT":
        weights[t] *= LEARN_TIMEOUT
    else:
        return None
    weights = _renormalize(weights)
    _save_weights_raw(weights)
    print(f"[BANDIT] {t}% -> {outcome} | weight: {weights[t]:.3f}")
    return weights


def _save_weights_raw(weights):
    """Save weights to local file, then best-effort-commit to repo."""
    os.makedirs(os.path.dirname(WEIGHTS_FILE), exist_ok=True)
    payload = {
        "weights": weights,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(WEIGHTS_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    try:
        subprocess.run(["git", "add", WEIGHTS_FILE], cwd=".", capture_output=True, timeout=10)
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet", WEIGHTS_FILE],
            capture_output=True, timeout=10,
        )
        if diff.returncode != 0:
            subprocess.run(
                ["git", "commit", "-m", f"bandit: update weights", WEIGHTS_FILE],
                cwd=".", capture_output=True, timeout=10,
            )
            subprocess.run(
                ["git", "push", "origin", "main"],
                cwd=".", capture_output=True, timeout=30,
            )
    except Exception as e:
        print(f"[BANDIT] git commit skipped: {e}")


def bandit_report():
    """Return a human-readable summary of current weights."""
    weights = _load_weights_raw()
    sigs = load_signals()
    by_arm = {str(t): {"wins": 0, "losses": 0, "open": 0, "timeout": 0} for t in THRESHOLDS}
    for s in sigs:
        arm = str(s.get("threshold", ""))
        if arm not in by_arm:
            continue
        st = s.get("status", "")
        if st == "OPEN":
            by_arm[arm]["open"] += 1
        elif "WIN" in st:
            by_arm[arm]["wins"] += 1
        elif st == "CLOSED_LOSS":
            by_arm[arm]["losses"] += 1
        elif st == "TIMEOUT":
            by_arm[arm]["timeout"] += 1
    lines = ["Bandit Threshold Weights:"]
    for t in THRESHOLDS:
        w = weights.get(str(t), 1.0)
        s = by_arm[str(t)]
        closed = s["wins"] + s["losses"] + s["timeout"]
        wr = (s["wins"] / closed * 100) if closed > 0 else 0
        bar = "#" * int(min(w, 5.0)) + "." * int(max(0, 5.0 - min(w, 5.0)))
        lines.append(f"  {t:>4}% {bar} w={w:.2f}  n={closed} (W:{s['wins']} L:{s['losses']} T:{s['timeout']} O:{s['open']}) WR={wr:.0f}%")
    return "\n".join(lines)



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
    usdt.sort(key=lambda x: -x["vol"])
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
    """Return top-volume USDT pairs passing EITHER 24h >= 0.3% OR 1h >= 0.3%.
    (Lowered from 5%/3% in 2026-07-03 to feed the bandit more candidates.
    The bandit still picks which arm (0.5/1/2/3/5%) plays on each signal.)"""
    top = get_top_usdt_tickers(TOP_N_BY_VOLUME)
    if not top: return []

    FLOOR = 0.3   # was min(THRESHOLDS)=0.5% — lowered 2026-07-03 for more signal volume (bandit needs data)
    candidates_24h = [x for x in top if abs(x["ch24"]) >= FLOOR]
    print(f"  Top {len(top)} by volume, {len(candidates_24h)} pass 24h >= {FLOOR}%")

    rest = [x for x in top if abs(x["ch24"]) < FLOOR]
    candidates_1h = []
    if rest:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {ex.submit(get_1h_change, x["t"]["symbol"]): x for x in rest}
            done = 0
            for fut in as_completed(futs):
                done += 1
                ch1h = fut.result()
                x = futs[fut]
                if ch1h is not None and abs(ch1h) >= FLOOR:
                    candidates_1h.append((x["t"], x["ch24"], ch1h))
                if done % 50 == 0:
                    print(f"  1h check: {done}/{len(rest)} done")
    print(f"  {len(candidates_1h)} pass 1h >= {FLOOR}% (parallel)")

    # Combine
    out = [(x["t"], x["ch24"], None) for x in candidates_24h] + candidates_1h
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

def cg_get(path, params=None, ttl=120):
    global CG_RATE_LIMITED_UNTIL
    if time.time() < CG_RATE_LIMITED_UNTIL: return None
    cache_key = f"{path}:{json.dumps(params or {}, sort_keys=True)}"
    if cache_key in CG_CACHE:
        ts, val = CG_CACHE[cache_key]
        if time.time() - ts < ttl: return val
    headers = {"x-cg-pro-api-key": CG_KEY} if CG_KEY else {}
    try:
        r = requests.get(f"{CG_BASE}{path}", params=params, headers=headers, timeout=15)
        if r.status_code == 200:
            CG_CACHE[cache_key] = (time.time(), r.json())
            return r.json()
        elif r.status_code == 429:
            print(f"CG rate limited (429), backing off 60s")
            CG_RATE_LIMITED_UNTIL = time.time() + 60
            return None
        else:
            print(f"CG error {r.status_code}: {path}")
            return None
    except Exception as e:
        print(f"CG exception: {e}")
        return None

def get_btc_eth_context():
    data = cg_get("/simple/price", {
        "ids": "bitcoin,ethereum",
        "vs_currencies": "usd",
        "include_24hr_change": "true",
    }, ttl=300)
    if not data: return {"btc_24h": 0, "eth_24h": 0, "btc_trend": "neutral"}
    btc_24h = data.get("bitcoin", {}).get("usd_24h_change", 0) or 0
    eth_24h = data.get("ethereum", {}).get("usd_24h_change", 0) or 0
    if btc_24h > 3: btc_trend = "bullish"
    elif btc_24h < -3: btc_trend = "bearish"
    else: btc_trend = "neutral"
    return {"btc_24h": btc_24h, "eth_24h": eth_24h, "btc_trend": btc_trend}

COINGECKO_ID_MAP = {
    "btc": "bitcoin", "eth": "ethereum", "sol": "solana", "bnb": "binancecoin",
    "xrp": "ripple", "doge": "dogecoin", "ada": "cardano", "matic": "matic-network",
    "pol": "matic-network", "dot": "polkadot", "avax": "avalanche-2",
    "link": "chainlink", "uni": "uniswap", "ltc": "litecoin", "trx": "tron",
    "ton": "the-open-network", "shib": "shiba-inu", "pepe": "pepe", "floki": "floki",
    "wif": "dogwifcoin", "bonk": "bonk", "sui": "sui", "apt": "aptos",
    "near": "near", "atom": "cosmos", "xlm": "stellar", "vet": "vechain",
    "fil": "filecoin", "etc": "ethereum-classic", "hbar": "hedera-hashgraph",
    "icp": "internet-computer", "kas": "kaspa", "inj": "injective-protocol",
    "rune": "thorchain", "ldo": "lido-dao", "arb": "arbitrum", "op": "optimism",
    "aave": "aave", "mkr": "maker", "crv": "curve-dao-token",
}

def get_coin_context(symbol):
    base = symbol.replace("USDT", "").lower()
    cg_id = COINGECKO_ID_MAP.get(base, base)
    data = cg_get(f"/coins/{cg_id}", {
        "localization": "false", "tickers": "false", "community_data": "false",
        "developer_data": "false", "sparkline": "false",
    }, ttl=900)
    if not data or "market_data" not in data: return None
    md = data["market_data"]
    return {
        "rank": data.get("market_cap_rank"),
        "mcap": md.get("market_cap", {}).get("usd"),
        "change_24h": md.get("price_change_percentage_24h", 0) or 0,
        "change_7d": md.get("price_change_percentage_7d", 0) or 0,
        "change_30d": md.get("price_change_percentage_30d", 0) or 0,
    }

# ===== Indicators =====
def rsi(closes, period=14):
    if len(closes) < period + 1: return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i-1]
        gains.append(max(diff, 0)); losses.append(max(-diff, 0))
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
    trs = [max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1])) for i in range(1, len(closes))]
    return sum(trs[-period:]) / period

def volume_spike(volumes, period=20):
    if len(volumes) < period: return 1.0
    avg = sum(volumes[-period:-1]) / (period - 1)
    return volumes[-1] / avg if avg > 0 else 1.0

# ===== Signal scoring =====
def score_setup(indicators):
    score = 0.0; reasons = []
    rsi_v = indicators["rsi"]
    if rsi_v is not None:
        if rsi_v < 25: score += 0.35; reasons.append(f"RSI extreme low {rsi_v:.1f}")
        elif rsi_v < 35: score += 0.2; reasons.append(f"RSI low {rsi_v:.1f}")
        elif rsi_v > 75: score += 0.35; reasons.append(f"RSI extreme high {rsi_v:.1f}")
        elif rsi_v > 65: score += 0.2; reasons.append(f"RSI high {rsi_v:.1f}")
    if indicators["ema_trend"] == "up": score += 0.15; reasons.append("EMA uptrend")
    elif indicators["ema_trend"] == "down": score += 0.15; reasons.append("EMA downtrend")
    bb_pos = indicators["bb_position"]
    if bb_pos == "below_lower": score += 0.2; reasons.append("below BB lower")
    elif bb_pos == "above_upper": score += 0.2; reasons.append("above BB upper")
    if indicators["volume_ratio"] > 2.0: score += 0.15; reasons.append(f"vol spike {indicators['volume_ratio']:.1f}x")
    if abs(indicators["momentum_1h"]) > 2: score += 0.1; reasons.append(f"1h mom {indicators['momentum_1h']:+.1f}%")
    if indicators.get("cg_30d") is not None:
        cg_30d = indicators["cg_30d"]
        if indicators["ema_trend"] == "up" and cg_30d > 10: score += 0.1; reasons.append(f"30d up {cg_30d:+.0f}%")
        elif indicators["ema_trend"] == "down" and cg_30d < -10: score += 0.1; reasons.append(f"30d down {cg_30d:+.0f}%")
    if indicators.get("cg_mcap_rank") is not None:
        rank = indicators["cg_mcap_rank"]
        if rank and rank <= 100: score += 0.05; reasons.append(f"top-100 (rank {rank})")
        elif rank and rank > 500: score -= 0.1; reasons.append(f"low rank {rank}")
    return score, reasons

def market_allows_side(side, market_ctx):
    if not market_ctx: return True, ""
    btc_trend = market_ctx.get("btc_trend", "neutral")
    btc_24h = market_ctx.get("btc_24h", 0)
    if btc_trend == "bearish" and side == "LONG" and btc_24h < -2:
        return False, f"blocked: BTC bearish ({btc_24h:+.1f}% 24h)"
    if btc_trend == "bullish" and side == "SHORT" and btc_24h > 2:
        return False, f"blocked: BTC bullish ({btc_24h:+.1f}% 24h)"
    return True, ""

def gemini_decide(symbol, side, indicators, market_ctx):
    ctx_line = ""
    if market_ctx: ctx_line = f" Market: BTC {market_ctx.get('btc_24h', 0):+.1f}% 24h, trend {market_ctx.get('btc_trend')}."
    prompt = f"""You are a strict trading filter. Setup: {side} {symbol}. Indicators: RSI {indicators['rsi']:.1f}, EMA trend {indicators['ema_trend']}, BB position {indicators['bb_position']}, volume ratio {indicators['volume_ratio']:.2f}x, 1h momentum {indicators['momentum_1h']:+.2f}%.{ctx_line} Reply ONLY 'YES' or 'NO'."""
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
    FLOOR = min(THRESHOLDS)
    n_24h = sum(1 for _, ch24, _ in candidates if ch24 is not None and abs(ch24) >= FLOOR)
    n_1h = len(candidates) - n_24h
    print(f"Found {len(candidates)} volatile coins (24h>={FLOOR}%: {n_24h}, 1h>={FLOOR}%: {n_1h})")

    market_ctx = get_btc_eth_context()
    print(f"BTC 24h: {market_ctx.get('btc_24h', 0):+.2f}%, trend: {market_ctx.get('btc_trend')}")

    signals = []; cg_calls = 0; skipped_lowhist = 0
    # Pre-filter: drop candidates that don't have 50+ 1h klines.
    # These are freshly-launched tokens that survived the vol filter on a single
    # 24h move but have no price history to analyze.
    pre_filtered = []
    for cand in candidates[:50]:
        sym = cand[0]["symbol"]
        kc = get_klines(sym, limit=50)
        if len(kc) < 50:
            skipped_lowhist += 1
            continue
        pre_filtered.append(cand)
    if skipped_lowhist:
        print(f"  {skipped_lowhist} skipped (insufficient history <50 1h klines)")
    candidates = pre_filtered
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
        if score < 0.10: print(f"  {symbol} -> score {score:.2f} (need 0.10+): {reasons}"); continue  # lowered 0.15->0.10 on 2026-07-03 for more volume
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
        chosen_t, _weights = select_threshold(relevant_ch)
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
            "threshold": chosen_t,
            "status": "OPEN",
        }
        signals.append(sig)
        print(f"  [BANDIT] armed {chosen_t}% for {symbol} (ch={relevant_ch:+.2f}%)")
    print(f"Generated {len(signals)} signals (CG calls used: {cg_calls})")
    return signals

def check_outcomes(open_signals):
    updated = []
    for sig in open_signals:
        symbol = sig["symbol"]
        old_status = sig.get("status", "OPEN")
        klines = get_klines(symbol, limit=10)
        if not klines: updated.append(sig); continue
        new_status = old_status
        for k in klines[-5:]:
            high = float(k[2]); low = float(k[3])
            if sig["side"] == "LONG":
                if low <= sig["sl"]: new_status = "CLOSED_LOSS"; break
                if high >= sig["tp3"]: new_status = "CLOSED_WIN_TP3"; break
                if high >= sig["tp2"]: new_status = "CLOSED_WIN_TP2"; break
                if high >= sig["tp1"]: new_status = "CLOSED_WIN_TP1"; break
            else:
                if high >= sig["sl"]: new_status = "CLOSED_LOSS"; break
                if low <= sig["tp3"]: new_status = "CLOSED_WIN_TP3"; break
                if low <= sig["tp2"]: new_status = "CLOSED_WIN_TP2"; break
                if low <= sig["tp1"]: new_status = "CLOSED_WIN_TP1"; break
        if new_status != old_status:
            sig["status"] = new_status
            sig["closed_at"] = datetime.now(timezone.utc).isoformat()
            t = sig.get("threshold")
            if t is not None:
                update_bandit(t, new_status)
        updated.append(sig)
    return updated

def load_signals():
    if not os.path.exists(STATE_FILE): return []
    try:
        with open(STATE_FILE) as f: return json.load(f)
    except: return []

def save_signals(sigs):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f: json.dump(sigs, f, indent=2)

def broadcast(sig):
    trigger = sig.get("trigger", "24h")
    ch_show = sig.get("ch1h") if trigger == "1h" else sig.get("ch24")
    msg = f"""{'🟢' if sig['side']=='LONG' else '🔴'} <b>{sig['side']} {sig['symbol']}</b> <i>({trigger}: {ch_show:+.1f}%)</i>
Entry: <code>{sig['entry']:.6f}</code>
SL: <code>{sig['sl']:.6f}</code>
TP1: <code>{sig['tp1']:.6f}</code>
TP2: <code>{sig['tp2']:.6f}</code>
TP3: <code>{sig['tp3']:.6f}</code>
Score: {sig['score']:.2f}
💡 {', '.join(sig['reasons'])}"""
    if sig.get("btc_context"):
        btc = sig["btc_context"]
        msg += f"\n🌐 BTC {btc.get('btc_24h', 0):+.1f}% (24h), trend {btc.get('btc_trend', 'n/a')}"
    tg_send(msg)
    print(f"Broadcasted: {sig['side']} {sig['symbol']} @ {sig['entry']:.6f}")

if __name__ == "__main__":
    print(f"MEXC keys: {'set' if MEXC_SECRET and MEXC_ACCESS else 'NOT set'}")
    print(f"CoinGecko key: {'set' if CG_KEY else 'NOT set'}")
    if MEXC_SECRET and MEXC_ACCESS:
        server_time = get_mexc_server_time()
        print(f"MEXC server time: {server_time} (signed endpoint available)")

    all_signals = load_signals()

    # Time-based timeout: signals open > 24h are marked TIMEOUT (bandit penalty)
    now_ts = time.time()
    timed_out = 0
    for s in all_signals:
        if s.get("status") != "OPEN":
            continue
        try:
            sig_ts = datetime.fromisoformat(s["ts"].replace("Z", "+00:00")).timestamp()
        except Exception:
            continue
        if (now_ts - sig_ts) > 24 * 3600:
            s["status"] = "TIMEOUT"
            s["closed_at"] = datetime.now(timezone.utc).isoformat()
            t = s.get("threshold")
            if t is not None:
                update_bandit(t, "TIMEOUT")
            timed_out += 1
    if timed_out:
        print(f"Timed out {timed_out} stale signal(s)")

    open_signals = [s for s in all_signals if s["status"] == "OPEN"]
    if open_signals:
        open_signals = check_outcomes(open_signals)
        closed = sum(1 for s in open_signals if s["status"] != "OPEN")
        if closed: print(f"Closed {closed} signal(s)")

    new_signals = scan_market()
    for sig in new_signals: broadcast(sig)
    if not new_signals and ADMIN_CHAT:
        try:
            btc_ctx = get_btc_eth_context()
            tg_send(f"🟡 Bot run: 0 signals. Market quiet (BTC {btc_ctx.get('btc_24h', 0):+.2f}%, trend {btc_ctx.get('btc_trend')}). Bot working as designed.", to_admin=True)
        except Exception as e:
            print(f"admin ping failed: {e}")
    by_id = {s["id"]: s for s in all_signals}
    for s in open_signals:
        by_id[s["id"]] = s
    for s in new_signals:
        by_id[s["id"]] = s
    merged = list(by_id.values())
    save_signals(merged)

    report = bandit_report()
    print(report)
    if ADMIN_CHAT:
        try:
            tg_send("Bandit report:\n" + report, to_admin=True)
        except Exception as e:
            print(f"bandit report send failed: {e}")
    print("Done.")
