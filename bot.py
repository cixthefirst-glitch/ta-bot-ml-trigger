"""
TA Bot — Main Scanner & Broadcaster
Runs every hour via GitHub Actions.

Flow:
  1. Fetch top volatile USDT pairs from MEXC
  2. Compute indicators: RSI, EMA, Bollinger, Volume, Momentum, ATR
  3. Run SMC analysis (swing highs/lows, BOS, discount/premium, supply/demand)
  4. Score setup (rule-based)
  5. Apply ML model adjustment (if model.pkl exists)
  6. If score passes threshold → ask Gemini YES/NO
  7. If YES → broadcast to Telegram subscribers + channel
  8. Track TP/SL outcomes on previously open signals
  9. Save everything to data/signals.json
"""

import os
import json
import time
import pickle
import hashlib
import requests
from datetime import datetime, timezone

# ─── ENV ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY      = os.environ.get("GEMINI_API_KEY", "")
TG_TOKEN            = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT             = os.environ.get("TELEGRAM_CHAT_ID", "")
ADMIN_CHAT          = os.environ.get("ADMIN_CHAT_ID", "")
MEXC_BASE           = "https://api.mexc.com"
GEMINI_MODEL        = "gemini-1.5-flash"

# ─── CONFIG ─────────────────────────────────────────────────────────────────
KLINE_INTERVAL      = "1h"
KLINE_LIMIT         = 200
SCAN_LIMIT          = 40
VOLATILITY_MIN      = 3.0
SCORE_THRESHOLD     = 0.45
ATR_SL_MULT         = 1.5
ATR_TP1_MULT        = 1.0
ATR_TP2_MULT        = 2.0
ATR_TP3_MULT        = 3.0
MAX_SIGNALS_PER_RUN = 3
SIGNAL_COOLDOWN_H   = 4
STATE_FILE          = "data/signals.json"
MODEL_FILE          = "data/model.pkl"
COEFS_FILE          = "data/model_coefs.json"
RULE_STATS_FILE     = "data/rule_stats.json"
SUBS_FILE           = "data/subscribers.json"
TRIAL_FILE          = "data/trial_users.json"
PENDING_FILE        = "data/pending_payments.json"

# ─── TELEGRAM ───────────────────────────────────────────────────────────────
def tg_send(text, chat_id=None, to_admin=False):
    if not TG_TOKEN:
        return
    target = chat_id or (ADMIN_CHAT if to_admin else TG_CHAT)
    if not target:
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        requests.post(
            url,
            json={"chat_id": target, "text": text, "parse_mode": "HTML",
                  "disable_web_page_preview": True},
            timeout=15,
        )
    except Exception as e:
        print(f"TG error: {e}")


def broadcast_to_subscribers(text):
    tg_send(text)
    subs = load_json(SUBS_FILE, {})
    now = time.time()
    for uid, info in subs.items():
        if info.get("expires_at", 0) > now:
            tg_send(text, chat_id=uid)


# ─── PERSISTENCE ────────────────────────────────────────────────────────────
def load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ─── MEXC DATA ──────────────────────────────────────────────────────────────
def get_top_volatile_coins(limit=SCAN_LIMIT):
    try:
        r = requests.get(f"{MEXC_BASE}/api/v3/ticker/24hr", timeout=20)
        data = r.json()
        candidates = [
            t for t in data
            if t.get("symbol", "").endswith("USDT")
            and not t["symbol"].startswith("USDC")
            and abs(float(t.get("priceChangePercent", 0))) >= VOLATILITY_MIN
            and float(t.get("quoteVolume", 0)) > 500_000
        ]
        candidates.sort(
            key=lambda t: abs(float(t.get("priceChangePercent", 0))),
            reverse=True,
        )
        return [(t["symbol"], float(t["priceChangePercent"])) for t in candidates[:limit]]
    except Exception as e:
        print(f"Ticker error: {e}")
        return []


def get_klines(symbol, interval=KLINE_INTERVAL, limit=KLINE_LIMIT):
    try:
        r = requests.get(
            f"{MEXC_BASE}/api/v3/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=20,
        )
        return r.json()
    except Exception as e:
        print(f"Klines error {symbol}: {e}")
        return []


def get_btc_context():
    try:
        r = requests.get(
            f"{MEXC_BASE}/api/v3/ticker/24hr",
            params={"symbol": "BTCUSDT"},
            timeout=10,
        )
        d = r.json()
        return {"btc_24h": float(d.get("priceChangePercent", 0))}
    except Exception:
        return {"btc_24h": 0.0}


# ─── INDICATORS ─────────────────────────────────────────────────────────────
def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_ema(values, period):
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    val = sum(values[:period]) / period
    for v in values[period:]:
        val = v * k + val * (1 - k)
    return val


def calc_bollinger(closes, period=20, std_mult=2):
    if len(closes) < period:
        return None, None, None
    sma = sum(closes[-period:]) / period
    variance = sum((c - sma) ** 2 for c in closes[-period:]) / period
    std = variance ** 0.5
    return sma + std_mult * std, sma, sma - std_mult * std


def calc_atr(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    return sum(trs[-period:]) / period


def calc_volume_ratio(volumes, period=20):
    if len(volumes) < period:
        return 1.0
    avg = sum(volumes[-period:-1]) / (period - 1)
    return volumes[-1] / avg if avg > 0 else 1.0


def compute_indicators(klines):
    if len(klines) < 60:
        return None
    closes  = [float(k[4]) for k in klines]
    highs   = [float(k[2]) for k in klines]
    lows    = [float(k[3]) for k in klines]
    volumes = [float(k[5]) for k in klines]

    last_close = closes[-1]
    rsi_v   = calc_rsi(closes)
    ema9    = calc_ema(closes, 9)
    ema21   = calc_ema(closes, 21)
    ema50   = calc_ema(closes, 50)
    bb_u, bb_m, bb_l = calc_bollinger(closes)
    atr_v   = calc_atr(highs, lows, closes) or (last_close * 0.02)
    vol_r   = calc_volume_ratio(volumes)
    mom_1h  = ((closes[-1] - closes[-2]) / closes[-2] * 100) if len(closes) > 1 else 0

    ema_trend    = "up" if (ema9 and ema21 and ema9 > ema21) else "down"
    ema_above_50 = bool(ema21 and ema50 and ema21 > ema50)

    if bb_u and bb_l and last_close:
        if last_close > bb_u:       bb_position = "above_upper"
        elif last_close < bb_l:     bb_position = "below_lower"
        else:                       bb_position = "middle"
    else:
        bb_position = "middle"

    return {
        "rsi":          rsi_v or 50,
        "ema_trend":    ema_trend,
        "ema_above_50": ema_above_50,
        "bb_position":  bb_position,
        "volume_ratio": vol_r,
        "momentum_1h":  mom_1h,
        "atr":          atr_v,
        "last_close":   last_close,
        "ema9":         ema9,
        "ema21":        ema21,
        "ema50":        ema50,
    }


# ─── SMC ────────────────────────────────────────────────────────────────────
MIN_KLINES_SMC = 60
SWING_LOOKBACK = 3
RANGE_LOOKBACK = 50
SWING_MIN_PROM = 0.001


def _swing_highs(highs, lb=SWING_LOOKBACK, min_p=SWING_MIN_PROM):
    swings = []
    n = len(highs)
    for i in range(lb, n - lb):
        if all(highs[i] > highs[i - j] and highs[i] > highs[i + j] for j in range(1, lb + 1)):
            base = min(min(highs[max(0, i - lb * 2):i]), min(highs[i + 1:i + lb * 2 + 1]))
            if base > 0 and (highs[i] - base) / base >= min_p:
                swings.append(i)
    return swings


def _swing_lows(lows, lb=SWING_LOOKBACK, min_p=SWING_MIN_PROM):
    swings = []
    n = len(lows)
    for i in range(lb, n - lb):
        if all(lows[i] < lows[i - j] and lows[i] < lows[i + j] for j in range(1, lb + 1)):
            cap = max(max(lows[max(0, i - lb * 2):i]), max(lows[i + 1:i + lb * 2 + 1]))
            if cap > 0 and (cap - lows[i]) / cap >= min_p:
                swings.append(i)
    return swings


def get_smc_features(klines):
    if not klines or len(klines) < MIN_KLINES_SMC:
        return {}

    highs  = [float(k[2]) for k in klines]
    lows   = [float(k[3]) for k in klines]
    closes = [float(k[4]) for k in klines]
    last_close = closes[-1]

    sl = slice(max(0, len(closes) - RANGE_LOOKBACK), len(closes))
    range_high = max(highs[sl])
    range_low  = min(lows[sl])
    if range_high <= range_low:
        return {}
    range_size   = range_high - range_low
    price_pos    = (last_close - range_low) / range_size
    in_discount  = price_pos < 0.5
    in_premium   = price_pos > 0.5
    in_deep_disc = price_pos < 0.3
    in_deep_prem = price_pos > 0.7

    sh_idx = _swing_highs(highs)
    sl_idx = _swing_lows(lows)

    nearest_swing_high = max((highs[i] for i in sh_idx if highs[i] < last_close), default=None)
    nearest_swing_low  = min((lows[i]  for i in sl_idx if lows[i]  > last_close), default=None)

    bos_direction  = None
    choch_recently = False
    if sh_idx and sl_idx:
        last_sh = highs[sh_idx[-1]]
        last_sl = lows[sl_idx[-1]]
        if last_close > last_sh * 1.0005:        bos_direction = "BULL"
        elif last_close < last_sl * 0.9995:      bos_direction = "BEAR"
        elif last_close > last_sh * 0.995:       bos_direction = "TESTING_HIGH"
        elif last_close < last_sl * 1.005:       bos_direction = "TESTING_LOW"

        breaks = []
        for idx in sh_idx:
            if idx >= len(closes) - 30:
                fut = closes[idx + 1:]
                if fut and all(c > highs[idx] for c in fut):
                    breaks.append((idx, "BULL"))
        for idx in sl_idx:
            if idx >= len(closes) - 30:
                fut = closes[idx + 1:]
                if fut and all(c < lows[idx] for c in fut):
                    breaks.append((idx, "BEAR"))
        breaks.sort()
        if len(breaks) >= 2 and breaks[-1][1] != breaks[-2][1]:
            choch_recently = True

    near_supply = bool(sh_idx and abs(last_close - highs[sh_idx[-1]]) / last_close < 0.01)
    near_demand = bool(sl_idx and abs(last_close - lows[sl_idx[-1]])  / last_close < 0.01)

    ss = 0.0
    if bos_direction == "BULL":           ss += 0.6
    elif bos_direction == "BEAR":         ss -= 0.6
    elif bos_direction == "TESTING_HIGH": ss += 0.1
    elif bos_direction == "TESTING_LOW":  ss -= 0.1
    if choch_recently: ss *= 0.5
    if in_deep_disc:   ss += 0.3
    elif in_discount:  ss += 0.15
    elif in_deep_prem: ss -= 0.3
    elif in_premium:   ss -= 0.15
    if near_demand:    ss += 0.2
    if near_supply:    ss -= 0.2
    ss = max(-1.0, min(1.0, ss))

    parts = []
    if bos_direction == "BULL":           parts.append("Bullish BOS")
    elif bos_direction == "BEAR":         parts.append("Bearish BOS")
    elif bos_direction == "TESTING_HIGH": parts.append("Testing swing high")
    elif bos_direction == "TESTING_LOW":  parts.append("Testing swing low")
    if choch_recently:  parts.append("ChoCH")
    if in_deep_disc:    parts.append("deep discount")
    elif in_discount:   parts.append("discount")
    elif in_deep_prem:  parts.append("deep premium")
    elif in_premium:    parts.append("premium")
    if near_demand:     parts.append("at demand")
    if near_supply:     parts.append("at supply")

    return {
        "bos_direction":      bos_direction,
        "choch_recently":     choch_recently,
        "in_discount":        in_discount,
        "in_premium":         in_premium,
        "in_deep_discount":   in_deep_disc,
        "in_deep_premium":    in_deep_prem,
        "near_supply_zone":   near_supply,
        "near_demand_zone":   near_demand,
        "structure_strength": ss,
        "description":        " | ".join(parts) if parts else "no structure",
        "nearest_swing_high": nearest_swing_high,
        "nearest_swing_low":  nearest_swing_low,
        "recent_range_high":  range_high,
        "recent_range_low":   range_low,
    }


# ─── SCORING ────────────────────────────────────────────────────────────────
def determine_side(ind, smc):
    long_votes  = 0
    short_votes = 0

    rsi_v = ind["rsi"]
    if rsi_v < 40:    long_votes  += 1
    elif rsi_v > 60:  short_votes += 1

    if ind["ema_trend"] == "up":    long_votes  += 1
    elif ind["ema_trend"] == "down": short_votes += 1

    if ind.get("ema_above_50"):  long_votes  += 0.5
    else:                        short_votes += 0.5

    bp = ind["bb_position"]
    if bp == "below_lower":   long_votes  += 1
    elif bp == "above_upper": short_votes += 1

    mom = ind["momentum_1h"]
    if mom > 1:    long_votes  += 0.5
    elif mom < -1: short_votes += 0.5

    bos = smc.get("bos_direction")
    if bos == "BULL":   long_votes  += 1.5
    elif bos == "BEAR": short_votes += 1.5

    if smc.get("in_discount"): long_votes  += 0.5
    if smc.get("in_premium"):  short_votes += 0.5

    total = long_votes + short_votes
    if total == 0:
        return "LONG", 0.5
    if long_votes >= short_votes:
        return "LONG", long_votes / total
    return "SHORT", short_votes / total


def score_setup(ind, smc, side):
    score = 0.0
    fired = []
    rsi_v = ind["rsi"]

    # RSI
    if side == "LONG":
        if rsi_v < 25:   score += 0.35; fired.append("RSI")
        elif rsi_v < 40: score += 0.2;  fired.append("RSI")
    else:
        if rsi_v > 75:   score += 0.35; fired.append("RSI")
        elif rsi_v > 60: score += 0.2;  fired.append("RSI")

    # EMA
    if (side == "LONG" and ind["ema_trend"] == "up") or \
       (side == "SHORT" and ind["ema_trend"] == "down"):
        score += 0.15; fired.append("EMA")

    # Bollinger
    if (side == "LONG" and ind["bb_position"] == "below_lower") or \
       (side == "SHORT" and ind["bb_position"] == "above_upper"):
        score += 0.2; fired.append("Bollinger")

    # Volume
    if ind["volume_ratio"] > 1.8:
        score += 0.15; fired.append("Volume")

    # Momentum
    mom = ind["momentum_1h"]
    if (side == "LONG" and mom > 1.5) or (side == "SHORT" and mom < -1.5):
        score += 0.1; fired.append("Momentum")

    # SMC
    smc_fired = False
    bos = smc.get("bos_direction")
    if (side == "LONG" and bos == "BULL") or (side == "SHORT" and bos == "BEAR"):
        score += 0.25; smc_fired = True
    if (side == "LONG" and smc.get("in_discount")) or \
       (side == "SHORT" and smc.get("in_premium")):
        score += 0.1; smc_fired = True
    if smc_fired:
        fired.append(f"SMC {smc.get('description', '')}")

    if smc.get("choch_recently"):
        score *= 0.7

    return min(score, 1.0), fired


# ─── ML ADJUSTMENT ──────────────────────────────────────────────────────────
def apply_ml_adjustment(ind, smc, score, side, btc_ctx):
    if not os.path.exists(MODEL_FILE):
        return score
    try:
        with open(MODEL_FILE, "rb") as f:
            bundle = pickle.load(f)
        model    = bundle["model"]
        scaler   = bundle["scaler"]
        features = bundle["features"]

        feat_map = {
            "rsi":               ind.get("rsi", 50),
            "volume_ratio":      ind.get("volume_ratio", 1.0),
            "momentum_1h":       ind.get("momentum_1h", 0),
            "score":             score,
            "btc_24h":           btc_ctx.get("btc_24h", 0),
            "structure_strength": smc.get("structure_strength", 0.0),
            "in_discount":       int(bool(smc.get("in_discount", False))),
            "in_premium":        int(bool(smc.get("in_premium", False))),
            "near_supply_zone":  int(bool(smc.get("near_supply_zone", False))),
            "near_demand_zone":  int(bool(smc.get("near_demand_zone", False))),
            "choch_recently":    int(bool(smc.get("choch_recently", False))),
            "side_long":         1 if side == "LONG" else 0,
        }

        X = [[feat_map.get(k, 0) for k in features]]
        win_prob = model.predict_proba(scaler.transform(X))[0][1]
        adjusted = score * 0.6 + win_prob * 0.4
        print(f"  ML win_prob={win_prob:.2f} → score {score:.2f} → {adjusted:.2f}")
        return adjusted
    except Exception as e:
        print(f"  ML error (using raw score): {e}")
        return score


# ─── GEMINI FILTER ──────────────────────────────────────────────────────────
def gemini_filter(symbol, side, ind, smc, score, btc_ctx):
    if not GEMINI_API_KEY:
        print("  No GEMINI_API_KEY — skipping AI filter")
        return True

    smc_desc = smc.get("description", "N/A")
    prompt = (
        f"You are a crypto trading signal filter. Reply with only YES or NO.\n\n"
        f"Signal details:\n"
        f"- Coin: {symbol}\n"
        f"- Side: {side}\n"
        f"- RSI: {ind['rsi']:.1f}\n"
        f"- EMA trend: {ind['ema_trend']}\n"
        f"- Bollinger position: {ind['bb_position']}\n"
        f"- Volume ratio: {ind['volume_ratio']:.2f}x\n"
        f"- 1h momentum: {ind['momentum_1h']:.2f}%\n"
        f"- SMC structure: {smc_desc}\n"
        f"- Rule score: {score:.2f}/1.0\n"
        f"- BTC 24h change: {btc_ctx.get('btc_24h', 0):.2f}%\n\n"
        f"Is this a high-quality {side} trade signal worth broadcasting to traders? "
        f"Reply YES or NO only."
    )

    try:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        )
        r = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=20)
        text = (
            r.json()
            .get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
            .strip()
            .upper()
        )
        print(f"  Gemini says: {text!r}")
        return text.startswith("YES")
    except Exception as e:
        print(f"  Gemini error: {e}")
        return False


# ─── HELPERS ─────────────────────────────────────────────────────────────────
def make_signal_id(symbol, side, ts):
    return hashlib.md5(f"{symbol}-{side}-{int(ts)}".encode()).hexdigest()[:10]


def calc_tp_sl(side, entry, atr_v, smc):
    if side == "LONG":
        sl  = entry - atr_v * ATR_SL_MULT
        tp1 = entry + atr_v * ATR_TP1_MULT
        tp2 = entry + atr_v * ATR_TP2_MULT
        tp3 = entry + atr_v * ATR_TP3_MULT
        nsh = smc.get("nearest_swing_high")
        if nsh and tp1 < nsh < tp3 * 1.1:
            tp2 = nsh
    else:
        sl  = entry + atr_v * ATR_SL_MULT
        tp1 = entry - atr_v * ATR_TP1_MULT
        tp2 = entry - atr_v * ATR_TP2_MULT
        tp3 = entry - atr_v * ATR_TP3_MULT
        nsl = smc.get("nearest_swing_low")
        if nsl and tp1 > nsl > tp3 * 0.9:
            tp2 = nsl
    return sl, tp1, tp2, tp3


def recently_signalled(symbol, signals, hours=SIGNAL_COOLDOWN_H):
    cutoff = time.time() - hours * 3600
    return any(s.get("symbol") == symbol and s.get("ts", 0) > cutoff for s in reversed(signals))


def fmt_price(p):
    if p < 0.0001:  return f"{p:.8f}"
    if p < 0.01:    return f"{p:.6f}"
    if p < 1:       return f"{p:.4f}"
    if p < 100:     return f"{p:.3f}"
    return f"{p:.2f}"


def format_signal_message(sig):
    side   = sig["side"]
    emoji  = "🟢" if side == "LONG" else "🔴"
    arrow  = "📈" if side == "LONG" else "📉"
    smc_d  = sig.get("indicators", {}).get("smc", {}).get("description", "N/A")
    rules  = ", ".join(sig.get("fired_rules", []))
    return (
        f"{emoji} <b>{side} Signal — {sig['symbol']}</b> {arrow}\n\n"
        f"📍 Entry:  <b>{fmt_price(sig['entry'])}</b>\n"
        f"🛑 SL:     <code>{fmt_price(sig['sl'])}</code>\n"
        f"🎯 TP1:   <code>{fmt_price(sig['tp1'])}</code>\n"
        f"🎯 TP2:   <code>{fmt_price(sig['tp2'])}</code>\n"
        f"🎯 TP3:   <code>{fmt_price(sig['tp3'])}</code>\n\n"
        f"📊 Score:  {sig['score']:.0%}\n"
        f"🧠 SMC:   {smc_d}\n"
        f"✅ Rules:  {rules}\n\n"
        f"⚠️ <i>Not financial advice. Manage your risk.</i>"
    )


# ─── OUTCOME TRACKING ────────────────────────────────────────────────────────
def update_open_signals(signals):
    open_sigs = [s for s in signals if s.get("status") == "OPEN"]
    if not open_sigs:
        return
    print(f"Checking {len(open_sigs)} open signals...")
    for sig in open_sigs:
        try:
            r = requests.get(
                f"{MEXC_BASE}/api/v3/ticker/price",
                params={"symbol": sig["symbol"]},
                timeout=10,
            )
            price = float(r.json().get("price", 0))
        except Exception:
            continue

        side = sig["side"]
        sl, tp1, tp2, tp3 = sig["sl"], sig["tp1"], sig["tp2"], sig["tp3"]

        if side == "LONG":
            if price <= sl:    sig["status"] = "CLOSED_LOSS"
            elif price >= tp3: sig["status"] = "CLOSED_WIN_TP3"
            elif price >= tp2: sig["status"] = "CLOSED_WIN_TP2"
            elif price >= tp1: sig["status"] = "CLOSED_WIN_TP1"
        else:
            if price >= sl:    sig["status"] = "CLOSED_LOSS"
            elif price <= tp3: sig["status"] = "CLOSED_WIN_TP3"
            elif price <= tp2: sig["status"] = "CLOSED_WIN_TP2"
            elif price <= tp1: sig["status"] = "CLOSED_WIN_TP1"

        if sig["status"] != "OPEN":
            sig["closed_at"]    = datetime.now(timezone.utc).isoformat()
            sig["closed_price"] = price
            print(f"  {sig['symbol']} {side} → {sig['status']} @ {price}")


# ─── RULE STATS ──────────────────────────────────────────────────────────────
def update_rule_stats(signals):
    stats = load_json(RULE_STATS_FILE, {})
    for sig in signals:
        status = sig.get("status", "")
        if not (status.startswith("CLOSED_WIN") or status == "CLOSED_LOSS"):
            continue
        won = status.startswith("CLOSED_WIN")
        for rule in sig.get("fired_rules", []):
            if rule not in stats:
                stats[rule] = {"wins": 0, "losses": 0, "samples": 0}
            stats[rule]["wins"]    += int(won)
            stats[rule]["losses"]  += int(not won)
            stats[rule]["samples"] += 1
    save_json(RULE_STATS_FILE, stats)


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*50}")
    print(f"TA-Bot scan @ {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*50}")

    signals = load_json(STATE_FILE, [])
    update_open_signals(signals)

    btc_ctx = get_btc_context()
    print(f"BTC 24h: {btc_ctx['btc_24h']:+.2f}%")

    coins = get_top_volatile_coins()
    print(f"Scanning {len(coins)} volatile coins...")

    broadcasted = 0

    for symbol, pct_change in coins:
        if broadcasted >= MAX_SIGNALS_PER_RUN:
            break
        if recently_signalled(symbol, signals):
            print(f"  {symbol} — cooldown, skipping")
            continue

        print(f"\n→ {symbol} ({pct_change:+.1f}%)")

        klines = get_klines(symbol)
        if len(klines) < 60:
            print(f"  Not enough klines ({len(klines)}), skip")
            continue

        ind = compute_indicators(klines)
        if not ind:
            continue

        smc  = get_smc_features(klines)
        side, confidence = determine_side(ind, smc)
        print(f"  Side: {side} (confidence {confidence:.0%})")

        score, fired_rules = score_setup(ind, smc, side)
        print(f"  Rules fired: {fired_rules} → score {score:.2f}")

        if not fired_rules:
            print("  No rules fired, skip")
            continue

        score = apply_ml_adjustment(ind, smc, score, side, btc_ctx)

        if score < SCORE_THRESHOLD:
            print(f"  Score {score:.2f} < threshold {SCORE_THRESHOLD}, skip")
            continue

        print(f"  Score {score:.2f} ✅ — asking Gemini...")
        if not gemini_filter(symbol, side, ind, smc, score, btc_ctx):
            print("  Gemini rejected")
            continue

        entry = ind["last_close"]
        atr_v = ind["atr"]
        sl, tp1, tp2, tp3 = calc_tp_sl(side, entry, atr_v, smc)
        now_ts = time.time()

        signal = {
            "id":          make_signal_id(symbol, side, now_ts),
            "symbol":      symbol,
            "side":        side,
            "entry":       entry,
            "sl":          sl,
            "tp1":         tp1,
            "tp2":         tp2,
            "tp3":         tp3,
            "score":       round(score, 4),
            "status":      "OPEN",
            "ts":          now_ts,
            "created_at":  datetime.now(timezone.utc).isoformat(),
            "fired_rules": fired_rules,
            "btc_context": btc_ctx,
            "indicators": {
                "rsi":          ind["rsi"],
                "ema_trend":    ind["ema_trend"],
                "bb_position":  ind["bb_position"],
                "volume_ratio": round(ind["volume_ratio"], 3),
                "momentum_1h":  round(ind["momentum_1h"], 3),
                "atr":          round(atr_v, 8),
                "smc":          smc,
            },
        }

        broadcast_to_subscribers(format_signal_message(signal))
        print(f"  ✅ Broadcasted {side} signal for {symbol}")

        signals.append(signal)
        broadcasted += 1
        time.sleep(2)

    save_json(STATE_FILE, signals)
    update_rule_stats(signals)

    open_count  = sum(1 for s in signals if s.get("status") == "OPEN")
    closed_wins = sum(1 for s in signals if s.get("status", "").startswith("CLOSED_WIN"))
    closed_loss = sum(1 for s in signals if s.get("status") == "CLOSED_LOSS")
    total_closed = closed_wins + closed_loss
    wr = (closed_wins / total_closed * 100) if total_closed else 0

    print(f"\n{'='*50}")
    print(f"Done. Broadcasted {broadcasted} signal(s). Total in DB: {len(signals)}")

    tg_send(
        f"🤖 <b>Hourly scan complete</b>\n\n"
        f"New signals: {broadcasted}\n"
        f"Open:        {open_count}\n"
        f"Closed wins: {closed_wins}\n"
        f"Closed loss: {closed_loss}\n"
        f"Win rate:    {wr:.1f}% ({total_closed} closed)\n"
        f"BTC 24h:     {btc_ctx['btc_24h']:+.2f}%",
        to_admin=True,
    )


if __name__ == "__main__":
    main()
