"""
Smart Money Concepts (SMC) structure analyzer.

Detects market structure features from OHLC klines:
  - Swing highs/lows
  - Break of Structure (BOS) - continuation
  - Change of Character (ChoCH) - reversal
  - Premium / Discount zones
  - Supply / Demand zones (simple version)

Input:  list of [openTime_ms, open, high, low, close, volume] (MEXC format)
Output: dict of features, plus a structure_strength score adjustment
        to be added to the existing rule-based score.

All features are pure math from klines. No external API. No state.
Falls back gracefully (returns empty dict) if not enough data.
"""
from typing import List, Dict, Optional, Tuple

# Minimum candles needed for reliable structure detection
MIN_KLINES = 60  # ~2.5 days of 1h klines
SWING_LOOKBACK = 3
RANGE_LOOKBACK = 50
SWING_MIN_PROMINENCE = 0.001  # 0.1%


def _find_swing_highs(highs, lookback=SWING_LOOKBACK, min_prominence=SWING_MIN_PROMINENCE):
    """Return indices of swing highs (local peaks)."""
    swings = []
    n = len(highs)
    for i in range(lookback, n - lookback):
        is_swing = True
        for j in range(1, lookback + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_swing = False
                break
        if is_swing:
            left_min = min(highs[max(0, i - lookback * 2):i])
            right_min = min(highs[i + 1:i + lookback * 2 + 1])
            base = min(left_min, right_min)
            if base > 0 and (highs[i] - base) / base >= min_prominence:
                swings.append(i)
    return swings


def _find_swing_lows(lows, lookback=SWING_LOOKBACK, min_prominence=SWING_MIN_PROMINENCE):
    """Return indices of swing lows (local troughs)."""
    swings = []
    n = len(lows)
    for i in range(lookback, n - lookback):
        is_swing = True
        for j in range(1, lookback + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_swing = False
                break
        if is_swing:
            left_max = max(lows[max(0, i - lookback * 2):i])
            right_max = max(lows[i + 1:i + lookback * 2 + 1])
            cap = max(left_max, right_max)
            if cap > 0 and (cap - lows[i]) / cap >= min_prominence:
                swings.append(i)
    return swings


def get_smc_features(klines):
    """
    Compute SMC features from MEXC-format klines.
    Returns dict with keys:
      bos_direction: BULL | BEAR | None
      choch_recently: bool
      in_discount: bool
      in_premium: bool
      in_deep_discount: bool
      in_deep_premium: bool
      near_supply_zone: bool
      near_demand_zone: bool
      structure_strength: float in [-1.0, +1.0]
      description: str (human-readable for telegram)
    Returns empty dict if insufficient data.
    """
    if not klines or len(klines) < MIN_KLINES:
        return {}

    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]
    closes = [float(k[4]) for k in klines]
    last_close = closes[-1]

    # 1) Premium / Discount (current price position in recent range)
    range_slice = slice(max(0, len(closes) - RANGE_LOOKBACK), len(closes))
    range_high = max(highs[range_slice])
    range_low = min(lows[range_slice])
    if range_high <= range_low:
        return {}
    range_size = range_high - range_low
    price_position = (last_close - range_low) / range_size
    in_discount = price_position < 0.5
    in_premium = price_position > 0.5
    in_deep_discount = price_position < 0.3
    in_deep_premium = price_position > 0.7

    # 2) Swing detection
    swing_highs = _find_swing_highs(highs)
    swing_lows = _find_swing_lows(lows)

    # 3) BOS direction
    bos_direction = None
    choch_recently = False
    if swing_highs and swing_lows:
        last_sh = highs[swing_highs[-1]]
        last_sl = lows[swing_lows[-1]]
        if last_close > last_sh * 1.0005:
            bos_direction = "BULL"
        elif last_close < last_sl * 0.9995:
            bos_direction = "BEAR"
        else:
            if last_close > last_sh * 0.995 and last_close < last_sh:
                bos_direction = "TESTING_HIGH"
            elif last_close < last_sl * 1.005 and last_close > last_sl:
                bos_direction = "TESTING_LOW"

        # 4) ChoCH detection
        breaks = []
        for idx in swing_highs:
            if idx >= len(closes) - 30 and idx < len(closes) - 1:
                future_closes = closes[idx + 1:]
                if future_closes and all(c > highs[idx] for c in future_closes):
                    breaks.append((idx, "BULL"))
        for idx in swing_lows:
            if idx >= len(closes) - 30 and idx < len(closes) - 1:
                future_closes = closes[idx + 1:]
                if future_closes and all(c < lows[idx] for c in future_closes):
                    breaks.append((idx, "BEAR"))
        breaks.sort()
        if len(breaks) >= 2 and breaks[-1][1] != breaks[-2][1]:
            choch_recently = True

    # 5) Near supply/demand zones (within 1% of recent swing high/low)
    near_supply_zone = False
    near_demand_zone = False
    if swing_highs:
        recent_sh = highs[swing_highs[-1]]
        if last_close > 0 and abs(last_close - recent_sh) / last_close < 0.01:
            near_supply_zone = True
    if swing_lows:
        recent_sl = lows[swing_lows[-1]]
        if last_close > 0 and abs(last_close - recent_sl) / last_close < 0.01:
            near_demand_zone = True

    # 6) Structure strength (the score adjustment)
    structure_strength = 0.0
    if bos_direction == "BULL":
        structure_strength += 0.6
    elif bos_direction == "BEAR":
        structure_strength -= 0.6
    elif bos_direction == "TESTING_HIGH":
        structure_strength += 0.1
    elif bos_direction == "TESTING_LOW":
        structure_strength -= 0.1

    if choch_recently:
        structure_strength *= 0.5

    if in_deep_discount:
        structure_strength += 0.3
    elif in_deep_premium:
        structure_strength -= 0.3
    elif in_discount:
        structure_strength += 0.15
    elif in_premium:
        structure_strength -= 0.15

    if near_demand_zone:
        structure_strength += 0.2
    if near_supply_zone:
        structure_strength -= 0.2

    structure_strength = max(-1.0, min(1.0, structure_strength))

    # 7) Description
    desc_parts = []
    if bos_direction == "BULL": desc_parts.append("Bullish BOS")
    elif bos_direction == "BEAR": desc_parts.append("Bearish BOS")
    elif bos_direction == "TESTING_HIGH": desc_parts.append("Testing swing high")
    elif bos_direction == "TESTING_LOW": desc_parts.append("Testing swing low")

    if choch_recently: desc_parts.append("ChoCH")
    if in_deep_discount: desc_parts.append("deep discount")
    elif in_discount: desc_parts.append("discount")
    elif in_deep_premium: desc_parts.append("deep premium")
    elif in_premium: desc_parts.append("premium")
    if near_demand_zone: desc_parts.append("at demand")
    if near_supply_zone: desc_parts.append("at supply")

    description = " | ".join(desc_parts) if desc_parts else "no structure"

    return {
        "bos_direction": bos_direction,
        "choch_recently": choch_recently,
        "in_discount": in_discount,
        "in_premium": in_premium,
        "in_deep_discount": in_deep_discount,
        "in_deep_premium": in_deep_premium,
        "near_supply_zone": near_supply_zone,
        "near_demand_zone": near_demand_zone,
        "structure_strength": structure_strength,
        "description": description,
        "swing_highs_count": len(swing_highs),
        "swing_lows_count": len(swing_lows),
    }
