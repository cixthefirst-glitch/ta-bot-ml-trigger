"""
Train a logistic regression model on past signals (from data/signals.json)
to predict the probability that a new signal ends in a WIN.
Saves the model to data/model.pkl and broadcasts accuracy/coefs to Telegram.
Also saves the model coefficients and writes them into the scoring for bot.py.
"""
import os
import json
import pickle
import requests
from datetime import datetime, timezone

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "")
ADMIN_CHAT = os.environ.get("ADMIN_CHAT_ID", "")
STATE_FILE = "data/signals.json"
MODEL_FILE = "data/model.pkl"
COEFS_FILE = "data/model_coefs.json"

# Features used by logistic regression. Order matters: it must match
# the array built in featurize() and the order in model_coefs.json
# (consumed by apply_ml_adjustment in bot.py).
FEATURE_KEYS = ["rsi", "volume_ratio", "momentum_1h", "score", "btc_24h"]

# SMC features come from indicators.smc (ml/smc.py), populated at scan
# time. Older signals (pre-SMC) will have smc=None and we fall back to
# neutral defaults — backward compatible.
SMC_FEATURE_KEYS = ["structure_strength", "in_discount", "in_premium",
                    "near_supply_zone", "near_demand_zone", "choch_recently"]

def load_signals():
    if not os.path.exists(STATE_FILE): return []
    try:
        with open(STATE_FILE) as f: return json.load(f)
    except Exception: return []

def featurize(sig):
    ind = sig.get("indicators", {}) or {}
    btc = sig.get("btc_context", {}) or {}
    smc = ind.get("smc", {}) or {}  # {} for pre-SMC signals (backward compat)
    return {
        "rsi": ind.get("rsi", 50) or 50,
        "volume_ratio": ind.get("volume_ratio", 1.0) or 1.0,
        "momentum_1h": ind.get("momentum_1h", 0) or 0,
        "score": sig.get("score", 0) or 0,
        "btc_24h": btc.get("btc_24h", 0) or 0,
        # SMC features — neutral defaults for backward compat
        "structure_strength": smc.get("structure_strength", 0.0) or 0.0,
        "in_discount": int(bool(smc.get("in_discount", False))),
        "in_premium": int(bool(smc.get("in_premium", False))),
        "near_supply_zone": int(bool(smc.get("near_supply_zone", False))),
        "near_demand_zone": int(bool(smc.get("near_demand_zone", False))),
        "choch_recently": int(bool(smc.get("choch_recently", False))),
        "side_long": 1 if sig.get("side") == "LONG" else 0,
    }

def outcome_label(sig):
    s = sig.get("status", "")
    if s.startswith("CLOSED_WIN"): return 1
    if s == "CLOSED_LOSS": return 0
    return None  # still open or unknown

def tg_send(text, to_admin=False):
    if not TG_TOKEN: return
    target = ADMIN_CHAT if to_admin else TG_CHAT
    if not target: return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": target, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}, timeout=15)
    except Exception as e:
        print(f"TG error: {e}")

def main():
    sigs = load_signals()
    closed = [s for s in sigs if outcome_label(s) is not None]
    print(f"Total signals: {len(sigs)}, closed: {len(closed)}")
    if len(closed) < 10:
        msg = f"🤖 <b>ML train</b>: need >=10 closed signals, have {len(closed)}. Skipping."
        print(msg); tg_send(msg, to_admin=True)
        return

    # Build combined feature list (base + SMC)
    all_features = FEATURE_KEYS + SMC_FEATURE_KEYS + ["side_long"]
    n_smc = len(SMC_FEATURE_KEYS)

    X, y, idxs, sig_with_smc = [], [], [], []
    for s in closed:
        f = featurize(s)
        X.append([f[k] for k in all_features])
        y.append(outcome_label(s))
        idxs.append(s.get("id", ""))
        sig_with_smc.append(bool(s.get("indicators", {}).get("smc")))
    X = np.array(X, dtype=float); y = np.array(y, dtype=int)
    n_with_smc = sum(sig_with_smc)
    print(f"Training on {len(X)} samples ({n_with_smc} with SMC features), win rate {y.mean():.2%}")

    if len(set(y)) < 2:
        msg = f"🤖 <b>ML train</b>: only one class in y (all wins or all losses). Need both."
        print(msg); tg_send(msg, to_admin=True)
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train); Xte = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(Xtr, y_train)
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = accuracy_score(y_test, pred)
    ll = log_loss(y_test, proba)

    coefs = {k: float(c) for k, c in zip(all_features, model.coef_[0])}
    intercept = float(model.intercept_[0])

    # Print SMC coefs separately so we can see their effect
    smc_coefs = {k: coefs[k] for k in SMC_FEATURE_KEYS}
    smc_summary = "\n".join(f"  {k:20s} {v:+.3f}" for k, v in sorted(smc_coefs.items(), key=lambda x: -abs(x[1])))

    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "features": all_features}, f)
    with open(COEFS_FILE, "w") as f:
        json.dump({"ts": datetime.now(timezone.utc).isoformat(), "coefs": coefs, "intercept": intercept, "accuracy": acc, "n_train": len(X_train), "n_test": len(X_test), "win_rate": float(y.mean()), "n_with_smc": n_with_smc, "smc_features": SMC_FEATURE_KEYS}, f, indent=2)

    coef_lines = "\n".join(f"  {k:20s} {v:+.3f}" for k, v in sorted(coefs.items(), key=lambda x: -abs(x[1])))
    msg = f"""🤖 <b>ML trained</b>

Samples: {len(X)} (train {len(X_train)}, test {len(X_test)}, with SMC: {n_with_smc})
Test accuracy: <b>{acc:.1%}</b>
Log-loss: {ll:.3f}
Win rate: {y.mean():.1%}

Top coefficients:
{coef_lines}

SMC features:
{smc_summary}"""
    print(msg); tg_send(msg, to_admin=True)
    print(f"Saved {MODEL_FILE} and {COEFS_FILE}")

if __name__ == "__main__":
    main()
