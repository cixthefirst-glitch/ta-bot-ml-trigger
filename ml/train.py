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

FEATURE_KEYS = ["rsi", "volume_ratio", "momentum_1h", "score", "btc_24h"]

def load_signals():
    if not os.path.exists(STATE_FILE): return []
    try:
        with open(STATE_FILE) as f: return json.load(f)
    except Exception: return []

def featurize(sig):
    ind = sig.get("indicators", {}) or {}
    btc = sig.get("btc_context", {}) or {}
    return {
        "rsi": ind.get("rsi", 50) or 50,
        "volume_ratio": ind.get("volume_ratio", 1.0) or 1.0,
        "momentum_1h": ind.get("momentum_1h", 0) or 0,
        "score": sig.get("score", 0) or 0,
        "btc_24h": btc.get("btc_24h", 0) or 0,
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
    if len(closed) < 20:
        msg = f"🤖 <b>ML train</b>: need >=20 closed signals, have {len(closed)}. Skipping."
        print(msg); tg_send(msg, to_admin=True)
        return

    X, y, idxs = [], [], []
    for s in closed:
        f = featurize(s)
        X.append([f[k] for k in FEATURE_KEYS] + [f["side_long"]])
        y.append(outcome_label(s))
        idxs.append(s.get("id", ""))
    X = np.array(X, dtype=float); y = np.array(y, dtype=int)
    print(f"Training on {len(X)} samples, win rate {y.mean():.2%}")

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

    coefs = {k: float(c) for k, c in zip(FEATURE_KEYS + ["side_long"], model.coef_[0])}
    intercept = float(model.intercept_[0])

    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "features": FEATURE_KEYS + ["side_long"]}, f)
    with open(COEFS_FILE, "w") as f:
        json.dump({"ts": datetime.now(timezone.utc).isoformat(), "coefs": coefs, "intercept": intercept, "accuracy": acc, "n_train": len(X_train), "n_test": len(X_test), "win_rate": float(y.mean())}, f, indent=2)

    coef_lines = "\n".join(f"  {k:15s} {v:+.3f}" for k, v in sorted(coefs.items(), key=lambda x: -abs(x[1])))
    msg = f"""🤖 <b>ML trained</b>

Samples: {len(X)} (train {len(X_train)}, test {len(X_test)})
Test accuracy: <b>{acc:.1%}</b>
Log-loss: {ll:.3f}
Win rate: {y.mean():.1%}

Top coefficients:
{coef_lines}"""
    print(msg); tg_send(msg, to_admin=True)
    print(f"Saved {MODEL_FILE} and {COEFS_FILE}")

if __name__ == "__main__":
    main()
