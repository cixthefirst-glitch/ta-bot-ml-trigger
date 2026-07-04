# TA-BOT ML Pipeline (public mirror)

Runs every hour on GitHub Actions free tier (public repo = reliable runner access). Scans MEXC, generates rule-based signals (RSI/EMA/BB/Volume), filters with Gemini one-word YES/NO, broadcasts to Telegram, saves signals, tracks TP/SL outcomes, trains XGBoost after 30+ closed signals.

## Cost
$0 — GitHub Actions public-repo free tier (4× the private quota).

## Setup
Set these as repo secrets (Settings > Secrets and variables > Actions):
- `GEMINI_API_KEY`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `ADMIN_CHAT_ID`
- `MEXC_SECRET_KEY` (optional)
- `MEXC_ACCESS_KEY` (optional)
- `COINGECKO_API_KEY` (optional)

## Files
- `bot.py` — main scanner + broadcaster (hourly)
- `ml/train.py` — weekly XGBoost trainer
- `backtest.py` — strategy backtest + ML noise audit
- `data/signals.json` — signal database (auto-committed)
- `data/model.pkl` — trained model (auto-committed when trained)
- `data/threshold_weights.json` — bandit arm weights (auto-committed)
- `data/subscribers.json`, `data/trial_users.json`, `data/pending_payments.json` — payment state (auto-committed)
