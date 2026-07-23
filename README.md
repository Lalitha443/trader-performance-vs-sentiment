# Trader Performance vs Market Sentiment
**Primetrade.ai — Data Science Intern Round-0 Assignment**

Analysis of how Bitcoin Fear/Greed sentiment relates to trader behavior and performance on Hyperliquid.

## Repo Structure

```
.
├── README.md
├── WRITEUP.md                          # 1-page summary: methodology, insights, strategy
├── notebooks/
│   └── trader_sentiment_analysis.ipynb # Main analysis notebook (pre-run, outputs included)
├── 01_data_prep.py                     # Part A — cleaning, merge, metrics (script version)
├── 02_analysis.py                      # Part B — Fear vs Greed analysis + segments (script version)
├── 03_bonus_model.py                   # Bonus — predictive model
├── data/
│   ├── daily_account_sentiment.csv     # Cleaned, aggregated daily account-level table
│   └── account_segments.csv            # Per-account segment labels
└── charts/
    ├── 01_performance_fear_vs_greed.png
    ├── 02_behavior_fear_vs_greed.png
    ├── 03_segment_pnl_by_sentiment.png
    ├── 04_full_spectrum_pnl.png
    └── 05_model_feature_importance.png
```

## Setup

```bash
pip install pandas numpy matplotlib scipy scikit-learn
```

Place the two source files (not included in this repo — see Data Sources below) at:
```
/mnt/user-data/uploads/historical_data.csv
/mnt/user-data/uploads/fear_greed_index.csv
```
(or edit the paths at the top of `01_data_prep.py` / the first notebook cell to point at your local copies).

## How to Run

**Option 1 — Notebook (recommended, already contains outputs/charts):**
```bash
jupyter notebook notebooks/trader_sentiment_analysis.ipynb
```

**Option 2 — Scripts, in order:**
```bash
python 01_data_prep.py      # writes data/daily_account_sentiment.csv
python 02_analysis.py       # writes charts/01-04 + data/account_segments.csv
python 03_bonus_model.py    # writes charts/05, prints model metrics
```

## Data Sources

1. **Bitcoin Fear & Greed Index** — daily sentiment classification (Extreme Fear → Extreme Greed), Feb 2018–May 2025.
2. **Hyperliquid historical trade data** — 211,224 individual trade fills across 32 accounts and 246 coins, May 2023–May 2025. Fields used: `Account`, `Timestamp IST`, `Closed PnL`, `Size USD`, `Direction`.

## Key Results (see `WRITEUP.md` for full detail)

- Traders are profitable on more days during Greed (64%) than Fear (60%), but Fear-day PnL has fatter tails and Greed-day losses run deeper on average.
- On Fear days, traders trade ~37% more often, take ~43% larger positions, and flip to a long-side bias (63% long vs 44% on Greed days).
- Large position-size traders earn ~86% more total PnL than small-size traders, and the gap widens further on Fear days.
- Win rate and profitability are not the same thing — accounts with win rate ≥ 50% actually have *lower* average total PnL than more inconsistent traders.
- A Random Forest predicting next-day account profitability from same-day behavior + sentiment achieves ROC-AUC ≈ 0.71; a trader's own behavior matters more than sentiment for predicting their own next-day result.

## Notes & Limitations

- The raw trade export has no leverage or account-equity field, so **average trade size (USD)** is used throughout as the closest available proxy for leverage/risk appetite.
- "Neutral" sentiment days are included in the full-spectrum chart but excluded from the core Fear-vs-Greed statistical comparisons, per the assignment's framing.
- Findings are correlational, drawn from 32 accounts over ~2 years — enough for directional insight, not a rigorously backtested trading strategy.
