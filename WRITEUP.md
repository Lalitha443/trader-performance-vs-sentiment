# Write-Up: Trader Performance vs Market Sentiment

## Methodology

Two datasets were merged on calendar date: the Bitcoin Fear/Greed Index (daily sentiment label) and 211,224 Hyperliquid trade fills across 32 accounts. Both sources were clean on inspection (zero missing values, zero duplicates), and only 6 trades (~0.003%) fell outside the sentiment index's date range and were dropped.

Trades were aggregated to an **account-day** level (2,340 account-day observations), computing: daily realized PnL, trade count, win rate, average trade size (USD — used as the leverage/risk-appetite proxy, since the raw data has no leverage field), long/short trade ratio, and total volume. The 5-way sentiment label was collapsed to a binary Fear/Greed bucket for the core comparisons (Neutral days excluded from that comparison, retained for a full-spectrum view). Three account-level segments were built via median splits: trade frequency (active days), position size, and win-rate consistency.

Statistical comparison used a Mann-Whitney U test (appropriate given the heavy-tailed, non-normal PnL distributions). A bonus Random Forest classifier was trained to predict next-day account profitability from same-day sentiment + behavior features.

## Key Insights

**1. Performance differs by sentiment, but not in the direction "Fear = bad."** Accounts are profitable on more *days* during Greed (64% of account-days) than Fear (60%), and median PnL is over 2x higher on Greed days. But mean PnL is actually higher on Fear days ($5,185 vs $4,144/day) because Fear-day outcomes have fatter tails — a few very large wins or losses dominate. Average loss-day depth is also worse on Greed days (−$12,217 vs −$9,159), consistent with overconfidence-driven larger losses when sentiment is bullish.

**2. Traders reliably change behavior with sentiment.** On Fear days, traders execute ~37% more trades/day (105 vs 77), take ~43% larger average positions ($8,530 vs $5,955), and flip from net-short-leaning to a **63% long-side bias** (vs 44% long on Greed days) — a contrarian "buy the fear" pattern that increases both activity and directional exposure exactly when outcomes are more volatile and less consistently profitable.

**3. Trader segment matters more than win rate.** Large-position traders earn ~86% more total PnL than small-position traders ($416,806 vs $224,099), and that edge widens further on Fear days. Counter-intuitively, accounts with a win rate ≥ 50% ("Consistent Winners") have *lower* average total PnL ($206,867) than more inconsistent traders ($332,203) — a few big asymmetric wins outperform a high hit-rate/low-payoff style.

**Bonus finding:** A Random Forest predicts next-day account profitability at ROC-AUC ≈ 0.71 from same-day features. Sentiment score is the *least* important of seven features — a trader's own current PnL, win rate, and trade count are far more predictive of their own next-day result than the market's mood.

## Strategy Recommendations

**Rule 1 — Cap position size on Fear days, especially for large-size traders.** This segment already captures the most edge, but its downside (deep loss-days, 63% long crowding) concentrates during Fear regimes. Capping size to ~70% of their own Greed-day average during Fear/Extreme Fear preserves upside while limiting fat-tail losses.

**Rule 2 — Don't chase trade frequency; favor selective, asymmetric setups, and reserve frequency increases for Greed days.** Since win-rate consistency doesn't predict total PnL and frequent traders barely beat infrequent ones, scaling up trade count isn't a reliable lever. Smaller/infrequent traders should prioritize fewer, higher-conviction setups over volume — and only ramp up activity when sentiment reads Greed, where profitable-day rate and win rate are both higher.
