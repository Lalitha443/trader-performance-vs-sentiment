"""
Part B - Analysis
Fear vs Greed: performance, behavior, segments
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['figure.dpi'] = 110
plt.rcParams['font.size'] = 10

daily = pd.read_csv('/home/claude/project/data/daily_account_sentiment.csv')
trades = pd.read_pickle('/home/claude/project/data/trades_clean.pkl')

fg = daily[daily['sentiment_bucket'].isin(['Fear', 'Greed'])].copy()

# =========================================================
# Q1: Does performance differ between Fear vs Greed days?
# =========================================================
print("="*70)
print("Q1: PERFORMANCE — FEAR vs GREED")
print("="*70)

perf = fg.groupby('sentiment_bucket').agg(
    mean_daily_pnl=('daily_pnl', 'mean'),
    median_daily_pnl=('daily_pnl', 'median'),
    win_rate=('win_rate', 'mean'),
    pct_profitable_days=('is_profitable_day', 'mean'),
    n_obs=('daily_pnl', 'count'),
)
print(perf)

fear_pnl = fg.loc[fg.sentiment_bucket == 'Fear', 'daily_pnl']
greed_pnl = fg.loc[fg.sentiment_bucket == 'Greed', 'daily_pnl']
t_stat, p_val = stats.mannwhitneyu(fear_pnl, greed_pnl, alternative='two-sided')
print(f"\nMann-Whitney U test (daily PnL, Fear vs Greed): U={t_stat:.0f}, p={p_val:.4f}")

# drawdown proxy: worst single-day loss per account, by sentiment
drawdown = fg.groupby('sentiment_bucket')['daily_pnl'].apply(lambda s: s[s < 0].mean())
print(f"\nAvg loss-day magnitude (drawdown proxy):\n{drawdown}")

# chart 1
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
perf['mean_daily_pnl'].plot(kind='bar', ax=axes[0], color=['#d62728', '#2ca02c'])
axes[0].set_title('Avg Daily PnL per Account-Day')
axes[0].set_ylabel('USD')
axes[0].axhline(0, color='k', lw=0.6)
axes[0].tick_params(axis='x', rotation=0)

perf['pct_profitable_days'].plot(kind='bar', ax=axes[1], color=['#d62728', '#2ca02c'])
axes[1].set_title('% of Profitable Account-Days')
axes[1].set_ylabel('Proportion')
axes[1].tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.savefig('/home/claude/project/charts/01_performance_fear_vs_greed.png')
plt.close()

# =========================================================
# Q2: Behavior shifts (frequency, size, long/short bias)
# =========================================================
print("\n" + "="*70)
print("Q2: BEHAVIOR — FEAR vs GREED")
print("="*70)

behavior = fg.groupby('sentiment_bucket').agg(
    avg_trades_per_day=('n_trades', 'mean'),
    avg_trade_size_usd=('avg_trade_size_usd', 'mean'),
    avg_long_short_ratio=('long_short_ratio', lambda s: s.replace([np.inf, -np.inf], np.nan).mean()),
    total_volume_usd=('total_volume_usd', 'sum'),
)
print(behavior)

long_pct = fg.groupby('sentiment_bucket').apply(
    lambda d: d['n_long'].sum() / (d['n_long'].sum() + d['n_short'].sum())
)
print(f"\n% of trades that are LONG, by sentiment:\n{long_pct}")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
behavior['avg_trades_per_day'].plot(kind='bar', ax=axes[0], color=['#d62728', '#2ca02c'])
axes[0].set_title('Avg Trades / Account / Day')
axes[0].tick_params(axis='x', rotation=0)

behavior['avg_trade_size_usd'].plot(kind='bar', ax=axes[1], color=['#d62728', '#2ca02c'])
axes[1].set_title('Avg Trade Size (USD)')
axes[1].tick_params(axis='x', rotation=0)

long_pct.plot(kind='bar', ax=axes[2], color=['#d62728', '#2ca02c'])
axes[2].set_title('% of Trades that are LONG')
axes[2].axhline(0.5, color='k', lw=0.6, ls='--')
axes[2].tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.savefig('/home/claude/project/charts/02_behavior_fear_vs_greed.png')
plt.close()

# =========================================================
# Q3: Segments
# =========================================================
print("\n" + "="*70)
print("Q3: TRADER SEGMENTS")
print("="*70)

acct_stats = trades.groupby('Account').agg(
    total_trades=('Closed PnL', 'count'),
    total_pnl=('Closed PnL', 'sum'),
    avg_trade_size=('Size USD', 'mean'),
    win_rate=('is_win', 'mean'),
    active_days=('date', 'nunique'),
).reset_index()

# Segment A: frequent vs infrequent traders (median split on active_days)
med_days = acct_stats['active_days'].median()
acct_stats['freq_segment'] = np.where(acct_stats['active_days'] >= med_days, 'Frequent', 'Infrequent')

# Segment B: large vs small position-size traders (proxy for leverage/risk appetite —
# per-account leverage figures are not present in the raw data, so avg trade size is
# used as the closest available proxy for risk-taking)
med_size = acct_stats['avg_trade_size'].median()
acct_stats['size_segment'] = np.where(acct_stats['avg_trade_size'] >= med_size, 'Large-size', 'Small-size')

# Segment C: consistent winners vs inconsistent (win rate >= 50%)
acct_stats['consistency_segment'] = np.where(acct_stats['win_rate'] >= 0.5, 'Consistent Winner', 'Inconsistent')

print(acct_stats[['freq_segment', 'total_pnl']].groupby('freq_segment').mean())
print()
print(acct_stats[['size_segment', 'total_pnl']].groupby('size_segment').mean())
print()
print(acct_stats[['consistency_segment', 'total_pnl']].groupby('consistency_segment').mean())

acct_stats.to_csv('/home/claude/project/data/account_segments.csv', index=False)

# how do segments behave differently in Fear vs Greed?
daily_seg = daily.merge(acct_stats[['Account', 'freq_segment', 'size_segment', 'consistency_segment']],
                         on='Account', how='left')
daily_seg_fg = daily_seg[daily_seg['sentiment_bucket'].isin(['Fear', 'Greed'])]

seg_pnl = daily_seg_fg.groupby(['size_segment', 'sentiment_bucket'])['daily_pnl'].mean().unstack()
print(f"\nAvg daily PnL by size segment x sentiment:\n{seg_pnl}")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
seg_pnl.plot(kind='bar', ax=ax, color=['#d62728', '#2ca02c'])
ax.set_title('Avg Daily PnL: Trade-Size Segment x Sentiment')
ax.set_ylabel('USD')
ax.axhline(0, color='k', lw=0.6)
ax.tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.savefig('/home/claude/project/charts/03_segment_pnl_by_sentiment.png')
plt.close()

# =========================================================
# Full sentiment spectrum (bonus context, incl. Neutral / Extreme)
# =========================================================
order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
full = daily.groupby('classification')['daily_pnl'].mean().reindex(order)
fig, ax = plt.subplots(figsize=(7, 4.2))
colors = ['#8b0000', '#d62728', '#999999', '#2ca02c', '#006400']
full.plot(kind='bar', ax=ax, color=colors)
ax.set_title('Avg Daily PnL Across Full Sentiment Spectrum')
ax.set_ylabel('USD')
ax.axhline(0, color='k', lw=0.6)
ax.tick_params(axis='x', rotation=20)
plt.tight_layout()
plt.savefig('/home/claude/project/charts/04_full_spectrum_pnl.png')
plt.close()

print("\nSaved 4 charts to /home/claude/project/charts/")
print("Saved account_segments.csv")
