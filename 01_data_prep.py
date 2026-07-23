"""
Part A - Data Preparation
Trader Performance vs Market Sentiment
"""
import pandas as pd
import numpy as np

pd.set_option('display.width', 120)

# ---------------------------------------------------------
# 1. Load datasets
# ---------------------------------------------------------
trades = pd.read_csv('/mnt/user-data/uploads/historical_data.csv')
sentiment = pd.read_csv('/mnt/user-data/uploads/fear_greed_index.csv')

print("="*70)
print("RAW SHAPES")
print("="*70)
print(f"Trades   : {trades.shape[0]:,} rows x {trades.shape[1]} cols")
print(f"Sentiment: {sentiment.shape[0]:,} rows x {sentiment.shape[1]} cols")

print("\nTRADES - missing values:")
print(trades.isna().sum()[trades.isna().sum() > 0] if trades.isna().sum().sum() else "  none")
print(f"TRADES - duplicate rows: {trades.duplicated().sum()}")

print("\nSENTIMENT - missing values:")
print(sentiment.isna().sum()[sentiment.isna().sum() > 0] if sentiment.isna().sum().sum() else "  none")
print(f"SENTIMENT - duplicate rows: {sentiment.duplicated().sum()}")

# ---------------------------------------------------------
# 2. Convert timestamps / align by date
# ---------------------------------------------------------
trades['date'] = pd.to_datetime(trades['Timestamp IST'], format='%d-%m-%Y %H:%M').dt.date
sentiment['date'] = pd.to_datetime(sentiment['date']).dt.date

# collapse 5-way classification to binary Fear/Greed (drop Neutral for the core
# Fear-vs-Greed question, keep it available for full-spectrum analysis)
def bucket(c):
    if 'Fear' in c:
        return 'Fear'
    if 'Greed' in c:
        return 'Greed'
    return 'Neutral'

sentiment['sentiment_bucket'] = sentiment['classification'].apply(bucket)

trades = trades.merge(
    sentiment[['date', 'classification', 'sentiment_bucket']],
    on='date', how='left'
)

missing_sent = trades['sentiment_bucket'].isna().sum()
print(f"\nTrades with no matching sentiment date: {missing_sent} ({missing_sent/len(trades):.2%})")
trades = trades.dropna(subset=['sentiment_bucket'])

# ---------------------------------------------------------
# 3. Trade-level derived fields
# ---------------------------------------------------------
trades['is_win'] = trades['Closed PnL'] > 0
trades['is_realized'] = trades['Closed PnL'] != 0     # only closes realize PnL
trades['leverage_proxy'] = trades['Size USD'] / trades['Size USD'].where(trades['Size USD']>0).median()  # placeholder, real leverage not in data
trades['long_side'] = trades['Direction'].isin(['Open Long', 'Close Long', 'Buy', 'Long > Short'])

# ---------------------------------------------------------
# 4. Daily per-account aggregation  (the core analysis table)
# ---------------------------------------------------------
daily = trades.groupby(['Account', 'date', 'sentiment_bucket', 'classification']).agg(
    daily_pnl=('Closed PnL', 'sum'),
    n_trades=('Closed PnL', 'count'),
    n_wins=('is_win', 'sum'),
    avg_trade_size_usd=('Size USD', 'mean'),
    total_volume_usd=('Size USD', 'sum'),
    n_long=('long_side', 'sum'),
).reset_index()

daily['win_rate'] = daily['n_wins'] / daily['n_trades']
daily['n_short'] = daily['n_trades'] - daily['n_long']
daily['long_short_ratio'] = daily['n_long'] / daily['n_short'].replace(0, np.nan)
daily['is_profitable_day'] = daily['daily_pnl'] > 0

print("\n" + "="*70)
print("DAILY PER-ACCOUNT TABLE (head)")
print("="*70)
print(daily.head())
print(f"\nDaily table shape: {daily.shape}")

daily.to_csv('/home/claude/project/data/daily_account_sentiment.csv', index=False)
trades.to_pickle('/home/claude/project/data/trades_clean.pkl')

print("\nSaved: data/daily_account_sentiment.csv, data/trades_clean.pkl")
