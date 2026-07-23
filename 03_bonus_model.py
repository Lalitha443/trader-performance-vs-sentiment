"""
Bonus - Predictive model: next-day profitability bucket per account,
using sentiment + prior-day behavior features.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

daily = pd.read_csv('/home/claude/project/data/daily_account_sentiment.csv')
daily['date'] = pd.to_datetime(daily['date'])
daily = daily.sort_values(['Account', 'date'])

# sentiment as ordinal score for modeling
sent_map = {'Extreme Fear': 0, 'Fear': 1, 'Neutral': 2, 'Greed': 3, 'Extreme Greed': 4}
daily['sentiment_score'] = daily['classification'].map(sent_map)

# next day's profitability (target), per account
daily['next_day_profitable'] = daily.groupby('Account')['is_profitable_day'].shift(-1)

feat_cols = ['sentiment_score', 'daily_pnl', 'n_trades', 'win_rate',
             'avg_trade_size_usd', 'long_short_ratio', 'total_volume_usd']

model_df = daily.dropna(subset=feat_cols + ['next_day_profitable']).copy()
model_df['long_short_ratio'] = model_df['long_short_ratio'].replace([np.inf, -np.inf], np.nan)
model_df = model_df.dropna(subset=['long_short_ratio'])

X = model_df[feat_cols]
y = model_df['next_day_profitable'].astype(int)

print(f"Modeling rows: {len(model_df)}, positive rate: {y.mean():.2%}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=10,
                              random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("\nClassification report (next-day profitable = 1):")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

# feature importance
importance = pd.Series(clf.feature_importances_, index=feat_cols).sort_values()
fig, ax = plt.subplots(figsize=(6.5, 4))
importance.plot(kind='barh', ax=ax, color='#1f77b4')
ax.set_title('Feature Importance — Predicting Next-Day Profitability')
plt.tight_layout()
plt.savefig('/home/claude/project/charts/05_model_feature_importance.png')
plt.close()

print("\nFeature importances:")
print(importance.sort_values(ascending=False))
print("\nSaved chart 05_model_feature_importance.png")
