import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv('data_cleaned.csv')
X = df.drop(columns=['label', 'high_invoice_reuse', 'multi_bank', 'frequent_drawdown'])
y = df['label']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
joblib.dump(model, 'risk_model.pkl')
print("模型已保存为 risk_model.pkl")
