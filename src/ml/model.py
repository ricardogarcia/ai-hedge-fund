# 📦 Imports
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import classification_report

# 🧪 Simulate dataset
np.random.seed(42)
n_samples = 200

data = pd.DataFrame({
    'rsi': np.random.uniform(10, 70, n_samples),
    'sma_50_above_sma_200': np.random.choice([0, 1], n_samples),
    'llm_sentiment_score': np.random.uniform(0, 1, n_samples),
    'label': np.random.choice(['bullish', 'bearish', 'neutral'], n_samples)
})

label_map = {'bullish': 0, 'bearish': 1, 'neutral': 2}
data['label_encoded'] = data['label'].map(label_map)

X = data.drop(columns=['label', 'label_encoded'])
y = data['label_encoded']

# 🧠 Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 🌲 Base models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss',
                    n_estimators=25, max_depth=3, random_state=42)

# 🔁 Cross-val prediction for meta-learner training
rf_preds_train = cross_val_predict(rf, X_train, y_train, method="predict_proba", cv=5)
xgb_preds_train = cross_val_predict(xgb, X_train, y_train, method="predict_proba", cv=5)
meta_features_train = np.hstack((rf_preds_train, xgb_preds_train))

# 🧮 Meta-learner
meta_model = LogisticRegression(multi_class='multinomial', max_iter=500)
meta_model.fit(meta_features_train, y_train)

# 🔍 Final evaluation on test set
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

rf_test_preds = rf.predict_proba(X_test)
xgb_test_preds = xgb.predict_proba(X_test)

meta_features_test = np.hstack((rf_test_preds, xgb_test_preds))
final_preds = meta_model.predict(meta_features_test)
final_probs = meta_model.predict_proba(meta_features_test)

# 🎯 Output: Three probabilities per row
for i, probs in enumerate(final_probs[:5]):
    print(f"Sample {i+1} → Bullish: {probs[0]:.2f}, Bearish: {probs[1]:.2f}, Neutral: {probs[2]:.2f}")

# 📊 Report
report = classification_report(y_test, final_preds, target_names=label_map.keys())
print(report)
