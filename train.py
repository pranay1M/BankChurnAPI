import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib

# === load data ===
df = pd.read_csv("/users/pranaypakki/Tech/CSVfiles/BankCustomerChurnPrediction/Bank Customer Churn Prediction.csv")

# features / target
target = "churn"
y = df[target].astype(int)
X = df.drop(columns=["customer_id", target])  # drop IDs

# columns by type (from your EDA)
cat_cols = ["country", "gender"]
num_cols = [c for c in X.columns if c not in cat_cols]

# === preprocessing ===
pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
    ]
)

# === model ===
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced_subsample",
    n_jobs=-1,
)

# === full pipeline ===
pipe = Pipeline(steps=[
    ("pre", pre),
    ("model", rf),
])

# === train/test just to sanity-check ===
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
pipe.fit(X_tr, y_tr)
print("ROC-AUC:", roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1]))

# === persist single artifact ===
joblib.dump(pipe, "model.joblib")
print("Saved model.joblib")
