import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix
)

# ── Config ──────────────────────────────────────────────
DATA_PATH  = "data/saas_data.csv"
MODEL_DIR  = "models"
MODEL_PATH = f"{MODEL_DIR}/churn_model.pkl"

FEATURES = [
    "tenure_months", "monthly_spend",
    "monthly_logins_m1", "monthly_logins_m2", "monthly_logins_m3",
    "support_tickets", "feature_usage", "nps_score",
    "usage_velocity", "avg_logins", "plan_encoded"
]
TARGET = "churn"
# ────────────────────────────────────────────────────────


def load_and_prepare(path):
    df = pd.read_csv(path)
    le = LabelEncoder()
    df["plan_encoded"] = le.fit_transform(df["plan"])
    return df, le


def train(df):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluation
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"🎯 ROC-AUC Score : {roc_auc_score(y_test, y_proba):.4f}")
    print(f"📉 Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=FEATURES)
    print("🔍 Top Features:")
    print(importances.sort_values(ascending=False).to_string())

    return model, X_test, y_test


def save_model(model, le):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"model": model, "label_encoder": le, "features": FEATURES}, MODEL_PATH)
    print(f"\n✅ Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    print("🚀 Loading data...")
    df, le = load_and_prepare(DATA_PATH)

    print("🏋️  Training model...")
    model, X_test, y_test = train(df)

    save_model(model, le)