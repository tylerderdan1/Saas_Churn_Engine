from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import os

MODEL_PATH = "models/churn_model.pkl"

app = FastAPI(
    title="SaaS Churn Engine API",
    description="Predict customer churn probability in real-time.",
    version="1.0.0"
)

# Load model at startup
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Run train.py first.")

artifact      = joblib.load(MODEL_PATH)
model         = artifact["model"]
label_encoder = artifact["label_encoder"]
FEATURES      = artifact["features"]


# ── Request / Response Schemas ───────────────────────────

class CustomerInput(BaseModel):
    tenure_months:      int   = Field(..., ge=0,   description="Months as customer")
    monthly_spend:      float = Field(..., ge=0,   description="Monthly spend in USD")
    monthly_logins_m1:  int   = Field(..., ge=0)
    monthly_logins_m2:  int   = Field(..., ge=0)
    monthly_logins_m3:  int   = Field(..., ge=0)
    support_tickets:    int   = Field(..., ge=0)
    feature_usage:      int   = Field(..., ge=0)
    nps_score:          int   = Field(..., ge=0, le=10)
    plan:               str   = Field(..., description="Free | Starter | Pro | Enterprise")

class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction:  bool
    risk_level:        str
    recommendation:    str


# ── Helper ──────────────────────────────────────────────

def get_risk_level(prob: float) -> tuple[str, str]:
    if prob >= 0.75:
        return "🔴 High",    "Immediate outreach needed — offer discount or dedicated support."
    elif prob >= 0.45:
        return "🟡 Medium",  "Schedule a check-in call and highlight unused features."
    else:
        return "🟢 Low",     "Customer is healthy — focus on upsell opportunities."


def prepare_features(data: CustomerInput) -> pd.DataFrame:
    plan_encoded = label_encoder.transform([data.plan])[0]
    usage_velocity = data.monthly_logins_m3 - data.monthly_logins_m2
    avg_logins = (data.monthly_logins_m1 + data.monthly_logins_m2 + data.monthly_logins_m3) / 3

    row = {
        "tenure_months":     data.tenure_months,
        "monthly_spend":     data.monthly_spend,
        "monthly_logins_m1": data.monthly_logins_m1,
        "monthly_logins_m2": data.monthly_logins_m2,
        "monthly_logins_m3": data.monthly_logins_m3,
        "support_tickets":   data.support_tickets,
        "feature_usage":     data.feature_usage,
        "nps_score":         data.nps_score,
        "usage_velocity":    usage_velocity,
        "avg_logins":        avg_logins,
        "plan_encoded":      plan_encoded,
    }
    return pd.DataFrame([row])[FEATURES]


# ── Routes ───────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "SaaS Churn Engine API is running 🚀"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    try:
        X    = prepare_features(customer)
        prob = float(model.predict_proba(X)[0][1])
        pred = prob >= 0.5
        risk, recommendation = get_risk_level(prob)

        return PredictionResponse(
            churn_probability = round(prob, 4),
            churn_prediction  = pred,
            risk_level        = risk,
            recommendation    = recommendation
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch")
def predict_batch(customers: list[CustomerInput]):
    results = []
    for c in customers:
        X    = prepare_features(c)
        prob = float(model.predict_proba(X)[0][1])
        pred = prob >= 0.5
        risk, rec = get_risk_level(prob)
        results.append({
            "churn_probability": round(prob, 4),
            "churn_prediction":  pred,
            "risk_level":        risk,
            "recommendation":    rec
        })
    return {"predictions": results, "total": len(results)}
if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading
    import os

    # Guard against double browser launch when reload=True spawns a second process
    if os.environ.get("UVICORN_WORKER") != "true":
        def open_browser():
            import time
            time.sleep(1.5)
            webbrowser.open("http://127.0.0.1:8000/docs")

        os.environ["UVICORN_WORKER"] = "true"
        threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)