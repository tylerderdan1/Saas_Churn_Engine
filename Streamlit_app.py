"""
streamlit_app.py
Interactive Streamlit dashboard for the SaaS Churn Engine.
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page Config ─────────────────────────────────────────
st.set_page_config(
    page_title="SaaS Churn Engine",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "models/churn_model.pkl"
DATA_PATH  = "data/saas_data.csv"

# ── Load Artifacts ───────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    a = joblib.load(MODEL_PATH)
    return a["model"], a["label_encoder"], a["features"]

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)

model, le, FEATURES = load_model()
df = load_data()

# ── Sidebar ──────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
st.sidebar.title("SaaS Churn Engine")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["📊 Dashboard", "🔮 Predict Churn", "📈 Survival Analysis"])

# ── Helper ──────────────────────────────────────────────
def risk_badge(prob):
    if prob >= 0.75:   return "🔴 High Risk"
    elif prob >= 0.45: return "🟡 Medium Risk"
    else:              return "🟢 Low Risk"

def prepare(inputs: dict):
    usage_velocity = inputs["monthly_logins_m3"] - inputs["monthly_logins_m2"]
    avg_logins     = (inputs["monthly_logins_m1"] + inputs["monthly_logins_m2"] + inputs["monthly_logins_m3"]) / 3
    plan_encoded   = le.transform([inputs["plan"]])[0]
    row = {**inputs, "usage_velocity": usage_velocity, "avg_logins": avg_logins, "plan_encoded": plan_encoded}
    return pd.DataFrame([row])[FEATURES]


# ════════════════════════════════════════════════════════
# PAGE 1 — Dashboard
# ════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("📊 SaaS Churn Dashboard")

    if df is None:
        st.warning("No data found. Run `generate_data.py` first.")
        st.stop()

    total     = len(df)
    churned   = df["churn"].sum()
    churn_rate = churned / total

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{total:,}")
    col2.metric("Churned",         f"{churned:,}")
    col3.metric("Churn Rate",      f"{churn_rate:.1%}")
    col4.metric("Avg Tenure",      f"{df['tenure_months'].mean():.1f} mo")

    st.markdown("---")

    # Charts row
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.patch.set_facecolor("#0f172a")
    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

    # Churn by plan
    plan_churn = df.groupby("plan")["churn"].mean().sort_values(ascending=False)
    colors = ["#f87171", "#fbbf24", "#34d399", "#818cf8"]
    axes[0].bar(plan_churn.index, plan_churn.values, color=colors[:len(plan_churn)])
    axes[0].set_title("Churn Rate by Plan", color="white", fontweight="bold")
    axes[0].set_ylabel("Churn Rate", color="white")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    # NPS vs churn
    axes[1].scatter(
        df[df["churn"]==0]["nps_score"], df[df["churn"]==0]["feature_usage"],
        alpha=0.4, color="#34d399", s=15, label="Retained"
    )
    axes[1].scatter(
        df[df["churn"]==1]["nps_score"], df[df["churn"]==1]["feature_usage"],
        alpha=0.4, color="#f87171", s=15, label="Churned"
    )
    axes[1].set_title("NPS vs Feature Usage", color="white", fontweight="bold")
    axes[1].set_xlabel("NPS Score", color="white")
    axes[1].set_ylabel("Feature Usage", color="white")
    axes[1].legend(facecolor="#1e293b", labelcolor="white")

    # Tenure distribution
    axes[2].hist(df[df["churn"]==0]["tenure_months"], bins=20, alpha=0.7, color="#34d399", label="Retained")
    axes[2].hist(df[df["churn"]==1]["tenure_months"], bins=20, alpha=0.7, color="#f87171", label="Churned")
    axes[2].set_title("Tenure Distribution", color="white", fontweight="bold")
    axes[2].set_xlabel("Months", color="white")
    axes[2].legend(facecolor="#1e293b", labelcolor="white")

    st.pyplot(fig)
    plt.close()

    st.markdown("### 📋 Sample Data")
    st.dataframe(df.head(20), use_container_width=True)


# ════════════════════════════════════════════════════════
# PAGE 2 — Predict
# ════════════════════════════════════════════════════════
elif page == "🔮 Predict Churn":
    st.title("🔮 Churn Predictor")

    if model is None:
        st.warning("Model not found. Run `train.py` first.")
        st.stop()

    with st.form("predict_form"):
        col1, col2 = st.columns(2)

        with col1:
            plan             = st.selectbox("Plan",             ["Free", "Starter", "Pro", "Enterprise"])
            tenure_months    = st.slider("Tenure (months)",     1, 60, 12)
            monthly_spend    = st.number_input("Monthly Spend ($)", 0.0, 1000.0, 50.0)
            nps_score        = st.slider("NPS Score",           0, 10, 7)

        with col2:
            monthly_logins_m1 = st.slider("Logins — Month 1",  0, 60, 20)
            monthly_logins_m2 = st.slider("Logins — Month 2",  0, 60, 18)
            monthly_logins_m3 = st.slider("Logins — Month 3",  0, 60, 5)
            support_tickets   = st.slider("Support Tickets",   0, 10, 1)
            feature_usage     = st.slider("Feature Usage",     0, 50, 15)

        submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

    if submitted:
        inputs = dict(
            plan=plan, tenure_months=tenure_months, monthly_spend=monthly_spend,
            monthly_logins_m1=monthly_logins_m1, monthly_logins_m2=monthly_logins_m2,
            monthly_logins_m3=monthly_logins_m3, support_tickets=support_tickets,
            feature_usage=feature_usage, nps_score=nps_score
        )
        X    = prepare(inputs)
        prob = float(model.predict_proba(X)[0][1])

        st.markdown("---")
        r1, r2, r3 = st.columns(3)
        r1.metric("Churn Probability", f"{prob:.1%}")
        r2.metric("Risk Level",        risk_badge(prob))
        r3.metric("Prediction",        "⚠️ Will Churn" if prob >= 0.5 else "✅ Will Retain")

        if prob >= 0.75:
            st.error("**Action Required:** Immediate outreach — offer discount or escalate to CS team.")
        elif prob >= 0.45:
            st.warning("**Watch closely:** Schedule a health check call and surface unused features.")
        else:
            st.success("**Healthy customer:** Focus on upsell / expansion opportunities.")

        # Feature importance bar
        st.markdown("### 🔍 Feature Importance")
        importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True)
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        fig2.patch.set_facecolor("#0f172a")
        ax2.set_facecolor("#1e293b")
        ax2.barh(importances.index, importances.values, color="#38bdf8")
        ax2.set_title("Feature Importance", color="white")
        ax2.tick_params(colors="white")
        for spine in ax2.spines.values(): spine.set_edgecolor("#334155")
        st.pyplot(fig2)
        plt.close()


# ════════════════════════════════════════════════════════
# PAGE 3 — Survival
# ════════════════════════════════════════════════════════
elif page == "📈 Survival Analysis":
    st.title("📈 Survival Analysis")

    if df is None:
        st.warning("No data found. Run `generate_data.py` first.")
        st.stop()

    st.markdown("Kaplan-Meier style survival curves showing how long customers stay before churning.")

    plans = ["All"] + list(df["plan"].unique())
    selected = st.multiselect("Filter by plan", plans, default=["All"])

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_edgecolor("#334155")

    palette = {"All": "#38bdf8", "Free": "#f87171", "Starter": "#fbbf24", "Pro": "#34d399", "Enterprise": "#818cf8"}

    def manual_km(durations, events):
        df_km = pd.DataFrame({"T": durations, "E": events}).sort_values("T").reset_index(drop=True)
        times, survival = [0], [1.0]
        S, n = 1.0, len(df_km)
        for t, g in df_km.groupby("T"):
            d = g["E"].sum()
            S *= (1 - d / n)
            n -= len(g)
            times.append(t)
            survival.append(S)
        return np.array(times), np.array(survival)

    for plan in selected:
        subset = df if plan == "All" else df[df["plan"] == plan]
        t, s = manual_km(subset["tenure_months"].values, subset["churn"].values)
        ax.step(t, s, where="post", label=plan, color=palette.get(plan, "#ffffff"), linewidth=2)
        ax.fill_between(t, s, alpha=0.08, step="post", color=palette.get(plan, "#ffffff"))

    ax.set_xlabel("Tenure (months)", color="white")
    ax.set_ylabel("Survival Probability", color="white")
    ax.set_title("Customer Survival Curves", color="white", fontweight="bold")
    ax.legend(facecolor="#1e293b", labelcolor="white")
    ax.grid(True, alpha=0.2)
    st.pyplot(fig)
    plt.close()

    st.markdown("### 📊 Churn Stats by Plan")
    stats = df.groupby("plan")["churn"].agg(
        Total="count", Churned="sum", Churn_Rate="mean"
    ).reset_index()
    stats["Churn_Rate"] = stats["Churn_Rate"].map("{:.1%}".format)
    st.dataframe(stats, use_container_width=True)
