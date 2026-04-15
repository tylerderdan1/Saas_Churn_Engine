"""
survival_analysis.py
Performs Kaplan-Meier survival analysis to estimate how long customers
stay before churning. Saves plots to outputs/survival/
Run AFTER generate_data.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Optional: lifelines library for proper KM curves
try:
    from lifelines import KaplanMeierFitter
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False
    print("⚠️  lifelines not installed. Using manual KM implementation.")
    print("    Install with: pip install lifelines")

OUTPUT_DIR = "outputs/survival"
DATA_PATH  = "data/saas_data.csv"


def manual_km(durations, event_observed):
    """Simple manual Kaplan-Meier estimator (fallback)."""
    df = pd.DataFrame({"T": durations, "E": event_observed})
    df = df.sort_values("T").reset_index(drop=True)
    times, survival = [0], [1.0]
    S = 1.0
    n = len(df)
    for t, group in df.groupby("T"):
        d = group["E"].sum()
        S *= (1 - d / n)
        n -= len(group)
        times.append(t)
        survival.append(S)
    return np.array(times), np.array(survival)


def plot_overall(df, ax):
    durations = df["tenure_months"]
    events    = df["churn"]

    if HAS_LIFELINES:
        kmf = KaplanMeierFitter()
        kmf.fit(durations, event_observed=events, label="All Customers")
        kmf.plot_survival_function(ax=ax, ci_show=True)
    else:
        t, s = manual_km(durations.values, events.values)
        ax.step(t, s, where="post", label="All Customers", color="#38bdf8")
        ax.fill_between(t, s, alpha=0.15, step="post", color="#38bdf8")

    ax.set_title("Overall Customer Survival Curve", fontweight="bold")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Survival Probability")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_by_plan(df, ax):
    colors = {"Free": "#f87171", "Starter": "#fbbf24", "Pro": "#34d399", "Enterprise": "#818cf8"}

    for plan, color in colors.items():
        subset = df[df["plan"] == plan]
        if subset.empty:
            continue
        if HAS_LIFELINES:
            kmf = KaplanMeierFitter()
            kmf.fit(subset["tenure_months"], event_observed=subset["churn"], label=plan)
            kmf.plot_survival_function(ax=ax, ci_show=False, color=color)
        else:
            t, s = manual_km(subset["tenure_months"].values, subset["churn"].values)
            ax.step(t, s, where="post", label=plan, color=color)

    ax.set_title("Survival by Plan Type", fontweight="bold")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Survival Probability")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_churn_timeline(df, ax):
    churned = df[df["churn"] == 1]["tenure_months"]
    ax.hist(churned, bins=20, color="#f87171", edgecolor="white", alpha=0.85)
    ax.axvline(churned.median(), color="white", linestyle="--", label=f"Median: {churned.median():.0f} mo")
    ax.set_title("When Do Customers Churn?", fontweight="bold")
    ax.set_xlabel("Tenure at Churn (months)")
    ax.set_ylabel("Number of Customers")
    ax.legend()
    ax.grid(True, alpha=0.3)


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("SaaS Churn — Survival Analysis", fontsize=16, fontweight="bold", y=1.02)

    plot_overall(df, axes[0])
    plot_by_plan(df, axes[1])
    plot_churn_timeline(df, axes[2])

    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/survival_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Survival plots saved to {out_path}")

    # Summary stats
    print("\n📊 Churn Summary by Plan:")
    print(df.groupby("plan")["churn"].agg(["sum", "mean", "count"])
            .rename(columns={"sum": "churned", "mean": "churn_rate", "count": "total"})
            .assign(churn_rate=lambda x: x["churn_rate"].map("{:.1%}".format))
            .to_string())


if __name__ == "__main__":
    run()
    