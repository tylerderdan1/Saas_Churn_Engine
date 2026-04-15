"""
generate_data.py
Generates synthetic SaaS customer data and saves it to data/saas_data.csv
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

def generate_saas_data(n_customers=1000):
    data = {}

    # Basic customer info
    data["customer_id"] = [f"CUST_{i:04d}" for i in range(n_customers)]
    data["plan"] = np.random.choice(["Free", "Starter", "Pro", "Enterprise"], n_customers, p=[0.3, 0.3, 0.25, 0.15])
    data["tenure_months"] = np.random.randint(1, 60, n_customers)
    data["monthly_spend"] = np.round(np.random.uniform(0, 500, n_customers), 2)

    # Engagement features
    data["monthly_logins_m1"] = np.random.randint(0, 60, n_customers)
    data["monthly_logins_m2"] = np.random.randint(0, 60, n_customers)
    data["monthly_logins_m3"] = np.random.randint(0, 60, n_customers)
    data["support_tickets"] = np.random.randint(0, 10, n_customers)
    data["feature_usage"] = np.random.randint(5, 50, n_customers)
    data["nps_score"] = np.random.randint(0, 11, n_customers)

    df = pd.DataFrame(data)

    # Engineered features
    df["usage_velocity"] = df["monthly_logins_m3"] - df["monthly_logins_m2"]
    df["avg_logins"] = (df["monthly_logins_m1"] + df["monthly_logins_m2"] + df["monthly_logins_m3"]) / 3

    # Churn logic (rule-based labeling)
    df["churn"] = np.where(
        (df["usage_velocity"] < -20) |
        (df["support_tickets"] > 7) |
        (df["nps_score"] <= 3) |
        ((df["avg_logins"] < 5) & (df["tenure_months"] > 6)),
        1, 0
    )

    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_saas_data(1000)
    df.to_csv("data/saas_data.csv", index=False)
    print(f"✅ Dataset created: {len(df)} customers | Churn rate: {df['churn'].mean():.1%}")