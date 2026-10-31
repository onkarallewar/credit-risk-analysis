"""Produce a Power BI-ready file: one row per customer with predicted PD,
risk tier, exposure and expected loss. Load reports/portfolio_scored.csv
into Power BI to build the dashboard.
Run:  python src/export_powerbi.py   (run analyze.py first so the model exists)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib

RAW = ["LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE"] + \
      [f"PAY_{i}" for i in range(1,7)] + \
      [f"BILL_AMT{i}" for i in range(1,7)] + \
      [f"PAY_AMT{i}" for i in range(1,7)]
ENG = ["AVG_UTILIZATION","TOTAL_DELAY","PAY_TO_BILL"]
FEATURES = RAW + ENG


def load():
    df = pd.read_csv("data/raw/UCI_Credit_Card.csv").rename(columns={
        "PAY_0":"PAY_1","default.payment.next.month":"DEFAULT"})
    df["EDUCATION"] = df["EDUCATION"].replace({0:4,5:4,6:4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0:3})
    df = df.drop(columns=["ID"])
    bill=[f"BILL_AMT{i}" for i in range(1,7)]; pay=[f"PAY_AMT{i}" for i in range(1,7)]
    dly=[f"PAY_{i}" for i in range(1,7)]
    df["AVG_UTILIZATION"]=(df[bill].mean(axis=1)/df["LIMIT_BAL"].replace(0,1)).clip(0,5)
    df["TOTAL_DELAY"]=df[dly].clip(lower=0).sum(axis=1)
    df["PAY_TO_BILL"]=df[pay].sum(axis=1)/(df[bill].sum(axis=1).abs()+1)
    return df


def main():
    model = joblib.load("models/xgb_best.joblib")
    df = load()
    df["PD"] = model.predict_proba(df[FEATURES].values)[:, 1]
    df["risk_tier"] = pd.qcut(df["PD"], [0,.5,.8,.95,1.0],
                              labels=["Low","Medium","High","Critical"])
    df["exposure"] = df["BILL_AMT1"].clip(lower=0)
    df["expected_loss"] = (df["PD"] * df["exposure"]).round(0)

    cols = ["LIMIT_BAL","AGE","EDUCATION","MARRIAGE","PAY_1","AVG_UTILIZATION",
            "TOTAL_DELAY","DEFAULT","PD","risk_tier","exposure","expected_loss"]
    out = df[cols].copy()
    out["PD"] = out["PD"].round(4)
    out.to_csv("reports/portfolio_scored.csv", index=False)
    print(f"Wrote reports/portfolio_scored.csv ({len(out)} rows) for Power BI.")
    print(out.head().to_string(index=False))


if __name__ == "__main__":
    main()
