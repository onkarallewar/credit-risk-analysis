# Credit Card Default — PD, Risk Tiers & Expected Loss

An end-to-end **credit risk** project: predict each customer's **Probability of
Default (PD)**, explain it, segment the portfolio into risk tiers, quantify
**expected loss**, and hand off a **Power BI** dashboard file.

Tools: **Python, SQL (SQLite), scikit-learn, XGBoost, SHAP, Power BI**

## Flow (see PROJECT_FLOW.txt for the detailed step-by-step)
Data → SQL exploration → EDA → cleaning → feature engineering → preprocessing
→ train 3 models (XGBoost wins) → SHAP explainability → risk tiers →
expected loss → Power BI dashboard.

## Run it
```bash
pip install -r requirements.txt
python src/sql_layer.py        # 1) SQL risk exploration
python src/analyze.py          # 2) clean, engineer, train, SHAP, tiers, loss
python src/export_powerbi.py   # 3) build the Power BI CSV
```
Then in Power BI: Get Data → Text/CSV → `reports/portfolio_scored.csv`.

## Key results (trained on the real 30,000-row dataset)
- XGBoost: **ROC-AUC 0.78 (Gini 0.56)**, **recall 63%** on defaulters
- SHAP top drivers: cumulative delinquency, latest repayment status, utilization
- 4 risk tiers with actual default rate rising **5% → 84%**
- Top 20% of accounts hold **56%** of all defaults
- Portfolio **expected loss ≈ US$18M** (PD × exposure)

## Files
- `src/sql_layer.py` — loads data into SQLite, runs risk queries
- `src/analyze.py` — cleaning, feature engineering, model benchmark, SHAP, tiers, expected loss
- `src/export_powerbi.py` — writes `reports/portfolio_scored.csv` for Power BI
- `reports/risk_tiers.png`, `reports/shap_drivers.png` — charts
- `data/raw/UCI_Credit_Card.csv` — the dataset (bundled)

## Honest notes
- SQLite is used locally; the SQL is standard and transfers to MySQL.
- Expected loss assumes loss-given-default = 100% (a simplifying assumption).
