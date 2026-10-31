"""SQL layer for the credit-risk pipeline.
Loads the raw data into a SQLite database and runs the kind of exploratory /
aggregation queries a risk analyst writes BEFORE modelling. Demonstrates basic
SQL: SELECT, WHERE, GROUP BY, CASE, aggregate functions, ORDER BY.
Run:  python src/sql_layer.py
"""
import sqlite3
import pandas as pd

DB = "data/credit.db"
CSV = "data/raw/UCI_Credit_Card.csv"


def build_database():
    """Load the CSV into a SQLite table (mimics data landing in a warehouse)."""
    df = pd.read_csv(CSV).rename(columns={
        "PAY_0": "PAY_1", "default.payment.next.month": "DEFAULT"})
    con = sqlite3.connect(DB)
    df.to_sql("credit_clients", con, if_exists="replace", index=False)
    con.close()
    print(f"Loaded {len(df)} rows into {DB} -> table 'credit_clients'")


def run(sql, desc):
    con = sqlite3.connect(DB)
    out = pd.read_sql_query(sql, con)
    con.close()
    print(f"\n--- {desc} ---")
    print(out.to_string(index=False))
    return out


def main():
    build_database()

    # 1. Portfolio-level default rate (basic aggregate)
    run("""
        SELECT COUNT(*)                         AS total_clients,
               SUM("DEFAULT")                   AS defaulters,
               ROUND(AVG("DEFAULT") * 100, 1)   AS default_rate_pct
        FROM credit_clients;
    """, "Portfolio default rate")

    # 2. Default rate by credit-limit band (CASE + GROUP BY)
    run("""
        SELECT CASE
                 WHEN LIMIT_BAL < 50000  THEN '1. <50k'
                 WHEN LIMIT_BAL < 150000 THEN '2. 50k-150k'
                 WHEN LIMIT_BAL < 300000 THEN '3. 150k-300k'
                 ELSE '4. 300k+'
               END                              AS credit_band,
               COUNT(*)                         AS clients,
               ROUND(AVG("DEFAULT") * 100, 1)   AS default_rate_pct
        FROM credit_clients
        GROUP BY credit_band
        ORDER BY credit_band;
    """, "Default rate by credit-limit band")

    # 3. Default rate by most-recent repayment status (risk driver check)
    run("""
        SELECT PAY_1                            AS last_repayment_status,
               COUNT(*)                         AS clients,
               ROUND(AVG("DEFAULT") * 100, 1)   AS default_rate_pct
        FROM credit_clients
        GROUP BY PAY_1
        ORDER BY default_rate_pct DESC;
    """, "Default rate by latest repayment status")

    # 4. High-risk segment: late last month AND highly utilised (WHERE + filter)
    run("""
        SELECT COUNT(*)                         AS high_risk_clients,
               ROUND(AVG("DEFAULT") * 100, 1)   AS default_rate_pct,
               ROUND(AVG(LIMIT_BAL), 0)         AS avg_credit_limit
        FROM credit_clients
        WHERE PAY_1 >= 2
          AND BILL_AMT1 > 0.8 * LIMIT_BAL;
    """, "High-risk segment (2+ months late & >80% utilised)")

    # 5. Default rate by education (data-quality + demographic view)
    run("""
        SELECT EDUCATION,
               COUNT(*)                         AS clients,
               ROUND(AVG("DEFAULT") * 100, 1)   AS default_rate_pct
        FROM credit_clients
        GROUP BY EDUCATION
        ORDER BY EDUCATION;
    """, "Default rate by education code")


if __name__ == "__main__":
    main()
