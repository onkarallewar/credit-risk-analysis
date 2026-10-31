#!/usr/bin/env bash
set -e
echo "[1/3] Installing libraries..."
pip install -r requirements.txt
echo "[2/3] Running SQL exploration..."
python src/sql_layer.py
echo "[3/3] Training model + SHAP + risk tiers, then Power BI export..."
python src/analyze.py
python src/export_powerbi.py
echo "Done. Charts in reports/, Power BI file at reports/portfolio_scored.csv"
