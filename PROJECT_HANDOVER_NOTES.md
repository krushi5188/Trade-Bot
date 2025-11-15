# Project Handover Notes

**From:** Jules
**Date:** 2025-11-15

## 1. High-Level Goal & Current Status

The primary goal of the project is to develop a profitable, autonomous AI trading model.

**Current Status:** We have successfully completed three tiers of development.
*   **Tier 1:** Built the foundational data pipelines and a baseline XGBoost model.
*   **Tier 2:** Developed an improved V2 feature set and a profitable LightGBM V2 model. This model is our current "champion" and achieved a **+9.28% return** in backtesting.
*   **Tier 3:** Conducted a full hyperparameter tuning study using Optuna to try and improve the V2 model. While the process was successful, the resulting "V3" model underperformed the V2 champion.

**Immediate Next Step:** The project is now ready to begin **Tier 4: Continuous Learning & Automated Deployment.** The full plan for this is detailed in the `ROADMAP.md` file. The goal is to build a system where the AI can automatically retrain and deploy itself.

## 2. Key Project Components & Scripts

*   **Master Plan:** The `ROADMAP.md` file contains the full technical specification and is the primary source of truth for the project's goals and history.
*   **Champion Model:** The best-performing model is the LightGBM V2 model. The backtest report is located at `analysis/backtest_reports/lgbm_v2_equity_curve.png`.
*   **Backtesting Engine:** A robust vectorized backtester is available in `src/tier2/backtester.py`. Scripts for running backtests are in `src/tier2/run_backtest_v2.py` and `src/tier3/run_backtest_v3.py`.
*   **Hyperparameter Tuning:** The scripts for the completed Tier 3 work are in `src/tier3/`. The Optuna study successfully ran and the code can be used as a template for future tuning work.

## 3. Critical Learnings & Instructions

*   **Environment Setup:** Before running any scripts, ensure all dependencies are installed by running `pip install -r requirements.txt`. This is the definitive list of required packages.
*   **Documentation is Key:** The user values up-to-date documentation. Always update the `ROADMAP.md` and this handover file after completing a major task.
*   **V2 Model is the Champion:** The hyperparameter tuning in Tier 3 did **not** produce a better model. The V2 model remains the best one we have. Future work should focus on the Tier 4 architecture or new feature engineering, not on re-tuning the V2 model.
*   **Verify, Don't Recompute:** The user prefers that we check for existing results (e.g., in `analysis/backtest_reports/`) before re-running computationally expensive scripts.
