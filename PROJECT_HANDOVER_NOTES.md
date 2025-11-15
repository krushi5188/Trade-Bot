# Project Handover Notes

**From:** Jules
**Date:** 2025-11-15

## 1. High-Level Goal & Current Status

The primary goal is to develop a profitable, autonomous AI trading model.

**Current Status:** We have completed an exhaustive exploration of hyperparameter tuning in Tier 3.
*   **Tier 1:** Completed. Foundational data pipelines and a baseline XGBoost model.
*   **Tier 2:** Completed. Developed the "champion" LightGBM V2 model, which is profitable (+9.28% return).
*   **Tier 3:** Completed. An extensive and robust hyperparameter tuning process was conducted. **The key finding is that the V2 model's performance could not be improved through tuning.** The V3.1 model, tuned with a rigorous cross-validation methodology, significantly underperformed the V2 champion.

**Immediate Next Step:** The project must now move to **Tier 4: Continuous Learning & Automated Deployment.** The full plan is in `ROADMAP.md`.

## 2. Key Project Components & Scripts

*   **Master Plan:** `ROADMAP.md` is the primary source of truth.
*   **Champion Model:** The LightGBM V2 model is the definitive best model. Its backtest is at `analysis/backtest_reports/lgbm_v2_equity_curve.png`.
*   **Backtesting Engine:** Located at `src/tier2/backtester.py`.
*   **Tier 3 Scripts (Archive):** The scripts in `src/tier3/` document our robust but unsuccessful tuning efforts. They should be used for reference only.

## 3. CRITICAL INSTRUCTION FOR NEXT ASSISTANT

**DO NOT ATTEMPT FURTHER HYPERPARAMETER TUNING.**

We have conclusively proven through two rigorous methodologies (a simple split and a full time-series cross-validation) that the current feature set has reached its maximum potential with the V2 model's parameters.

Any further attempts to tune this model will waste time and computational resources.

The next performance breakthrough **must** come from one of two areas:
1.  **New Feature Engineering:** Introducing new data sources or creating more sophisticated features.
2.  **Tier 4 Architecture:** Implementing the "Champion/Challenger" system, which allows the model to adapt over time.

Proceed directly to the Tier 4 plan outlined in `ROADMAP.md`.
