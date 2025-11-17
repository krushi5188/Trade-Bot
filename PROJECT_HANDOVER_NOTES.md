# Project Handover Notes

**From:** Jules
**Date:** 2025-11-16

## 1. CRITICAL: Strategic Pivot to a Professional Pipeline

**This document supersedes all previous handover notes.**

Based on critical expert feedback, the project has undergone a major strategic pivot. The previous experimental approach, termed "Genetic Programming" (which involved attempting to "breed" models by swapping their internal decision trees), has been **completely abandoned**.

**Reasoning for Abandonment:** The approach was fundamentally flawed.
1.  **Conceptual Flaw:** In gradient boosting models like LightGBM, decision trees are interdependent. Each tree is built to correct the errors of the previous ones. Swapping them between models is mathematically invalid and destroys the model's integrity.
2.  **Technical Flaw:** The LightGBM library does not support loading a model from the modified JSON structure we were creating. This represents a hard technical block.
3.  **Financial Flaw:** The approach was naive and did not account for the non-stationary nature of financial markets, regime changes, or proper risk management.

**All code related to `gene_extractor.py` and `genetic_breeder.py` should now be considered deprecated and is for historical reference only.**

## 2. The New Strategy: Automated Retraining & Validation

The project is now focused on building a robust, industry-standard **Automated Retraining and Validation Pipeline**. The goal is not to create a single perfect model, but to create a system that constantly adapts to changing market conditions.

The official plan is now documented in **`ROADMAP.md` under "Tier 4: Automated Retraining & Professional Validation Pipeline"**.

The key components to be built are:
*   **Structured Hyperparameter Tuning:** Using Optuna with `TimeSeriesSplit` cross-validation.
*   **Walk-Forward Validation Engine:** The gold standard for financial backtesting.
*   **Dynamic Feature Selection:** To adapt to changing market regimes by identifying the most currently predictive features.
*   **Integrated Risk Management:** To evaluate performance within a realistic risk framework.

## 3. Immediate Next Steps

The `ROADMAP.md` has been updated to reflect this new direction.

The immediate next task is to begin implementing the first component of this new pipeline: building the **structured hyperparameter tuning script** using Optuna and TimeSeriesSplit.
