**Project State & Memory Handover for Jules:**

**1. High-Level Goal:**
We are building a multi-asset (Crypto, Forex, Metals) trading AI. We have completed Tier 1 (data pipelines, baseline model) and are now in Tier 2 (model improvement and experimentation).

**2. Current State & Immediate Task:**
*   **Active Task:** We have successfully trained and evaluated a new LightGBM model, which is our new champion model.
*   **Immediate Next Step:** The next logical step is to develop a strategy overlay, which will involve building a rules-based system for dynamic position sizing and risk management.

**3. Critical History & Key Learnings (The "Smaller Details"):**

*   **Data Pipeline Evolution:**
    *   The original data fusion logic existed only in a Jupyter Notebook (`notebooks/tier1/01_data_fusion.ipynb`). This was a major issue.
    *   I created a production script, `src/tier1/data_pipelines/fuse_data.py`, to fix this.
    *   During this, I solved a critical **timezone bug** (`TypeError: Cannot join tz-naive with tz-aware`). The solution was to standardize all datetime indices to UTC within the `fuse_data.py` script.
    *   The final `master_dataset.parquet` is currently missing economic calendar data, but the fusion script is built to handle this gracefully.

*   **The Hurst Exponent Performance Debacle:**
    *   The original Hurst calculation in `pandas.rolling().apply()` was far too slow and caused repeated timeouts.
    *   I attempted to fix this with the `hurst-exponent` library, but it was **broken and unreliable**, leading to a series of `ModuleNotFoundError` issues with its hidden dependencies (`powerlaw-function` vs. the correct `powerlaw`).
    *   **The final, working solution** is a high-performance NumPy implementation in `feature_engineering_v2.py` that uses `numpy.lib.stride_tricks.as_strided` to create rolling windows efficiently.
    *   **CRITICAL BUG:** The V2 feature engineering script is currently producing all `NaN` values for the Hurst Exponent, which is why we reverted to the V1 features. This needs to be debugged in the future.

*   **Dependency Management:**
    *   The `requirements.txt` file has been through significant debugging. Key findings: `MarketAux` is not a real PyPI library (we use `requests` for it), and `hurst-exponent` should be avoided. The current `requirements.txt` is stable.

*   **Modeling Strategy Insights:**
    *   Our original champion model was the **baseline XGBoost**, with **54% accuracy** and a **3.06% backtested return**.
    *   An attempt to hyperparameter-tune the XGBoost model resulted in **worse** performance, indicating overfitting.
    *   Our new champion model is the **LightGBM model**, which achieved **64% accuracy** on the stable V1 features. This is a significant improvement and is our new benchmark to beat.

*   **Project Structure:**
    *   The project is organized into tiers (e.g., `src/tier1`, `src/tier2`). Original exploration was done in `notebooks/tier1`, and production-ready code is being built in the `src/tierX` directories.

**Goal:** To build upon our new, more accurate LightGBM model by developing a sophisticated strategy overlay that can intelligently manage risk and position sizing, with the ultimate goal of creating a profitable trading strategy.
