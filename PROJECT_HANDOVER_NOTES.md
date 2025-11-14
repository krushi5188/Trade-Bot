# **PROJECT HANDOVER & CONTINUATION PLAN**

## **SECTION 1: INSTRUCTIONS FOR THE NEW AGENT (READ THIS FIRST)**
You are Jules, continuing a session after a restart. Your entire memory and plan are in this document. Your first action MUST be to set the plan outlined in Section 3 below using the `set_plan` tool. Then, you must sequentially mark steps 1, 2, 3, and 4 as complete using the `plan_step_complete` tool, as they have already been done. This will bring you to the current active step.

## **SECTION 2: CRITICAL HISTORY & KEY LEARNINGS (Your Memory)**
*   **Data Pipeline:** The original data fusion was stuck in a notebook. I created `src/tier1/data_pipelines/fuse_data.py` to fix this. I also solved a critical **timezone bug** by standardizing all data to UTC.
*   **Feature Engineering (V2 Failure):** My attempt to build advanced V2 features is currently blocked. The feature engineering script (`src/tier2/feature_engineering_v2.py`) produces `NaN` values for the Hurst Exponent, leading to a completely empty dataset after cleaning. This is a major bug to be addressed later. To keep the project moving, we have **reverted to using the stable V1 feature set** (`data/processed/features_03_final.parquet`).
*   **Hurst Performance:** The original Hurst calculation was too slow. My attempt to use the `hurst-exponent` library failed because the library is broken. The final, working solution is a high-performance NumPy implementation in `feature_engineering_v2.py`, but it contains the `NaN` bug mentioned above.
*   **Model Performance:**
    *   The original XGBoost baseline had **54% accuracy** and a **3.06% backtested return**.
    *   Our new champion model is **LightGBM**, which achieved **64% accuracy** on the stable V1 features. This is our new benchmark to beat.

## **SECTION 3: THE OFFICIAL PLAN OF RECORD**
This is the plan you MUST adopt. Copy the text below *exactly* into the `set_plan` tool as your first action.

1.  *Build the Backtesting Engine.*
2.  *Refine the Model with Hyperparameter Tuning.*
3.  *Engineer Advanced and Interaction Features.*
4.  *Experiment with Alternative Model Architectures.*
5.  *Develop a Strategy Overlay.*
6.  *Incorporate Advanced Quantitative Techniques (GARCH, Skew/Kurtosis).*
7.  *Complete pre commit steps*
8.  *Submit the change.*

## **SECTION 4: CURRENT STATUS**
- **Steps 1-4 are COMPLETE.**
- **The current active step is #5: Develop a Strategy Overlay.** You must reach this step by setting the plan and completing the previous steps as instructed in Section 1.
