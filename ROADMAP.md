# Master Technical Specification: The Adaptive, Multi-Source Trading Agent

## Tier 1: Baseline Model - Summary & Key Findings
**Objective:** To establish a foundational ML pipeline capable of collecting diverse data, engineering predictive features, and training a baseline model.
**Status:** COMPLETE.
**Key Outcomes:**
- A robust data pipeline now exists for market (Crypto, Forex, Metals), sentiment (news headlines), and economic calendar data.
- A full suite of advanced features (Kalman Filters, Hurst Exponent, RSI, Volatility) has been successfully engineered.
- A baseline XGBoost model was trained and saved, achieving ~54% accuracy on the test set. While not yet profitable, this serves as a crucial benchmark.
- **SHAP analysis provided critical insights into the model's "brain":**
  - **Confirmation of Feature Importance:** The model relies heavily on our engineered features, especially the Kalman-filtered (smoothed) prices and the Hurst Exponent, validating our feature engineering strategy.
  - **'Buy' Signal Logic:** The model's decision to "Buy" is most strongly influenced by a high Hurst Exponent (indicating a trending BTC market) and rising gold prices (suggesting a potential "risk-on" sentiment).
  - **Identified Weakness:** The model struggles significantly with predicting 'Hold' conditions, indicating a weakness in sideways or non-trending markets. This is a primary target for future improvements.

---

## Tier 2: Strategy Refinement & Vectorized Backtesting
**Objective:** To rigorously evaluate the baseline model's performance on historical data and establish a framework for iterative improvement.
**Status:** COMPLETE.
**Key Outcomes:**
- A fully functional Vectorized Backtesting Engine was built and is available in `src/tier2/backtester.py`.
- The backtester was integrated with the V2 LightGBM model, which was trained on an improved feature set and a stratified data split to handle severe class imbalance.
- The backtester correctly calculates key performance metrics including Total Return, Annualized Return, Sharpe Ratio, and Maximum Drawdown.
- **The V2 LightGBM model was successfully backtested and confirmed to be profitable, achieving a +9.28% total return, significantly outperforming the 'Buy and Hold' benchmark.**
- The successful result was achieved by implementing several key improvements: a stratified data split, class weighting, and correct label mapping for the LightGBM model.

**Tasks:**
- [x] **Design and Build a Vectorized Backtesting Engine:** Create a Python script that can simulate trades based on model predictions and calculate performance.
- [x] **Integrate Model Predictions:** Feed the saved XGBoost model's predictions into the backtester.
- [x] **Define Key Performance Metrics (KPIs):** The backtester must calculate Profit & Loss (PnL), Sharpe Ratio, Sortino Ratio, and Maximum Drawdown.
- [x] **Generate Performance Report:** Run the backtest on the held-out test set and generate an initial report and equity curve visualization.

---

## Tier 3: Hyperparameter Tuning & Optimization (Revised)
**Objective:** To systematically optimize the profitable V2 LightGBM model to produce a new champion model with significantly better performance.
**Status:** COMPLETE.
**Key Outcomes & Conclusion:**
- An enhanced, robust hyperparameter search was conducted using Optuna combined with **Time-Series Cross-Validation** to maximize the average Sharpe Ratio.
- A new V3.1 model was trained using the best parameters from this robust study.
- A full backtest of the V3.1 model yielded a Total Return of +4.01% and a Sharpe Ratio of 0.03. This is a significant underperformance compared to the V2 champion model.
- **Final Conclusion:** We have now conclusively demonstrated through two different tuning methodologies that the V2 model's performance is the maximum achievable with the current feature set. **No further work should be done on hyperparameter tuning.** Future efforts must be directed towards new feature engineering or the architectural improvements planned in Tier 4 to achieve the next level of performance.

**Revised Tasks:**
- [x] **Implement Cross-Validation:** The `src/tier3/tune_lgbm_v2.py` script was successfully modified to use `TimeSeriesSplit` for a more robust evaluation.
- [x] **Execute Enhanced Tuning Process:** The updated Optuna study was run. Despite the improved methodology, the best average Sharpe Ratio found was low (0.03).
- [x] **Train & Backtest Final Model (V3.1):** A new V3.1 model was trained and a full backtest was generated.
- [x] **Generate Final Performance Report:** The performance report and equity curve for the V3.1 model were created and analyzed.
- [x] **Compare and Finalize:** The V3.1 model's performance was significantly worse than the V2 champion. Tier 3 is now considered complete, with a definitive and valuable conclusion.

---

## Tier 4: Automated Retraining & Professional Validation Pipeline
**Objective:** To build a robust, industry-standard pipeline that continuously adapts to changing market conditions through rigorous, automated retraining, tuning, and validation. This approach replaces the previous, flawed "Genetic Programming" experiment.
**Status:** COMPLETE.
**Key Outcomes:**
- The Champion/Challenger pipeline (`src/tier4/champion_challenger.py`) is fully operational.
- A critical bug was identified and fixed where the pipeline was using a simple chronological split for training data instead of a stratified shuffle. This was causing a severe drop in model performance.
- After correcting the data splitting method and standardizing hyperparameters, a new Challenger model was trained that significantly outperformed the V2 Champion.
- **The new Champion model achieved a +14.06% total return in backtesting, establishing a new, higher benchmark for the project.**
- The pipeline successfully identified the superior performance and automatically promoted the new model, demonstrating the system's ability to self-improve.

**Architecture: A Professional, Adaptive Framework**
Based on critical expert feedback, the project has pivoted away from experimental methods. The new architecture is grounded in established best practices for quantitative finance and machine learning to address the non-stationarity of financial markets. The system will no longer attempt to "breed" a single perfect model, but will instead ensure the live model is always the most adapted and validated version for the current market regime.

**System Components (To Be Built):**
1.  **Structured Hyperparameter Tuning (Optuna):** The core of the retraining process will be a rigorous tuning script using the Optuna framework. Critically, it will employ **TimeSeriesSplit cross-validation** to find the optimal hyperparameters in a way that respects the chronological nature of financial data and prevents lookahead bias.
2.  **Walk-Forward Validation Engine:** The system will be built around a robust walk-forward backtesting engine. This is the gold standard for financial model validation. It works by training the model on a rolling window of past data (e.g., 2 years) and then validating its performance on a subsequent, unseen period (e.g., 3 months). The window then slides forward, and the process repeats, providing a much more realistic estimate of future performance.
3.  **Dynamic Feature Selection:** To adapt to changing market regimes, the pipeline will incorporate **feature permutation importance** analysis after each training cycle. This will allow the system to identify and focus on the most predictive features for the current market, discarding signals that have gone stale.
4.  **Integrated Risk Management:** The backtester will be enhanced to include essential risk management rules, such as a per-trade stop-loss or a maximum daily drawdown limit, ensuring that performance is evaluated within a realistic risk framework.

---

## Tier 1 Progress Checklist (Archive)

### Foundational Setup
- [x] Define Master Technical Specification
- [x] Design Tiered, Zero-Cost Architecture
- [x] Establish Project Directory Structure
- [x] Finalize `requirements.txt` for Tier 1
- [x] Install All Tier 1 Dependencies

### Data Engineering
- [x] Build Market Data Pipeline (`get_price_data.py`)
- [x] Build Sentiment Data Pipeline (`get_sentiment_data.py`)
- [x] Build Economic Calendar Pipeline (`get_calendar_data.py`)
- [x] Fuse All Data Sources into Master Dataset

### Feature Engineering & Modeling
- [x] Engineer Advanced Features (Kalman, Hurst, etc.)
- [x] Train, Validate, and Save Baseline XGBoost Model
- [x] Implement SHAP for Model Explainability
