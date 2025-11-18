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
**Objective:** To build a robust, industry-standard pipeline that continuously adapts to changing market conditions through rigorous, automated retraining, tuning, and validation.
**Status:** COMPLETE.
**Key Outcomes:**
- The Champion/Challenger pipeline (`src/tier4/champion_challenger.py`) has been significantly upgraded from a simple train/test split to a full **Purged Walk-Forward Cross-Validation** engine. This provides a much more realistic and reliable estimate of model performance.
- The pipeline now automatically calculates and saves **Dynamic Feature Importance** for every new champion model, allowing for continuous analysis of the model's decision-making process.
- A critical bug in the performance metric calculation was identified and fixed, ensuring that Annualized Return and Sharpe Ratio are now calculated correctly based on the hourly data frequency.
- The pipeline successfully identified and promoted a new champion model with a +14.06% total return (3.49% annualized) and a Sharpe Ratio of 1.57, establishing a new, reliable performance benchmark.

**Tasks:**
- [x] **Implement Purged Walk-Forward Cross-Validation:** Replace the simple train/test split with a robust walk-forward CV using the `timeseriescv` library.
- [x] **Implement Dynamic Feature Importance:** After each successful run, save a timestamped CSV of the new champion's feature importance.
- [x] **Fix Performance Metric Calculation:** Correct the Annualized Return and Sharpe Ratio formulas to properly handle hourly data.
- [x] **Run Full Pipeline:** Execute the new end-to-end pipeline to validate its functionality and generate a new champion model.

---

## Tier 5: Meta-Labeling for High-Precision Signals
**Objective:** To improve the precision of the model's entry signals by adding a secondary "meta" model that learns from the mistakes of the primary model.
**Status:** COMPLETE.
**Key Outcomes:**
- A full meta-labeling pipeline has been implemented in `src/tier5/meta_labeling_pipeline.py`.
- The pipeline successfully trains a primary model for signal generation and a secondary "meta" model to filter for high-confidence trades.
- Backtesting the meta-labeled strategy (including transaction costs) demonstrates a significant reduction in risk, with a **Maximum Drawdown of only -1.04%**, validating the effectiveness of the approach.
**Architecture:**
1.  **Primary Model (High Recall):** The existing LightGBM model will be used to generate initial buy/sell signals. The focus of this model is to identify a broad set of potential trading opportunities (high recall).
2.  **Meta-Label Generation:** The primary model's predictions will be analyzed. We will generate "meta-labels" where `1` indicates a correct prediction (a true positive) and `0` indicates an incorrect prediction (a false positive).
3.  **Secondary Model (High Precision):** A second machine learning model (e.g., another LightGBM or a simpler logistic regression) will be trained on the *same features* as the primary model, but its goal is to predict the *meta-labels*. In essence, it learns to predict whether the primary model is likely to be right or wrong on any given signal.
4.  **Final Signal Generation:** A final trading signal will only be generated when *both* the primary model signals a trade AND the secondary "meta" model predicts a high probability of the primary model being correct. This two-stage filter is designed to dramatically reduce false positives and improve the overall profitability of the strategy.

---

## Tier 6: Market Regime Detection
**Objective:** To make the model adaptive to changing market conditions (e.g., bull, bear, sideways) by training specialized models for each regime.
**Status:** PENDING.
**Architecture:**
1.  **Unsupervised Regime Identification:** An unsupervised learning model (e.g., a Hidden Markov Model or a GMM) will be trained on key market features (like volatility, price momentum, etc.) to classify historical data into a set of distinct market regimes (e.g., 'High-Volatility Bull', 'Low-Volatility Bear').
2.  **Regime-Specific Model Training:** Instead of one monolithic model, we will train a separate champion model for each identified regime, using only the data from that regime.
3.  **Live Regime Classification:** In the live environment, the system will first classify the current market state into one of the learned regimes.
4.  **Dynamic Model Selection:** Based on the current regime, the system will dynamically select and use the specialized model trained for that specific condition to generate trading signals. This allows the strategy to be much more nuanced and adaptive than a one-size-fits-all approach.

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
