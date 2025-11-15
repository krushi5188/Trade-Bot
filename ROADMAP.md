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

## Tier 3: Hyperparameter Tuning & Optimization
**Objective:** To systematically optimize the profitable V2 LightGBM model to further enhance its performance and robustness.
**Status:** PENDING.
**Methodology:**
We will use a sophisticated Bayesian optimization library, such as Optuna, to efficiently search the vast hyperparameter space. The primary objective will be to maximize the Sharpe Ratio, as this metric provides the best measure of risk-adjusted returns. The `src/tier2/tune_hyperparameters.py` script will be adapted for this purpose.

**Tasks:**
- [ ] **Define Search Space:** Identify the key hyperparameters for the LightGBM model (e.g., `num_leaves`, `learning_rate`, `n_estimators`, `reg_alpha`, `reg_lambda`, `colsample_bytree`) and define a sensible range for each.
- [ ] **Implement Optuna Study:** Configure an Optuna study to run the search. The objective function will train a LightGBM model with a given set of hyperparameters and evaluate it using our backtesting engine, returning the Sharpe Ratio.
- [ ] **Execute Tuning Process:** Run the optimization study for a significant number of trials to allow the algorithm to converge on the best parameter set.
- [ ] **Train & Backtest Final Model:** Train a new model using the optimal hyperparameters found by Optuna.
- [ ] **Generate Final Performance Report:** Run a full backtest on the optimized model and generate a final report comparing its performance against the baseline V2 model.

---

## Tier 4: Continuous Learning & Automated Deployment
**Objective:** To create a fully autonomous system where the AI can retrain, evaluate, and deploy itself continuously, allowing it to adapt to changing market conditions without manual intervention.
**Status:** PENDING.
**Architecture: Champion/Challenger System**
The core of this tier will be a "Champion/Challenger" model. The currently live "Champion" model will be periodically challenged by a newly trained "Challenger." The challenger will only be promoted to the new champion if it demonstrates superior performance on the most recent data.

**System Components:**
1.  **Automated Data Pipeline Scheduler:** A cron job or other scheduling service (e.g., using `APScheduler` in Python) will run the data ingestion scripts from Tier 1 on a regular basis (e.g., daily) to ensure our dataset is always up-to-date.
2.  **Retraining Trigger:** A trigger will initiate the training of a new "Challenger" model. This can be based on a fixed schedule (e.g., the first of every month) or be event-driven (e.g., a significant dip in the current champion's performance).
3.  **Automated Training & Backtesting Pipeline:** A master script will orchestrate the entire end-to-end process:
    *   Execute the V2 feature engineering pipeline on the latest dataset.
    *   Train a new LightGBM "Challenger" model using the optimal hyperparameters defined in Tier 3.
    *   Run a full backtest on the "Challenger" model on a recent, held-out portion of the data.
4.  **Champion/Challenger Evaluation & Promotion Logic:**
    *   The system will programmatically compare the challenger's backtest report (focusing on Sharpe Ratio and Total Return) against the champion's last-known performance.
    *   If the challenger's performance is statistically superior, the system will automatically promote it to the new "Champion."
5.  **Automated Deployment:**
    *   Upon promotion, the system will automatically replace the `lgbm_v2_model.txt` file with the new champion model file. This ensures the live trading system (or paper trading system) always uses the best-performing model.
    *   The system will log the entire process, including the performance of both models and the reason for the promotion decision.

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
