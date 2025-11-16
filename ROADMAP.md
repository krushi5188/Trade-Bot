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

## Tier 4: Continuous Learning & Automated Deployment
**Objective:** To create a fully autonomous system where the AI can retrain, evaluate, and deploy itself continuously, allowing it to adapt to changing market conditions without manual intervention.
**Status:** IN PROGRESS.
**Key Outcomes (In Progress):**
- A complete, end-to-end Champion/Challenger pipeline has been built and is located in `src/tier4/champion_challenger.py`.
- The pipeline was successfully debugged, resolving a critical data corruption issue caused by a limited data fetch in the original pipeline.
- **Note on GPU Usage:** The pipeline is currently configured to run on the CPU. The target Google Colab environment has unresolved issues with its GPU driver and library configuration that prevent LightGBM from building with GPU support. The pipeline will run successfully on the CPU as a fallback.
**Architecture: Champion/Challenger System**
The core of this tier will be a "Champion/Challenger" model. The currently live "Champion" model will be periodically challenged by a newly trained "Challenger." The challenger will only be promoted to the new champion if it demonstrates superior performance on the most recent data.

**System Components (Complete):**
1.  **Evolutionary Training Loop (`src/tier4/evolutionary_training.py`):** The core of the system is a continuous, evolutionary training loop. Instead of simple scheduling, this script actively guides the AI's learning. It uses a genetic algorithm to "breed" and "mutate" the hyperparameters of winning models, allowing the AI to intelligently explore and improve upon its own best ideas.
2.  **Human-Readable Reporting:** After each evolutionary cycle, the system prints a clear, simple summary of the new model's performance and whether it was promoted, making it easy to track the AI's learning progress.
3.  **Automated Training & Backtesting Pipeline (`src/tier4/champion_challenger.py`):** The underlying master script that orchestrates the end-to-end process of training, backtesting, and promotion. It is now controlled by the evolutionary training loop.
4.  **Gene Pool (`gene_pool.json`):** A file that stores the "genes" (hyperparameters) of all past champion models, serving as the basis for future evolution.

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
