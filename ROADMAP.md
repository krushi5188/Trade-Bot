# Project Charter & Roadmap: The Adaptive, Multi-Source Trading Agent

## I. Core Philosophy

The objective is to build a trading system that transcends simple technical analysis. Standard trading bots fail because they operate on a single, incomplete data stream (price) and use static models that cannot adapt to changing market conditions.

**Our system's superiority will be derived from three core principles:**

1.  **Data Fusion:** The system will make decisions based on a unified understanding of market price action, human sentiment (news & social media), and the overarching macroeconomic environment. It will have a more complete picture of reality than any system relying on charts alone.
2.  **Explainability (XAI):** The system will be built to explain the reasoning behind its decisions. Every trade, right or wrong, becomes a learning opportunity because we can pinpoint the exact factors (e.g., "positive news sentiment," "oversold RSI") that led to the action. This eliminates the "black box" problem and enables intelligent, targeted improvements.
3.  **Adaptability (Reinforcement Learning):** The system will not be a static model; it will be a true learning agent. By using a Reinforcement Learning (RL) framework, it will learn from its own mistakes in a simulated environment, optimizing its strategy for profitability. An automated retraining pipeline will ensure it continuously adapts to new market regimes.

---

## II. The Detailed Execution Plan

### **Phase 1: The Data Fusion Engine (The Sensory System)**

*   **Objective:** To build an automated pipeline that gathers all necessary data and fuses it into a single, time-synchronized dataset. This is the foundation upon which all intelligence is built.

*   **Component 1.1: Core Price Data**
    *   **What:** Hourly OHLCV (Open, High, Low, Close, Volume) data for a primary trading pair (e.g., BTC/USDT).
    *   **How:** A Python script will be written using the `ccxt` library to connect to the Binance API. `ccxt` is the industry standard for exchange connectivity, providing a unified interface to over 100 exchanges. The data will be requested in batches, with error handling for API rate limits, and stored in the highly efficient Parquet format.
    *   **Why It's Superior:** We avoid unreliable CSV files and use a professional-grade library (`ccxt`) and storage format (`Parquet`) from day one. This ensures data integrity and high-performance reads, which are critical for backtesting and model training.

*   **Component 1.2: Human Sentiment Data**
    *   **What:** Real-time sentiment analysis of financial news.
    *   **How:** A Python script using `requests` and `BeautifulSoup4` will scrape the RSS feed of Reuters' business section. Each headline will be fed into the `ProsusAI/finbert` model, a state-of-the-art NLP model specifically pre-trained on trillions of words of financial text. It will output a sentiment score (e.g., -0.8 to +0.9).
    *   **Why It's Superior:** Most sentiment analysis tools use generic models that don't understand financial context. FinBERT understands the nuance of headlines like "Fed Signals Hawkish Stance," providing a far more accurate and actionable sentiment signal than any off-the-shelf solution.

*   **Component 1.3: Macroeconomic Context**
    *   **What:** Key economic indicators, primarily the US Federal Funds Rate and the Consumer Price Index (Inflation).
    *   **How:** A script using the `pandas_datareader` library will connect directly to the Federal Reserve's FRED database API.
    *   **Why It's Superior:** The AI will not be blind to the macroeconomic environment. Its decisions will be informed by the same data that moves institutional capital. When interest rates change, our AI will know about it and will learn the impact of that change on the market.

*   **Component 1.4: Data Fusion & Synchronization**
    *   **What:** A master script to combine all the above data streams into a single, hourly-indexed master DataFrame.
    *   **How:** The price data will serve as the master index. Sentiment scores will be averaged within each hour. Macroeconomic data, being low-frequency, will be forward-filled (the last known value will apply to subsequent hours).
    *   **Why It's Superior:** This painstaking synchronization process is what creates the "fused" reality for our AI. It ensures that at any given point in time, the AI's "state" includes not just the price, but the prevailing sentiment and economic conditions, preventing lookahead bias and ensuring causal integrity.

---

### **Phase 2: The Explainable AI Core (The Conscious Mind)**

*   **Objective:** To train a powerful predictive model that can explain its reasoning, turning every mistake into a clear lesson.

*   **Component 2.1: Advanced Feature Engineering**
    *   **What:** Transforming the fused raw data into intelligent features.
    *   **How:** We will compute a suite of features:
        *   **Technical Indicators:** RSI, MACD, Bollinger Bands, Average True Range (ATR).
        *   **Sentiment Features:** 24-hour and 7-day rolling averages of the sentiment score to capture trends in market mood.
        *   **Macro Features:** Direct inclusion of the interest rate and inflation data.
        *   **Target Label:** A sophisticated "Triple Barrier Method" label will be used. A trade will be labeled `1` (buy) only if the price hits a 2% profit target before it hits a 1% stop-loss target within the next 24 hours. This builds risk management directly into the model's objective.
    *   **Why It's Superior:** We are not just predicting "up or down." We are training the AI on a realistic trading scenario with a defined risk-reward ratio, leading to a much more practical and robust model.

*   **Component 2.2: The Predictive Model**
    *   **What:** Training a model to predict the outcome of our target label.
    *   **How:** We will use the `XGBoost` library. It is a form of Gradient Boosted Decision Tree that is consistently the top performer for the kind of structured, tabular data we are creating. The data will be split chronologically (e.g., 2018-2022 for training, 2023 for testing) to prevent any leakage of future information.
    *   **Why It's Superior:** Unlike a deep learning neural network which is a "black box," XGBoost models are inherently more interpretable. This directly enables the next, most critical step: explainability.

*   **Component 2.3: The Explainability Interface**
    *   **What:** A system to query any trade and understand its rationale.
    *   **How:** We will integrate the `SHAP` (SHapley Additive exPlanations) library. After the model is trained, we will build a function that takes a timestamp as input and outputs a "force plot." This plot will visually decompose the prediction, showing, for example: `Prediction: BUY. Driven by: Low RSI (+0.3), High News Sentiment (+0.25), Overruling: High Inflation (-0.1)`.
    *   **Why It's Superior:** This is our "flight data recorder." When the AI makes a wrong trade, we don't have to guess why. We can see *exactly* what it was thinking. This allows us to intelligently improve the system by adding new data sources or engineering better features to correct for its blind spots.

---

### **Phase 3: The Adaptive Learning Loop (The Subconscious)**

*   **Objective:** To evolve the system from a static predictor into a dynamic agent that learns and refines its *strategy* through continuous practice.

*   **Component 3.1: The Market Simulator**
    *   **What:** A virtual trading environment for the AI to practice in.
    *   **How:** We will build a custom environment using the `gymnasium` library (the successor to OpenAI Gym). This environment will be fed our entire historical dataset. It will include a realistic simulation of transaction fees (e.g., 0.1% per trade).
    *   **Why It's Superior:** The agent will be trained in an environment that punishes over-trading and rewards efficient, profitable actions. It will learn not just *when* to trade, but also when *not* to trade.

*   **Component 3.2: The Reinforcement Learning Agent**
    *   **What:** The AI agent that will learn to trade through trial and error.
    *   **How:** We will use the `PPO` (Proximal Policy Optimization) algorithm from the `stable-baselines3` library. The agent's goal is simple: maximize the portfolio value. It will be rewarded for profitable trades and punished for losing trades. It will run millions of simulated trades, gradually discovering a complex strategy that a human might never find.
    *   **Why It's Superior:** This directly addresses the core requirement for an AI that "learns from its own mistakes." It is not just learning patterns from data; it is learning a *behavioral policy* that is robust and optimized for profit, not just prediction accuracy.

---

### **Phase 4: Live Deployment & Evolution (The Final Form)**

*   **Objective:** To deploy the trained agent and ensure it never stops learning.

*   **Component 4.1: The Live Execution Bot**
    *   **What:** A script that runs 24/7, connects to the exchange, and executes trades.
    *   **How:** A master Python script will run in a loop. Every hour, it will execute the entire data fusion pipeline to get the most recent data, feed it to the trained RL agent, receive an action (Buy, Sell, or Hold), and execute that action via the `ccxt` API on a paper trading account.
    *   **Why It's Superior:** The bot's architecture ensures it operates on the freshest possible data from all sources, making its decisions maximally relevant to the current market state.

*   **Component 4.2: The Automated Retraining Pipeline**
    *   **What:** A system that automatically updates the AI's "brain" with new information.
    *   **How:** A `cron` job (an automated task scheduler) will be configured on the server. Every Sunday, it will automatically:
        1.  Run the data collection scripts to download the past week's data.
        2.  Append this new data to our historical datasets.
        3.  Re-run the Reinforcement Learning training process, allowing the agent to fine-tune its strategy based on the newest market data.
        4.  Save the newly-trained model, which the live bot will automatically use for the following week.
    *   **Why It's Superior:** This is the capstone of the entire system. The market is not static, and neither is our AI. It is designed to evolve. A strategy that worked in a bull market may fail in a bear market. This retraining loop gives our AI the ability to adapt its behavior, ensuring its long-term viability.
