# Master Technical Specification: The Adaptive, Multi-Source Trading Agent

## 1. Project Philosophy & Core Mandates

### 1.1. Overarching Vision
This document outlines the architecture for a state-of-the-art, adaptive quantitative trading system. It is designed to achieve a sustainable edge by moving beyond simple predictive models into a holistic, multi-layered system that excels in data engineering, advanced modeling, adaptive learning, and robust risk management.

### 1.2. Core Mandate: Zero-Cost, FOSS-First
This entire system will be built using Free and Open-Source Software (FOSS). All data will be sourced from free, public APIs or collected via our own custom web scrapers. The only unavoidable cost will be a standard, low-cost Virtual Private Server (VPS) for 24/7 operation. There will be no reliance on paid, third-party services.

### 1.3. Development Philosophy: Tiered & Validated
The project is structured in distinct Tiers. Each Tier represents a significant leap in capability and must be completed and rigorously validated before the next begins. This ensures a stable foundation and prevents wasted effort.

---

## 2. Technology Stack

| Category                  | Library/Tool                               | Version       | Purpose                                                                                             |
| ------------------------- | ------------------------------------------ | ------------- | --------------------------------------------------------------------------------------------------- |
| **Core Data Science**     | `pandas`                                   | `2.1.2`       | The primary tool for data manipulation, time-series analysis, and data structuring.                |
|                           | `numpy`                                    | `1.26.1`      | Foundational library for high-performance numerical computation and vectorized operations.        |
|                           | `scikit-learn`                             | `1.3.2`       | Used for data pre-processing, splitting, and performance metrics.                                 |
| **Market Data**           | `ccxt`                                     | `4.1.3`       | The unified API for fetching OHLCV data directly from crypto exchanges.                             |
| **Sentiment Data**        | `feedparser`                               | `6.0.10`      | To reliably parse structured XML data from public news RSS feeds.                                 |
|                           | `transformers`                             | `4.34.1`      | To load and use the pre-trained FinBERT model for financial sentiment analysis.                   |
|                           | `torch`                                    | `2.3.0`       | The underlying framework for the `transformers` library.                                          |
| **Web Scraping**          | `requests`                                 | `2.31.0`      | To send HTTP requests and download HTML content from websites.                                    |
|                           | `BeautifulSoup4`                           | `4.12.2`      | To parse, navigate, and extract data from unstructured HTML.                                      |
| **Feature Engineering**   | `pykalman`                                 | `0.9.5`       | To implement Kalman Filters for price signal noise reduction.                                     |
|                           | `hurst`                                    | `0.0.5`       | To calculate the Hurst Exponent for market regime detection.                                      |
| **Tier 1 Modeling**       | `xgboost`                                  | `2.0.1`       | Our foundational, explainable model for baseline performance.                                     |
|                           | `shap`                                     | `0.43.0`      | For model explainability (XAI) to understand the "why" behind predictions.                      |
| **Tier 2+ Modeling**      | `lightgbm`                                 | (TBD)         | High-performance model to be introduced to beat the XGBoost benchmark.                            |
|                           | `pytorch`                                  | (TBD)         | For building advanced sequential models (LSTM, Transformers).                                     |
| **Execution Bridge**      | `pyzmq`                                    | `25.1.2`      | Python library for the ZeroMQ high-speed, low-latency messaging bus.                              |
|                           | `Zmq.mqh`                                  | (External)    | The corresponding MQL5 library for the ZeroMQ connection (requires manual installation in MT5). |
| **Performance Opt.**      | `numba`                                    | (TBD)         | For Just-In-Time (JIT) compilation of Python code to C-speed for latency-critical functions.  |

---

## 3. Data Architecture & Pipelines (Tier 1)

### 3.1. Market Data (Chart Data)
*   **Source:** Direct from exchange APIs (e.g., Kraken, Binance) via `ccxt`.
*   **Methodology:** The `src/tier1/data_pipelines/get_price_data.py` script implements a robust, paginated data download loop.
    1.  It instantiates the exchange with a built-in rate limiter (`'enableRateLimit': True`) to avoid being banned.
    2.  It fetches data in 1000-candle chunks, using the timestamp of the last candle to request the next chunk, until the full history is retrieved.
    3.  The entire process is wrapped in a `try...except` block with a `time.sleep()` retry mechanism to handle temporary API or network failures.
*   **Schema:** OHLCV (Open, High, Low, Close, Volume) at a 1-minute resolution.
*   **Storage:** `data/raw/market/SYMBOL_1m.parquet`

### 3.2. Sentiment Data
*   **Source:** Public RSS feeds (e.g., Reuters Business News).
*   **Methodology:** The `src/tier1/data_pipelines/get_sentiment_data.py` script will:
    1.  Use `feedparser.parse(URL)` to reliably parse the structured XML feed.
    2.  For each news item, it extracts the `title` and `published` date.
    3.  The title is fed to a pre-loaded FinBERT model via the `transformers` library.
    4.  The model's output logits are converted to a single sentiment score from -1 (max negative) to +1 (max positive).
*   **Schema:** `timestamp_utc`, `headline`, `sentiment_score`.
*   **Storage:** `data/raw/sentiment/news_sentiment.parquet`

### 3.3. Economic Calendar Data
*   **Source:** A public financial website with a structured calendar (e.g., Investing.com).
*   **Methodology:** A new script, `get_calendar_data.py`, will be built to:
    1.  Perform a `requests.get()` call with a standard User-Agent header to download the page HTML.
    2.  Use `BeautifulSoup4` to parse the HTML, specifically targeting the calendar table by its unique HTML `id` or `class` attributes.
    3.  Iterate through the rows of the table to extract the event name, time, and importance ("High", "Medium", "Low").
    4.  The script will include rate-limiting (`time.sleep()`) and robust error handling to detect if the website's HTML structure changes.
*   **Schema:** `timestamp_utc`, `event_name`, `importance`.
*   **Storage:** `data/raw/macro/economic_calendar.parquet`

---

## 4. Modeling & Execution Architecture

### 4.1. Tier 1 Model: XGBoost for Explainable Foundation
*   **Rationale:** As detailed in our discussions, XGBoost is chosen as the initial model for its robustness against overfitting on noisy, newly-engineered data and its superior interpretability with SHAP. The primary goal of Tier 1 is to validate our data and features, making explainability more important than raw speed.
*   **Training Process:**
    1.  All data streams will be fused into a single time-series DataFrame.
    2.  The data will be split **chronologically** into training and testing sets.
    3.  A `XGBClassifier` will be trained on the training set.
    4.  A `shap.TreeExplainer` will be used on the trained model to analyze the feature importance and debug individual trade decisions.

### 4.2. Execution Architecture: The Brain & Hands Model
This architecture is designed for maximum performance and stability by separating concerns.

*   **The AI Brain (Python):** A single, long-running Python process. Its only job is to:
    1.  Ingest live data (via WebSocket in later tiers).
    2.  Calculate all features and run the AI model to produce a signal.
    3.  Publish this signal to a ZeroMQ socket.

*   **The EA "Hands" (MQL5):** A minimal Expert Advisor on the MT5 terminal. Its only job is to:
    1.  Subscribe to the ZeroMQ socket from the Python Brain.
    2.  Listen for JSON messages in a non-blocking `OnTimer()` loop.
    3.  Upon receiving a valid JSON signal, immediately execute the trade using `OrderSend()`.

### 4.3. The Communication Bridge: ZeroMQ
*   **Pattern:** `PUB-SUB` (Publish-Subscribe). The Python Brain is the `PUB`, the MQL5 EA is the `SUB`.
*   **Endpoint:** `tcp://127.0.0.1:5555`. This is a local, in-memory communication bus, not a network hop. This is the key to microsecond-level internal communication.
*   **Signal Format:** A standardized JSON string with required fields: `action`, `symbol`, `entry_price`, `stop_loss`, `take_profit`, `trade_id`.

---

## 5. Latency Mitigation & Verification

Latency will be managed and verified through a rigorous, three-pillar engineering discipline.

### 5.1. Design
*   **Infrastructure:** The system will be deployed on a single, low-cost VPS strategically chosen to be in a data center physically close to the broker's servers.
*   **Architecture:** The "Brain & Hands" model with ZeroMQ IPC is inherently low-latency.

### 5.2. Measurement
*   **High-Resolution Timestamps:** All key stages of the decision-making and execution pipeline will be timestamped using `time.time_ns()` in Python and the equivalent high-resolution timer in MQL5.
*   **Latency Logging:** A dedicated `latency.log` file will be generated, recording the time taken between each stage (Data In -> Prediction Out -> Signal Sent -> Signal Received -> Order Sent). This will allow us to measure and enforce a strict latency budget.

### 5.3. Optimization
*   **Code:** Python code for feature engineering will be optimized using `NumPy` vectorization and, where necessary, `Numba` JIT compilation.
*   **Model:** In later tiers, trained models will be converted to the high-performance **ONNX** format for faster inference.
*   **Data:** In later tiers, data ingestion will move from REST APIs to persistent **WebSocket** connections for the lowest possible data-reception latency.
