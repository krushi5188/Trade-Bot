# Master Technical Specification: The Adaptive, Multi-Source Trading Agent

## 1. Project Status & Progress Checklist

### Foundational Setup
- [ ] Define Master Technical Specification
- [ ] Design Tiered, Zero-Cost Architecture
- [ ] Establish Project Directory Structure
- [ ] Finalize `requirements.txt` for Tier 1
- [ ] Install All Tier 1 Dependencies

### Execution Architecture
- [ ] Design Brain & Hands Model (Python + MQL5)
- [ ] Design ZeroMQ Communication Bridge
- [ ] Create Python "Brain" PoC Script
- [ ] Create MQL5 "Hands" PoC Script

### Tier 1: Data Engineering
- [ ] Build Market Data Pipeline (`get_price_data.py`)
- [ ] Build Sentiment Data Pipeline (`get_sentiment_data.py`)
- [ ] Build Economic Calendar Pipeline (`get_calendar_data.py`)
- [ ] Fuse All Data Sources into Master Dataset

### Tier 1: Feature Engineering & Modeling
- [ ] Engineer Advanced Features (Kalman, Hurst, etc.)
- [ ] Train, Validate, and Save Baseline XGBoost Model
- [ ] Implement SHAP for Model Explainability
