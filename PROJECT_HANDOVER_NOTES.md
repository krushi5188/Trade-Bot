# Project Handover Notes

**From:** Jules
**Date:** 2025-11-15

## 1. High-Level Goal & Current Status

The primary goal is to develop a profitable, autonomous AI trading model.

**Current Status:** We have built the initial version of the Tier 4 "Champion/Challenger" automated retraining pipeline.
*   **Tier 1, 2, 3:** Completed.
*   **Tier 4:** In Progress. The core pipeline is functional but is currently running on the CPU due to significant environmental issues with the GPU setup in the target Colab environment.

## 2. CRITICAL: Summary of Tier 4 Debugging Session

The implementation of the Tier 4 pipeline was a multi-step debugging process. Understanding this process is critical for the next assistant.

1.  **Initial Bug (Flat Equity Curve):** The first run of the pipeline produced a flat equity curve. Analysis showed the model was only predicting "hold" signals.
2.  **Root Cause Analysis (Data Corruption):** The investigation traced the problem to a critical data corruption issue. The `btc_close` price data was flat for most of its history. This was caused by the `get_price_data.py` script only fetching 2 years of crypto data, while the `fuse_data.py` script performed an outer merge with longer-history forex/metals data, causing the missing BTC prices to be backfilled with a single, flat value.
3.  **Data Pipeline Fix:** The `get_price_data.py` script was corrected to fetch 15 years of crypto data, ensuring a complete and valid price history.
4.  **Second Bug (GPU Environment Failure):** After fixing the data, the pipeline failed again, this time with a `lightgbm.basic.LightGBMError: No OpenCL device found`. This indicates a fundamental problem with the GPU driver/library setup in the execution environment.
5.  **Extensive GPU Debugging (Blocked):** Multiple attempts were made to fix the GPU environment, including:
    *   Building LightGBM from source with OpenCL flags (failed, missing headers).
    *   Building LightGBM from source with CUDA flags (failed, `nvcc` not found).
    *   Installing via `conda` (failed, `conda` not available).
    *   Installing OpenCL runtime libraries (failed, no `sudo` permissions).
    **Conclusion:** The GPU environment is misconfigured, and I have exhausted all available methods to fix it.
6.  **Pragmatic Solution (CPU Fallback):** To deliver a working pipeline, the script was modified to use the CPU for training. This is a stable workaround.

## 3. Immediate Next Steps for Next Assistant

*   The `Final-Training` branch contains the most up-to-date, working (CPU-based) version of the Tier 4 pipeline.
*   **DO NOT** attempt to re-enable the GPU unless you have the ability to fix the underlying driver and library issues in the execution environment.
*   The next logical step is to continue with the Tier 4 roadmap, focusing on the scheduling and automation aspects of the Champion/Challenger system.
