# Deep Learning Time-Series Forecasting: Extrapolating RSI with N-BEATS

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Dissertation-BSc%20CS%20%26%20Maths-green.svg)](https://github.com/MisguidedTrooper/NBEATS-RSI-Forecasting)

> **BSc Final Year Dissertation Project**  
> **Title:** Evaluating the Effectiveness of Applying Machine Learning to Extrapolate Technical Indicators for Stock Trading Algorithms  
> **Author:** Parvesh — BSc Computer Science with Mathematics (Joint Honours), University of Leeds

---

## Project Overview

Quantitative trading algorithms traditionally evaluate technical indicators such as the **Relative Strength Index (RSI)** using historical price action to signal overbought or oversold market regimes. However, reactive indicators lag during rapid regime shifts.

This research investigates whether deep neural time-series architectures—specifically **N-BEATS (Neural Basis Expansion Analysis for Interpretable Time Series Forecasting)**—can accurately extrapolate future RSI indicator trajectories on US equities and benchmark ETFs, providing predictive forward bounds for algorithmic execution and multi-stage trading pipelines.

### High-Level Architecture Pipeline

1. **Market Ingestion:** Historical US Equities / ETF OHLCV data streaming via `yFinance` (SPY, AAPL, MSFT, TSLA).
2. **Feature Engineering:** Calculation of 14-day Wilder's Smoothed RSI indicator sequences.
3. **N-BEATS Deep Architecture:** PyTorch multi-stack implementation utilizing doubly-residual stacking (Backcast & Forecast).
4. **Validation & Backtesting:** Rolling multi-horizon forecast evaluation against ground-truth out-of-sample regimes.

---

## Core Methodology

* **Technical Indicator Formulation:**
  * Evaluates sequential windows of 14-day Wilder's Relative Strength Index calculated from adjusted closing price data:
    * `RSI = 100 - (100 / (1 + RS))`
    * `RS = (Smoothed Average Gain) / (Smoothed Average Loss)`
* **N-BEATS Model Architecture:**
  * Pure deep learning time-series architecture operating directly on sequential windows without recurrence (no RNN/LSTM) or attention mechanisms.
  * Hierarchical doubly-residual stacking where each block predicts a **backcast** (reconstructing input to remove explained variance) and a **forecast** (projecting future steps).
* **Evaluation Framework:**
  * Backtested on out-of-sample market price series to assess rolling forecast fidelity, cycle tracking, and mean-reversion detection.

---

## Key Findings & Quantitative Results

| Metric / Attribute | Training Set Performance | Out-of-Sample Test Set Performance |
|---|---|---|
| **Explained Variance (R²)** | **99.1%** | **57.4%** |
| **Mean Absolute Error (MAE)** | **~0.8 RSI Points** | **5.6 RSI Points** |
| **Trend & Cycle Capture** | Near-perfect alignment on rolling cyclic trends | Captures primary macro inflections; lags on high-impact news shocks |

### Primary Conclusions
* **Macro Regime Tracking:** N-BEATS demonstrated strong general trend-tracking capabilities, successfully anticipating standard mean-reversion curves and momentum build-ups.
* **Exogenous Shock Sensitivity:** Model degradation on unseen data primarily occurred during high-volatility spike regimes driven by real-world news and fundamental macro announcements.
* **Algorithmic Utility:** While purely autoregressive technical extrapolation is insufficient as a standalone execution signal, an average error of **5.6 RSI points** makes the model an effective input feature when combined with fundamental indicators in hybrid multi-modal trading strategies.

---

## Visualisations & Model Outputs

### 1. Multi-Step Rolling Forecast vs. Ground Truth
![Rolling Forecast](nBeatsRSI/rolling_forecasts.png)

### 2. Smoothed Trajectory Tracking
![Smoothed Trajectory](nBeatsRSI/smooth_forecast_vs_actual.png)

### 3. Training Loss & Convergence
![Training Loss](nBeatsRSI/training_loss.png)

---

## Tech Stack & Tools

* **Languages & Deep Learning Frameworks:** Python 3.12, PyTorch, PyTorch Lightning, TensorFlow / Keras, Darts, `nbeats-pytorch`
* **Time-Series & Numerical Computing:** NumPy, Pandas, SciPy, Statsmodels, `pandas-ta`
* **Financial Data Ingestion:** `yFinance`
* **Visualisation:** Matplotlib, Seaborn, Plotly, HoloViews

---

## Getting Started & Environment Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/MisguidedTrooper/NBEATS-RSI-Forecasting.git](https://github.com/MisguidedTrooper/NBEATS-RSI-Forecasting.git)
cd NBEATS-RSI-Forecasting
```

### 2. Set Up Conda Environment
Create the quant environment from the provided YAML specification:

```bash
conda env create --file quant-environment.yml
```
### 3. Activate and Install Dependency Packages
Activate the environment and install the required forecasting libraries:

```bash
conda activate quant
pip install nbeats-pytorch tensorflow yfinance darts --upgrade-strategy eager
```
### 4. Running the Model (Train vs. Test)
The pipeline is executed by running one of the four asset-specific scripts (train_rsi_SPY.py, train_rsi_AAPL.py, train_rsi_MSFT.py, or train_rsi_TSLA.py).

Execution behavior is controlled by modifying the train_model boolean flag at the bottom of your chosen script.

To Train the Model:
Open the script and ensure the main function is called with True:

```python
if __name__ == "__main__":
    main(train_model=True)
```
Then run the script from your terminal:

```bash
python nBeatsRSI/train_rsi_SPY.py
```
To Test / Evaluate the Model:
Open the script and change the flag to False. This skips the training loop, loads the saved .pth weights, and runs the out-of-sample testing and visualisation:

```python
if __name__ == "__main__":
    main(train_model=False)
```
Then run the script from your terminal:

```bash
python nBeatsRSI/train_rsi_SPY.py
```
