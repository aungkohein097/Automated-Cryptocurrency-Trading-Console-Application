# 📌 Automated Crypto Strategy Backtester  
*A Python console application for evaluating predefined crypto trading strategies.*

---

## 📖 Overview  
This project is an automated **cryptocurrency backtesting system** built with Python.  
It retrieves historical OHLCV data from Binance and tests multiple technical-indicator trading strategies to measure profitability and risk.

---

## 🧠 Features  

- Fetch historical data from Binance via **CCXT**
- Multiple indicator-based strategies:
  - RSI
  - SMA
  - EMA Crossover
  - MACD
  - Bollinger Band breakout
  - RSI + Pivot Pattern
- Combination strategy using **majority vote**
- Detailed performance report:
  - Final balance & net return
  - Win rate
  - Max drawdown
  - Profit factor
  - # of trades
  - Average P/L per trade
- Equity curve visualization (auto-save as PNG)
- Logging to CSV + TXT
- JSON settings for RSI optimization

---

## 🧩 Strategy Table

| Strategy | Indicators | Signal Concept |
|---------|------------|----------------|
| RSI | RSI + ATR filter | High probability bounce / rejection points |
| SMA | 20-period SMA | Trend-following cross + slope confirmation |
| EMA Cross | EMA 9 & EMA 21 | Fast MA crossing slow MA |
| MACD | MACD & Signal | Cross confirmed by EMA trend direction |
| Bollinger | BB + EMA | Breakout + candle confirmation |
| RSI + Pivot | Pivot recognition + RSI | Detects reversal pivot points |
| Strategy Combination | 3-indicator vote | Must have 2+ agreement |

---

## 🛠 Tech Stack

| Category | Library |
|---------|---------|
| Exchange API | `ccxt` |
| Indicators | `ta` |
| Data | `pandas` |
| Plotting | `matplotlib` |

---

## ▶ Usage Example

```python
from backtest_bot import TradingBot

bot = TradingBot()
bot.backtest_strategy("BTC/USDT", "RSI", 2023)
