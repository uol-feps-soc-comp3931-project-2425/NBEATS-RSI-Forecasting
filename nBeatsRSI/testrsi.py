from darts import TimeSeries
import numpy as np
from darts.models import NBEATSModel
import yfinance as yf
import pandas_ta as ta

start_date = '2024-01-01'
end_date = '2025-01-01'
sp500 = yf.Ticker("SPY").history(start=start_date, end=end_date)
data = sp500.drop(columns=['Dividends', 'Stock Splits','Capital Gains'])
print(data.tail())

rsi = data.ta.rsi(close='Close', length=14, append=True)
rsi = rsi.dropna()
print(rsi.tail())
rsi_norm = rsi/100


series = TimeSeries.from_values(np.float32(rsi_norm))
train, val = series[:100], series[100:]
model = NBEATSModel(input_chunk_length=14, output_chunk_length=5)
model.fit(train)
pred = model.predict(n=len(val),series=train)
