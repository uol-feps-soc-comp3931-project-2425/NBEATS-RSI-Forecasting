import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from nbeats_pytorch.model import NBeatsNet
import torch
import matplotlib.pyplot as plt
from darts import TimeSeries

def fetch_data(ticker: str = "SPY", start_date: str = "2020-01-01", end_date: str = "2024-12-31"):
    # Fetch data
    data = yf.Ticker("SPY").history(start=start_date, end=end_date)
    print(f"Downloaded data shape: {data.shape}")
    print(f"Downloaded data columns: {data.columns}")
    print(data.head())
    
    # Calculate RSI and normalize
    data['RSI'] = ta.rsi(data["Close"], length=14)  # Explicitly set RSI period
    data['RSI'] = data['RSI'].dropna()
    data['RSI_norm'] = norm_rsi(data['RSI'])
    
    print(f"Normalized RSI calculation result:\n{data['RSI_norm'].tail()}")
    print(f"Number of non-NA RSI values: {data['RSI_norm'].notna().sum()}")
    
    return data

def prepare_data(data: pd.Series, input_size: int, forecast_size: int):
    # Ensure we have enough data points
    if len(data) < input_size + forecast_size:
        raise ValueError(f"Not enough data points. Need at least {input_size + forecast_size}, got {len(data)}")
    
    # Convert to numpy array and ensure float type
    data_array = data.values.astype(np.float32)
    print(f"Data array shape: {data_array.shape}")
    print(f"Data array sample: {data_array[:5]}")
    
    X, y = [], []
    for i in range(len(data_array) - input_size - forecast_size + 1):
        X.append(data_array[i:i+input_size])
        y.append(data_array[i+input_size:i+input_size+forecast_size])
    
    print(f"Number of samples prepared: {len(X)}")
    
    # Convert to tensors and add channel dimension
    X = torch.FloatTensor(np.array(X)).unsqueeze(-1)  # Shape: (samples, sequence_length, 1)
    y = torch.FloatTensor(np.array(y)).unsqueeze(-1)  # Shape: (samples, forecast_length, 1)
    
    print(f"Final tensor shapes - X: {X.shape}, y: {y.shape}")
    return X, y

def main():
    # Parameters
    input_size = 30  # Number of days to look back
    forecast_size = 7  # Number of days to forecast
    batch_size = 32
    epochs = 100
    
    # Fetch data
    data = fetch_data()
    rsi_data = data['RSI_norm']
    rsi_data = rsi_data.dropna()
    

    if len(rsi_data) < input_size + forecast_size:
        raise ValueError(f"Not enough RSI data points after cleaning. Need at least {input_size + forecast_size}, got {len(rsi_data)}")
    
    # Prepare data for training
    X, y = prepare_data(rsi_data, input_size, forecast_size)
    
    # Create N-BEATS model
    model = NBeatsNet(
        stack_types=['seasonal', 'generic'],
        nb_blocks_per_stack=2,
        forecast_length=forecast_size,
        backcast_length=input_size,
        hidden_layer_units=128,
        thetas_dim=(4, 8, 4)
    )
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()
    
    # Training loop
    losses = []
    for epoch in range(epochs):
        permutation = torch.randperm(X.size(0))
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, X.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_X, batch_y = X[indices], y[indices]
            
            optimizer.zero_grad()
            backcast, forecast = model(batch_X)
            # Ensure forecast and batch_y have the same shape
            forecast = forecast.squeeze(-1)  # Remove the last dimension if it exists
            batch_y = batch_y.squeeze(-1)    # Remove the last dimension if it exists
            loss = criterion(forecast, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.savefig('training_loss.png')
    plt.close()
    
    # Make predictions on the last window
    model.eval()
    with torch.no_grad():
        last_window = rsi_data[-input_size:].values.astype(np.float32)
        print(f"Last window shape before tensor: {last_window.shape}")
        last_window_tensor = torch.FloatTensor(last_window).unsqueeze(0).unsqueeze(-1)
        print(f"Prediction input shape: {last_window_tensor.shape}")
        _, forecast = model(last_window_tensor)
        print(f"Forecast shape: {forecast.shape}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(rsi_data)), rsi_data, label='Actual RSI')
    plt.plot(range(len(rsi_data), len(rsi_data) + forecast_size), 
             forecast.squeeze().numpy(), 
             label='Predicted RSI')
    plt.axvline(x=len(rsi_data), color='r', linestyle='--')
    plt.title('RSI Forecast')
    plt.xlabel('Days')
    plt.ylabel('RSI')
    plt.legend()
    plt.savefig('rsi_forecast.png')
    plt.close()

if __name__ == "__main__":
    main() 