import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from nbeats_pytorch.model import NBeatsNet
import torch
import matplotlib.pyplot as plt
from darts import TimeSeries
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def fetch_data(ticker: str = "TSLA", start_date: str = "2019-06-01", end_date: str = "2024-06-01"):
    # Fetch data
    data = yf.Ticker(ticker).history(start=start_date, end=end_date)
    print(f"Downloaded data shape: {data.shape}")
    print(f"Downloaded data columns: {data.columns}")
    print(data.head())
    
    # Calculate RSI and normalize
    data['RSI'] = ta.rsi(data["Close"], length=14)  # set RSI period
    data['RSI'] = data['RSI'].dropna()
    data['RSI_norm'] = (data['RSI'].dropna())/100
    
    print(f"Normalized RSI calculation result:\n{data['RSI_norm'].tail()}")
    print(f"Number of non-NA RSI values: {data['RSI_norm'].notna().sum()}")
    
    return data

def prepare_data(data: pd.Series, input_size: int, forecast_size: int):
    # Ensure we have enough data points
    if len(data) < input_size + forecast_size:
        raise ValueError(f"Not enough data points. Need at least {input_size + forecast_size}, got {len(data)}")
    
    # Convert to numpy array for TimeSeries
    data_array = data.values.astype(np.float32)
    print(f"Data array shape: {data_array.shape}")
    print(f"Data array sample: {data_array[:5]}")
    
    X, y = [], []
    for i in range(len(data_array) - input_size - forecast_size + 1):
        X.append(data_array[i:i+input_size])
        y.append(data_array[i+input_size:i+input_size+forecast_size])
    
    print(f"Number of samples prepared: {len(X)}")
    
    # Convert to tensors
    X = torch.FloatTensor(np.array(X)).unsqueeze(-1) 
    y = torch.FloatTensor(np.array(y)).unsqueeze(-1)  
    
    print(f"Final tensor shapes - X: {X.shape}, y: {y.shape}")
    return X, y

def rolling_forecast_evaluation(model, data, input_size, forecast_size, steps_ahead=180):
    
    model.eval()
    all_preds = []
    all_actuals = []

    data = data.dropna().values.astype(np.float32)
    
    with torch.no_grad():
        for i in range(steps_ahead):
            start_idx = i
            end_idx = i + input_size
            
            if end_idx + forecast_size > len(data):
                break

            input_window = data[start_idx:end_idx]
            actual = data[end_idx:end_idx+forecast_size]

            input_tensor = torch.FloatTensor(input_window).unsqueeze(0).unsqueeze(-1)
            _, forecast = model(input_tensor)
            forecast = forecast.squeeze().numpy()

            all_preds.append(forecast)
            all_actuals.append(actual)

    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)
    
    return all_preds, all_actuals

def plot_rolling_forecasts(preds, actuals, forecast_size):
    plt.figure(figsize=(14, 6))
    for i in range(len(preds)):
        offset = i + forecast_size
        plt.plot(range(offset, offset + forecast_size), actuals[i], color='blue', alpha=0.3)
        plt.plot(range(offset, offset + forecast_size), preds[i], color='orange', alpha=0.3)
    plt.title("Rolling Forecasts vs. Actual RSI")
    plt.xlabel("Days")
    plt.ylabel("RSI")
    plt.legend(["Actual", "Forecast"], loc="upper left")
    plt.tight_layout()
    plt.savefig("rolling_forecasts.png")
    plt.close()

def average_overlapping_forecasts(preds: np.ndarray) -> np.ndarray:
  
    #Averages overlapping forecasts into a single smoothed forecast time series.

 
    num_windows, forecast_size = preds.shape
    total_length = num_windows + forecast_size - 1

    summed = np.zeros(total_length)
    count = np.zeros(total_length)

    for i in range(num_windows):
        for j in range(forecast_size):
            summed[i + j] += preds[i, j]
            count[i + j] += 1

    smooth = summed / np.maximum(count, 1)  # Avoid division by zero
    return smooth

def plot_smooth_vs_actual(smooth_forecast, actual_series, offset=0):
    plt.figure(figsize=(12, 6))
    time_axis = range(offset, offset + len(smooth_forecast))
    plt.plot(time_axis, actual_series[:len(smooth_forecast)], label="Actual", color="blue")
    plt.plot(time_axis, smooth_forecast, label="Smoothed Forecast", color="orange")

    plt.title("Smoothed Forecast vs. Actual RSI")
    plt.xlabel("Days")
    plt.ylabel("RSI")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("smooth_forecast_vs_actual.png")
    plt.close()

def main(train_model=True):
    
    
   # Parameters
    input_size = 14  # Number of days to look back
    forecast_size = 5  # Number of days to forecast
    batch_size = 32
    epochs = 150
        
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
        stack_types=['seasonal', 'generic', 'seasonal'],
        nb_blocks_per_stack=3,
        forecast_length=forecast_size,
        backcast_length=input_size,
        hidden_layer_units=256,
        thetas_dim=(4, 8, 4)
    )
    if train_model:
        
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
                forecast = forecast.squeeze(-1) 
                batch_y = batch_y.squeeze(-1)   
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
        
        torch.save(model.state_dict(), "nbeats_TSLA.pth")

        # Plot training loss
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.savefig('training_loss.png')
        plt.close()

    
    else:
         # Load pretrained model
        model.load_state_dict(torch.load("nbeats_TSLA.pth"))
        model.eval()
        print("[INFO] Using pre-trained model.")

       
        
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

        # Make predictions on test set
        test_data = fetch_data(start_date='2024-06-01', end_date='2025-01-01')
        rsi_test_data = test_data['RSI_norm']
        rsi_test_data = rsi_test_data.dropna()


        all_preds, all_actuals = rolling_forecast_evaluation(model, rsi_test_data, input_size, forecast_size, steps_ahead=rsi_test_data.shape[0])
        all_preds = all_preds*100
        all_actuals=all_actuals*100
        print(f"all_preds shape: {all_preds.shape}")  # (num_windows, forecast_size)
        print(f"rsi_test_data length: {len(rsi_test_data)}")
        print(f"expected smoothed forecast length: {all_preds.shape[0] + all_preds.shape[1] - 1}")

        plot_rolling_forecasts(all_preds, all_actuals, forecast_size)
        
        smooth = average_overlapping_forecasts(all_preds)
        expected_len = len(smooth)
        start_idx = len(rsi_test_data) - expected_len
        actual_aligned = rsi_test_data.values[start_idx:]*100

        print(f"Smooth forecast length: {len(smooth)}")
        print(f"RSI test data length: {len(rsi_test_data)}")
        print(f"Aligned actual slice starts at index {start_idx}")
        print(f"Aligned actual length: {len(actual_aligned)}")
        plot_smooth_vs_actual(smooth, actual_aligned)



        mse = mean_squared_error(actual_aligned, smooth)
        mae = mean_absolute_error(actual_aligned, smooth)
        r2 = r2_score(actual_aligned, smooth)

        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R² Score: {r2:.6f}")

if __name__ == "__main__":
    main(train_model=True)