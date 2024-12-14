import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

def get_bitcoin_data():
    print("Downloading Bitcoin data...")
    btc = yf.download('BTC-USD', start='2019-01-01')
    print("\nData downloaded successfully!")
    print(f"Data shape: {btc.shape}")
    return btc

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(prices, period=20, num_std=2):
    """Calculate Bollinger Bands"""
    middle_band = prices.rolling(window=period).mean()
    std_dev = prices.rolling(window=period).std()
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)
    return upper_band, middle_band, lower_band

def create_advanced_features(df):
    df_features = df.copy()
    
    # Basic Price Features
    df_features['Daily_Return'] = df_features['Close'].pct_change()
    df_features['Log_Return'] = np.log(df_features['Close']/df_features['Close'].shift(1))
    
    # Multiple Moving Averages
    for window in [7, 14, 30, 50, 200]:
        df_features[f'SMA_{window}'] = df_features['Close'].rolling(window=window).mean()
        df_features[f'EMA_{window}'] = df_features['Close'].ewm(span=window).mean()
    
    # Volatility Indicators
    df_features['Volatility_7'] = df_features['Daily_Return'].rolling(window=7).std()
    df_features['Volatility_30'] = df_features['Daily_Return'].rolling(window=30).std()
    
    # Volume Features
    df_features['Volume_SMA_7'] = df_features['Volume'].rolling(window=7).mean()
    df_features['Volume_SMA_30'] = df_features['Volume'].rolling(window=30).mean()
    df_features['Volume_Daily_Change'] = df_features['Volume'].pct_change()
    
    # Price Momentum
    for window in [7, 14, 30]:
        df_features[f'Momentum_{window}'] = df_features['Close'] - df_features['Close'].shift(window)
        df_features[f'ROC_{window}'] = df_features['Close'].pct_change(periods=window)
    
    # RSI
    df_features['RSI'] = calculate_rsi(df_features['Close'])
    
    # Bollinger Bands
    upper, middle, lower = calculate_bollinger_bands(df_features['Close'])
    df_features['BB_Upper'] = upper
    df_features['BB_Middle'] = middle
    df_features['BB_Lower'] = lower
    df_features['BB_Width'] = (upper - lower) / middle
    
    # Remove any rows with NaN values
    df_features = df_features.dropna()
    
    return df_features

def prepare_data(df, target_column='Close', prediction_windows=[1, 7, 30]):
    df_ml = df.copy()
    
    for window in prediction_windows:
        df_ml[f'Target_{window}d'] = df_ml[target_column].shift(-window)
    
    df_ml = df_ml.dropna()
    
    features = df_ml.drop(['Target_1d', 'Target_7d', 'Target_30d', 
                          'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], axis=1)
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_scaled = pd.DataFrame(features_scaled, columns=features.columns, index=features.index)
    
    train_test_sets = {}
    for window in prediction_windows:
        target = df_ml[f'Target_{window}d']
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, target, test_size=0.2, random_state=42)
        train_test_sets[window] = (X_train, X_test, y_train, y_test)
    
    return train_test_sets, scaler

def train_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=10
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = np.mean((predictions - y_test) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - y_test))
    return predictions, rmse, mae

def create_visualizations(y_test, predictions, window):
    fig = plt.figure(figsize=(15, 10))
    
    # Scatter plot
    plt.subplot(2, 2, 1)
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.title(f'Bitcoin Price: Predicted vs Actual ({window}-day forecast)')
    
    # Error distribution
    plt.subplot(2, 2, 2)
    errors = predictions - y_test
    plt.hist(errors, bins=30, alpha=0.75, color='blue')
    plt.xlabel('Prediction Error ($)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Prediction Errors ({window}-day forecast)')
    
    # Time series plot
    plt.subplot(2, 1, 2)
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': predictions
    }, index=y_test.index)
    results_df.plot(figsize=(15, 5), title=f'Bitcoin Price Over Time ({window}-day forecast)', ax=plt.gca())
    
    plt.tight_layout()
    plt.savefig(f'bitcoin_analysis_{window}day.png')
    plt.close()

if __name__ == "__main__":
    try:
        # Get and prepare data
        bitcoin_data = get_bitcoin_data()
        bitcoin_features = create_advanced_features(bitcoin_data)
        
        # Prepare data for multiple prediction windows
        prediction_windows = [1, 7, 30]  # 1 day, 1 week, 1 month
        train_test_sets, scaler = prepare_data(bitcoin_features, prediction_windows=prediction_windows)
        
        # Train and evaluate models for each prediction window
        models = {}
        for window in prediction_windows:
            print(f"\nTraining model for {window}-day prediction...")
            X_train, X_test, y_train, y_test = train_test_sets[window]
            
            model = train_model(X_train, y_train)
            predictions, rmse, mae = evaluate_model(model, X_test, y_test)
            
            print(f"\nModel Performance Metrics for {window}-day prediction:")
            print(f"Root Mean Square Error: ${rmse:,.2f}")
            print(f"Mean Absolute Error: ${mae:,.2f}")
            
            # Create visualizations for each prediction window
            create_visualizations(y_test, predictions, window)
            
            models[window] = model
            
    except Exception as e:
        print(f"An error occurred: {e}")