# Bitcoin Price Predictor

## Overview
This project uses machine learning to predict Bitcoin prices across different time horizons (1-day, 7-day, and 30-day forecasts) using scikit-learn's Random Forest Regressor. The model incorporates various technical indicators and market features to make its predictions.

## Requirements
```
python >= 3.6
yfinance
pandas
numpy
scikit-learn
matplotlib
```

## Installation
1. Clone this repository:
```bash
git clone <repository-url>
```

2. Install required packages:
```bash
pip install yfinance pandas numpy scikit-learn matplotlib
```

## Usage
Simply run the main script:
```bash
python bitcoin_predictor.py
```

The script will:
1. Download Bitcoin historical data from Yahoo Finance (2019-present)
2. Generate technical indicators and features
3. Train models for different prediction windows
4. Create visualization plots saved as PNG files
5. Display performance metrics for each model

## Features Generated
- Multiple Moving Averages (SMA & EMA)
- RSI (Relative Strength Index)
- Bollinger Bands
- Volatility Indicators
- Volume Analysis
- Price Momentum
- Daily Returns
- Log Returns

## Output
The script generates three visualization files:
- `bitcoin_analysis_1day.png`: 1-day forecast analysis
- `bitcoin_analysis_7day.png`: 7-day forecast analysis
- `bitcoin_analysis_30day.png`: 30-day forecast analysis

Each visualization includes:
- Predicted vs Actual price scatter plot
- Prediction error distribution
- Price trends over time

## Performance Metrics
The model's performance is evaluated using:
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)

## Model Details
- Algorithm: Random Forest Regressor
- Training/Test Split: 80/20
- Features are standardized using StandardScaler
- Model Parameters:
  - n_estimators: 100
  - max_depth: 10
  - random_state: 42

## Limitations
- Past performance doesn't guarantee future results
- Market conditions and external factors can impact accuracy
- Predictions should not be used as sole financial advice

## Contributing
Feel free to fork this repository and submit pull requests for improvements.

## License
[Insert your chosen license here]
