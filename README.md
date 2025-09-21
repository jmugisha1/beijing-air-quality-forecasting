# Air Quality Forecasting with LSTM

This project predicts PM2.5 air pollution levels using historical air quality and weather data. It leverages an LSTM-based Recurrent Neural Network to capture temporal patterns in the data and generate future forecasts.

## Dataset

- **Train data:** `pm.csv` (historical PM2.5 and weather features)
- **Test data:** `test.csv` (features for generating predictions)
- **Output:** `joel-predictions.csv` (predicted PM2.5 values)

The `No` column is removed as it is not relevant. The datetime column is converted to a datetime index for proper time series processing.

## Features & Preprocessing

- Interpolation is used to fill missing PM2.5 values.
- Data is split into training and validation sets (last 6100 rows for validation).
- Features and target values are scaled using `MinMaxScaler`.
- `TimeseriesGenerator` is used to create sequences of 96 time steps for the LSTM.

## Model Architecture

- **Layers:**
  - LSTM(128, relu, return_sequences=True)
  - Dropout(0.2)
  - LSTM(64, relu, return_sequences=False)
  - Dropout(0.2)
  - Dense(64, relu, L2 regularization)
  - Dropout(0.2)
  - Dense(1, linear output)
- **Optimizer:** Adam (learning rate 0.001)
- **Loss:** Mean Squared Error (MSE)
- **Metrics:** Root Mean Squared Error (RMSE)

The model is trained for 25 epochs with a batch size of 32.

## Prediction & Submission

- Last 96 rows from training data are used as seed inputs for test predictions.
- Predictions are rescaled to original PM2.5 values and rounded to integers.
- Output CSV is formatted with `row ID` and `pm2.5` columns for Kaggle submission.

