# Define your portfolio as a dictionary
portfolio = {
    'AAPL': 0.4,  # 40% in Apple
    'MSFT': 0.3,  # 30% in Microsoft
    'GOOGL': 0.3  # 30% in Google
}
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from joblib import dump, load

def get_stock_data(ticker, start="2020-01-01", end=None):
    df = yf.download(ticker, start=start, end=end, auto_adjust=False)
    df['Return'] = df['Close'].pct_change().shift(-1)
    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['Volume_Change'] = df['Volume'].pct_change()
    df = df.dropna()
    return df
def train_models(portfolio):
    models = {}
    for ticker in portfolio.keys():
        df = get_stock_data(ticker)
        X = df[['SMA_5', 'SMA_10', 'Volume_Change']]
        y = df['Return']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        models[ticker] = model
        dump(model, f"{ticker}_model.joblib")
    return models
def predict_next_day(portfolio, models=None):
    if models is None:
        models = {ticker: load(f"{ticker}_model.joblib") for ticker in portfolio.keys()}

    predictions = {}
    for ticker, weight in portfolio.items():
        df = get_stock_data(ticker)
        X_latest = df[['SMA_5', 'SMA_10', 'Volume_Change']].iloc[-1].values.reshape(1, -1)
        prediction = models[ticker].predict(X_latest)[0]
        predictions[ticker] = prediction
    return predictions
def calculate_portfolio_return(predictions, portfolio):
    total_return = sum(predictions[ticker] * weight for ticker, weight in portfolio.items())
    return total_return
if __name__ == "__main__":
    # Step 1: Train the models (only need to do once or periodically)
    models = train_models(portfolio)

    # Step 2: Make predictions
    predictions = predict_next_day(portfolio, models)
    print("Predicted individual returns for tomorrow:")
    for ticker, pred in predictions.items():
        print(f"{ticker}: {pred:.2%}")

    # Step 3: Aggregate
    port_return = calculate_portfolio_return(predictions, portfolio)
    print(f"\nPredicted portfolio return for tomorrow: {port_return:.2%}")

