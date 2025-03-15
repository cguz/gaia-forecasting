import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import matplotlib.pyplot as plt

DATA_LOCATION = '../../data/training/'

def load_data():
    X_test = pd.read_csv(f'{DATA_LOCATION}X_test.csv')
    y_test = pd.read_csv(f'{DATA_LOCATION}y_test.csv').squeeze()
    return X_test, y_test

def test_model():
    X_test, y_test = load_data()

    # Ensure Timestamp is in datetime format
    X_test['Timestamp'] = pd.to_datetime(X_test['Timestamp'])

    # Load the model
    model = joblib.load(f'{DATA_LOCATION}prophet_model.pkl')

    # Prepare test data
    test_data = pd.DataFrame({'ds': X_test['Timestamp'], 'y': y_test})

    # Test the model
    test_forecast = model.predict(test_data[['ds']])
    test_forecast = test_forecast[['ds', 'yhat']]
    test_forecast = test_forecast.rename(columns={'yhat': 'y_pred'})

    # Calculate metrics
    mae = mean_absolute_error(y_test, test_forecast['y_pred'])
    mse = mean_squared_error(y_test, test_forecast['y_pred'])
    print(f"Test MAE: {mae}")
    print(f"Test MSE: {mse}")

    # Plot test data and predictions
    plt.figure(figsize=(10, 6))
    plt.plot(test_data['ds'], test_data['y'], label='Test Data')
    plt.plot(test_forecast['ds'], test_forecast['y_pred'], label='Test Predictions')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Test Data and Predictions')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    test_model()