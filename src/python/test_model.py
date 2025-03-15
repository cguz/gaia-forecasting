import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_LOCATION = '../../data/training/'

def load_data():
    X_test = pd.read_csv(f'{DATA_LOCATION}X_test.csv')
    y_test = pd.read_csv(f'{DATA_LOCATION}y_test.csv')
    return X_test, y_test

def test_model():
    X_test, y_test = load_data()

    # Load the model
    model = Prophet()
    model = model.load(f'{DATA_LOCATION}prophet_model.pkl')

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

if __name__ == '__main__':
    test_model()