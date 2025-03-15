import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_LOCATION = '../../data/training/'

def load_data():
    X_train = pd.read_csv(f'{DATA_LOCATION}X_train.csv')
    y_train = pd.read_csv(f'{DATA_LOCATION}y_train.csv')
    X_val = pd.read_csv(f'{DATA_LOCATION}X_val.csv')
    y_val = pd.read_csv(f'{DATA_LOCATION}y_val.csv')
    return X_train, y_train, X_val, y_val

def train_and_validate():
    X_train, y_train, X_val, y_val = load_data()

    # Prepare data for Prophet
    train_data = pd.DataFrame({'ds': X_train['Timestamp'], 'y': y_train})
    val_data = pd.DataFrame({'ds': X_val['Timestamp'], 'y': y_val})

    # Train the model
    model = Prophet()
    model.fit(train_data)

    # Validate the model
    val_forecast = model.predict(val_data[['ds']])
    val_forecast = val_forecast[['ds', 'yhat']]
    val_forecast = val_forecast.rename(columns={'yhat': 'y_pred'})

    # Calculate metrics
    mae = mean_absolute_error(y_val, val_forecast['y_pred'])
    mse = mean_squared_error(y_val, val_forecast['y_pred'])
    print(f"Validation MAE: {mae}")
    print(f"Validation MSE: {mse}")

    # Save the model
    model.save(f'{DATA_LOCATION}prophet_model.pkl')

if __name__ == '__main__':
    train_and_validate()