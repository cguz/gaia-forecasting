import pandas as pd
import os
import glob
from file_reader import read_csv_files
from data_processor import process_data

DATA_LOCATION = '../../data/training/'

def split_data(df, column_name, sample_fraction=0.1):
    
    # Sample a fraction of the data for initial testing
    df_sampled = df.sample(frac=sample_fraction, random_state=42)

    # Calculate the train size and split data
    train_size = int(len(df_sampled) * 0.8)
    train = df_sampled.iloc[:train_size]
    test = df_sampled.iloc[train_size:]

    # Further split the train data into training and validation sets
    val_size = int(len(train) * 0.2)
    X_train = train.iloc[:-val_size].drop(columns=[column_name])
    y_train = train.iloc[:-val_size][column_name]
    X_val = train.iloc[-val_size:].drop(columns=[column_name])
    y_val = train.iloc[-val_size:][column_name]

    # Split the test data into features and target
    X_test = test.drop(columns=[column_name])
    y_test = test[column_name]

    return X_train, y_train, X_val, y_val, X_test, y_test

def main(base_filename, num_files):
    # Read and process data
    all_data = read_csv_files(base_filename, num_files)
    processed_data, column_name = process_data(all_data)

    # Split the data
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(processed_data, column_name)

    # Save the splits for future use
    os.makedirs(DATA_LOCATION, exist_ok=True)
    X_train.to_csv(f'{DATA_LOCATION}X_train.csv', index=False)
    y_train.to_csv(f'{DATA_LOCATION}y_train.csv', index=False)
    X_val.to_csv(f'{DATA_LOCATION}X_val.csv', index=False)
    y_val.to_csv(f'{DATA_LOCATION}y_val.csv', index=False)
    X_test.to_csv(f'{DATA_LOCATION}X_test.csv', index=False)
    y_test.to_csv(f'{DATA_LOCATION}y_test.csv', index=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Split data into training, validation, and testing sets.')
    parser.add_argument('data', type=str, help='The base name of the CSV files')
    parser.add_argument('num_files', type=int, help='The number of CSV files')
    args = parser.parse_args()

    main(args.data, args.num_files)