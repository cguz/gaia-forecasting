import pandas as pd
import os

def read_csv_files(base_filename, num_files):
    all_data = []
    for i in range(1, num_files + 1):
        filename = f"{base_filename}.{i}.csv"
        if not os.path.exists(filename):
            print(f"Warning: File {filename} not found. Skipping.")
            continue
        try:
            df = pd.read_csv(filename)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ns')
            all_data.append(df)
        except pd.errors.EmptyDataError:
            print(f"Warning: File {filename} is empty. Skipping.")
            continue
        except Exception as e:
            print(f"Error reading or processing {filename}: {e}")
            return []
    return all_data