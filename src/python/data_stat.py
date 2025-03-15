import pandas as pd
import glob
import argparse
from file_reader import read_csv_files

def calculate_statistics(combined_df, data_column, timestamp_column):
    # Calculate descriptive statistics
    description = combined_df[data_column].describe()
    print("Descriptive Statistics:\n", description)

    # Find the start and end dates
    start_date = combined_df[timestamp_column].min()
    end_date = combined_df[timestamp_column].max()

    print("\nStart Date:", start_date)
    print("End Date:", end_date)

    # Sort the DataFrame by 'Timestamp'
    combined_df = combined_df.sort_values(by=timestamp_column)
    print("\nFirst 5 rows of sorted data:\n", combined_df.head())

    # Check for duplicated timestamps
    duplicates = combined_df.duplicated(subset=[timestamp_column], keep=False)
    print("\nDuplicated Timestamps (if any):\n", combined_df[duplicates])

def main(base_filename, data_column):
    num_files = len(glob.glob(f"{base_filename}.*.csv"))

    all_data = read_csv_files(base_filename, num_files)
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        calculate_statistics(combined_df, data_column, "Timestamp")
    else:
        print("No data to process.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate statistics from multiple CSV files.')
    parser.add_argument('base_filename', type=str, help='The base name of the CSV files')
    parser.add_argument('data_column', type=str, help='The name of the data column')

    args = parser.parse_args()

    main(args.base_filename, args.data_column)