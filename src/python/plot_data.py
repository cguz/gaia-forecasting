import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def plot_csv_data(base_filename, num_files, show_plot=True, output_filename=None):
    """
    Plots data from multiple CSV files, handles timestamp conversion,
    and optionally saves the plot to a file.

    Args:
        base_filename: The base name of the CSV files.
        num_files: The number of CSV files.
        show_plot: Whether to display the plot (default: True).
        output_filename: Optional filename to save the plot (e.g., "my_plot.png").
                         If None, the plot is not saved.
    """
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
            return

    if not all_data:
        print("No data to plot.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values(by='Timestamp')
    column_name = combined_df.columns[1]

    threshold_down = -0.6  # Values below this are clearly anomalous
    threshold_up = 0.15  # Values below this are clearly anomalous
    # Mark as NaN
    combined_df.loc[combined_df[column_name] < threshold_down, column_name] = np.nan  
    combined_df.loc[combined_df[column_name] > threshold_up, column_name] = np.nan 

    plt.figure(figsize=(12, 6))
    plt.plot(combined_df['Timestamp'], combined_df[column_name], label=column_name)
    plt.xlabel('Timestamp (ns)')
    plt.ylabel(column_name)
    plt.title(f'{column_name} over Time')
    plt.grid(True)
    plt.legend(loc='upper left')  # Use a specific location instead of "best"
    plt.tight_layout()

    # plt.yscale('log')

    if output_filename:
        plt.savefig(output_filename) 
        print(f"Plot saved to {output_filename}")

    if show_plot:
        plt.show() 
    else:
        plt.close() 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot data from multiple CSV files.')
    parser.add_argument('base_filename', type=str, help='The base name of the CSV files')
    parser.add_argument('num_files', type=int, help='The number of CSV files')
    parser.add_argument('--output', type=str, default=None,
                        help='Optional output filename to save the plot (e.g., plot.png)')
    parser.add_argument('--no-show', action='store_false', dest='show_plot',
                        help='Do not display the plot interactively')

    args = parser.parse_args()

    plot_csv_data(args.base_filename, args.num_files, show_plot=args.show_plot, output_filename=args.output)