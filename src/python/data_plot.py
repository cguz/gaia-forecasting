import argparse
from file_reader import read_csv_files
from data_processor import process_data
from plotter import plot_data

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
    all_data = read_csv_files(base_filename, num_files)
    combined_df, column_name = process_data(all_data)
    if combined_df is not None:
        plot_data(combined_df, column_name, show_plot, output_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot data from multiple CSV files.')
    parser.add_argument('data', type=str, help='The base name of the CSV files')
    parser.add_argument('num_files', type=int, help='The number of CSV files')
    parser.add_argument('--output', type=str, default=None,
                        help='Optional output filename to save the plot (e.g., plot.png)')
    parser.add_argument('--no-show', action='store_false', dest='show_plot',
                        help='Do not display the plot interactively')

    args = parser.parse_args()

    plot_csv_data(args.data, args.num_files, show_plot=args.show_plot, output_filename=args.output)