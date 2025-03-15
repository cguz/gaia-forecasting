import matplotlib.pyplot as plt

def plot_data(combined_df, column_name, show_plot=True, output_filename=None):
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