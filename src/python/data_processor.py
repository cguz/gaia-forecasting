import pandas as pd
import numpy as np

def process_data(all_data):
    if not all_data:
        print("No data to plot.")
        return None, None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values(by='Timestamp')
    column_name = combined_df.columns[1]

    threshold_down = -0.6  # Values below this are clearly anomalous
    threshold_up = 0.15  # Values below this are clearly anomalous
    # Mark as NaN
    combined_df.loc[combined_df[column_name] < threshold_down, column_name] = np.nan  
    combined_df.loc[combined_df[column_name] > threshold_up, column_name] = np.nan 

    return combined_df, column_name