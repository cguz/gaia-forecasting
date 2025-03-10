import pandas as pd
import glob

# Create a list of file names.  Adjust path as needed.
file_names = glob.glob("../../data/pre-processed/GFSD1115.*.csv")
# Initialize an empty list to hold dataframes
dfs = []

# Loop for every file
for file_name in file_names:
    # Load each CSV file into a pandas DataFrame
    df = pd.read_csv(file_name)

    # Convert 'Timestamp' column to datetime objects (assuming microseconds)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='us')

    # Append the current dataframe to the list
    dfs.append(df)

# Concatenate all dataframes in the list into a single dataframe
combined_df = pd.concat(dfs, ignore_index=True)

# Calculate descriptive statistics
description = combined_df['GFSD1115'].describe()
print("Descriptive Statistics:\n", description)

# Find the start and end dates
start_date = combined_df['Timestamp'].min()
end_date = combined_df['Timestamp'].max()

print("\nStart Date:", start_date)
print("End Date:", end_date)

# Sort the DataFrame by 'Timestamp'
combined_df = combined_df.sort_values(by='Timestamp')
print("\nFirst 5 rows of sorted data:\n", combined_df.head())

# check for duplicated timestamps
duplicates = combined_df.duplicated(subset=['Timestamp'], keep=False)
print("\nDuplicated Timestamps (if any):\n", combined_df[duplicates])