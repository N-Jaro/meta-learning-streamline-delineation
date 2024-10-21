import pandas as pd
import os
from sklearn.model_selection import train_test_split

# List of HUC codes to exclude, ensuring they are strings
exclude_huc_codes = [
    "190503021001", "190503021002", "190503021003", "190503021004", "190503021005",
    "190503021006", "190503021007", "190503021008", "190503021009", "190503021101",
    "190503021102", "190503021103", "190503021104", "190503021105", "190503021106",
    "190503021107", "190503021108", "190503021109", "190503021201", "190503021202",
    "190503021203", "190503021204", "190503021205", "190503021206", "190503021207",
    "190503021208", "190503021209", "190503021210", "190503021211", "190503021212",
    "190503021300", "190503021401", "190503021402", "190503021403", "190503021404",
    "190503021405", "190503021406", "190503021407", "190503021408", "190503021501",
    "190503021502", "190503021503", "190503021504", "190503021505", "190503021506",
    "190503021507", "190503021508", "190503021509", "190503021510", "190503021511"
]

# Convert exclude_huc_codes to a set for faster lookup
exclude_huc_codes = set(exclude_huc_codes)

# Read the CSV file and ensure huc_code is treated as strings
file_path = '/u/nathanj/meta-learning-streamline-delineation/scripts/Alaska/data_gen/huc_code_10_clusters.csv'  # Update with the correct path if necessary
data = pd.read_csv(file_path, dtype={'huc_code': str})

# Filter out the HUC codes that are in the exclude list
filtered_data = data[~data['huc_code'].isin(exclude_huc_codes)]

# Extract the HUC codes that are in the exclude list
excluded_data = data[data['huc_code'].isin(exclude_huc_codes)]

# Directory to save the output CSV files
output_directory = '/u/nathanj/meta-learning-streamline-delineation/scripts/Alaska/data_gen/huc_code_arrays'
os.makedirs(output_directory, exist_ok=True)

# Initialize a DataFrame to store all HUC codes with their splits
all_data_with_split = data.copy()
all_data_with_split['exclude_train_test'] = None

# Check if excluded_data is not empty before splitting
if not excluded_data.empty:
    # Split the excluded HUC codes into training (70%) and testing (30%) sets
    train_data_excluded, test_data_excluded = train_test_split(excluded_data, test_size=0.3, random_state=42)

    # Add a column to indicate the split
    train_data_excluded['exclude_train_test'] = 'train'
    test_data_excluded['exclude_train_test'] = 'test'

    # Save the training and testing HUC codes to CSV files
    train_output_file = os.path.join(output_directory, "huc_code_train.csv")
    test_output_file = os.path.join(output_directory, "huc_code_test.csv")
    train_data_excluded[['huc_code', 'exclude_train_test']].to_csv(train_output_file, index=False)
    test_data_excluded[['huc_code', 'exclude_train_test']].to_csv(test_output_file, index=False)

    # Update all_data_with_split with the train/test status
    all_data_with_split.loc[all_data_with_split['huc_code'].isin(train_data_excluded['huc_code']), 'exclude_train_test'] = 'train'
    all_data_with_split.loc[all_data_with_split['huc_code'].isin(test_data_excluded['huc_code']), 'exclude_train_test'] = 'test'

# List of columns to group by
columns_to_group_by = ['5_cluster', '10_cluster', 'random_5_cluster', 'random_10_cluster']

# Group huc_codes by each column and save to CSV files
for column in columns_to_group_by:
    train_test_column = f'{column}_train_test'
    all_data_with_split[train_test_column] = None

    for group_value, group_df in filtered_data.groupby(column):
        if len(group_df) < 20:  # If not enough data for splitting, save as test
            group_df[train_test_column] = 'test'
            test_group = group_df
            train_group = pd.DataFrame(columns=group_df.columns)  # Empty DataFrame
        else:
            # Split each group into 5% training and 95% testing
            train_group, test_group = train_test_split(group_df, test_size=0.95, random_state=42)
            train_group[train_test_column] = 'train'
            test_group[train_test_column] = 'test'
        
        # Save the training and testing sets of each group to CSV files
        train_array_name = f"huc_code_{column}_{group_value}_train"
        test_array_name = f"huc_code_{column}_{group_value}_test"
        
        train_output_file = os.path.join(output_directory, f"{train_array_name}.csv")
        test_output_file = os.path.join(output_directory, f"{test_array_name}.csv")
        
        train_group[['huc_code', train_test_column]].to_csv(train_output_file, index=False)
        test_group[['huc_code', train_test_column]].to_csv(test_output_file, index=False)
        
        # Update all_data_with_split with the train/test status
        all_data_with_split.loc[all_data_with_split['huc_code'].isin(train_group['huc_code']), train_test_column] = 'train'
        all_data_with_split.loc[all_data_with_split['huc_code'].isin(test_group['huc_code']), train_test_column] = 'test'

# Save the all_data_with_split DataFrame to a CSV file
all_data_with_split_output_file = os.path.join(output_directory, "all_huc_codes_with_split.csv")
all_data_with_split.to_csv(all_data_with_split_output_file, index=False)

print("CSV files have been created successfully.")
