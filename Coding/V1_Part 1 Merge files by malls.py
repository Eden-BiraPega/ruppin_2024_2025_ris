import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

def merge_and_organize_excel(files):
    """
    Merges and organizes data from three Excel files by mall names, keeping the column format.

    Args:
    files (list of str): List of file paths to the three Excel files.

    Returns:
    pd.DataFrame: Merged and organized DataFrame.
    """
    # Initialize an empty DataFrame
    merged_data = pd.DataFrame()

    # Read and concatenate data from the Excel files
    for file in files:
        df = pd.read_excel(file)
        merged_data = pd.concat([merged_data, df], ignore_index=True)

    # Ensure the column order
    merged_data = merged_data[['מרכז מסחרי', 'מספר חנויות במדגם', 'יום', 'סך הכול']]

    # Fill down the mall name in 'מרכז מסחרי' column
    merged_data['מרכז מסחרי'] = merged_data['מרכז מסחרי'].fillna(method='ffill')

    # Drop rows that have 'ממוצע' in 'מרכז מסחרי'
    merged_data = merged_data[merged_data['מרכז מסחרי'] != 'ממוצע']

    # Convert the 'יום' column to datetime format
    merged_data['יום'] = pd.to_datetime(merged_data['יום'], format='%d/%m/%Y', errors='coerce')

    # Handle missing or invalid values
    merged_data['מספר חנויות במדגם'] = pd.to_numeric(merged_data['מספר חנויות במדגם'], errors='coerce')
    merged_data['סך הכול'] = pd.to_numeric(merged_data['סך הכול'], errors='coerce')

    # Fill missing values or handle them (e.g., fill with 0 or the column mean)
    merged_data['מספר חנויות במדגם'].fillna(0, inplace=True)
    merged_data['סך הכול'].fillna(0, inplace=True)

    # Convert types after handling NaN
    merged_data['מספר חנויות במדגם'] = merged_data['מספר חנויות במדגם'].astype(int)

    # Sort data by mall names and dates
    merged_data = merged_data.sort_values(by=['מרכז מסחרי', 'יום'])

    # Remove rows with zero values and no date
    merged_data = merged_data[~((merged_data['סך הכול'] == 0) & (merged_data['יום'].isna()))]

    return merged_data

# Example usage:
# Provide paths to your Excel files
file_paths = ["/content/אופנה והנעלה 2010-2014 עותק.xlsx", "/content/אופנה והנעלה 2015-2019 עותק.xlsx",
            "/content/אופנה והנעלה 2020-2024 עותק.xlsx"]
result = merge_and_organize_excel(file_paths)

# Save the result to a new Excel file
result.to_excel('merged_and_organized.xlsx', index=False)

print("Merged and organized data saved to 'merged_and_organized.xlsx'")


# Define a function to replace zeros with NaN
def replace_zeros_with_nan(df):
    """
    Replaces all occurrences of 0 in the DataFrame with NaN.

    Args:
    df (pd.DataFrame): The DataFrame to modify.

    Returns:
    pd.DataFrame: Modified DataFrame with 0s replaced by NaN.
    """
    return df.replace(0, np.nan)

def calculate_sample_stores_average(df):
    mall_averages = df.groupby('מרכז מסחרי')['מספר חנויות במדגם'].mean()
    return df['מרכז מסחרי'].map(mall_averages)

def add_total_weekdays_percentage_column(df):
    """
    Adds a new column "Total Weekdays %" to the DataFrame.

    This column calculates the percentage of weekdays excluding Saturdays
    from the 'Actual Days' column using the formula:
    ((Actual Days - Saturdays Open) * 100) / Actual Days

    Args:
    df (pd.DataFrame): The input DataFrame containing 'Actual Days' and 'Saturdays Open' columns.

    Returns:
    pd.DataFrame: DataFrame with the added "Total Weekdays %" column.
    """
    if 'Actual Days' in df.columns and 'Saturdays Open' in df.columns:
        # Calculate Total Weekdays %
        df['Total Weekdays %'] = ((df['Actual Days'] - df['Saturdays Open']) * 100) / df['Actual Days']
        # Round to 2 decimal places
        df['Total Weekdays %'] = df['Total Weekdays %'].round(2)
    else:
        raise KeyError("The DataFrame must contain 'Actual Days' and 'Saturdays Open' columns.")

    return df

# Define the function to add the "Keeping_Mall" column
def add_keeping_mall_column(df):
    """
    Adds a column 'Keeping_Mall' to the DataFrame.
    Returns 0 if at least one of the following is true:
        - The mall is under 60% in the 'Filled Percentage (%)' column.
        - The mall's 'Ending Date' is before 31/12/2023 00:00:00.
        - The mall's 'Total Days' is under 730 days (2 years).
        - The mall's 'Total Weekdays %' is 60 or below.
    Otherwise, returns 1.

    Parameters:
        df (pd.DataFrame): The DataFrame to update.

    Returns:
        pd.DataFrame: Updated DataFrame with the new column.
    """
    # Convert 'Ending Date' to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df['Ending Date']):
        df['Ending Date'] = pd.to_datetime(df['Ending Date'], errors='coerce')

    # Calculate conditions
    condition_ending_date = df['Ending Date'] < pd.Timestamp("2023-12-31")
    condition_total_days = df['Total Days'] < 1460
    condition_total_weekdays = df['Total Weekdays %'] <= 90

    # Add 'Keeping_Mall' column
    df['Keeping_Mall'] = (~(
        condition_ending_date | condition_total_days | condition_total_weekdays
    )).astype(int)

    return df

def add_saturdays_count_column(df):
    """
    Adds a new column "Total Saturdays" to the DataFrame.

    This column calculates the number of Saturdays between 'Starting Date' and 'Ending Date' for each row.

    Args:
    df (pd.DataFrame): The input DataFrame containing 'Starting Date' and 'Ending Date' columns.

    Returns:
    pd.DataFrame: DataFrame with the added "Total Saturdays" column.
    """
    if 'Starting Date' in df.columns and 'Ending Date' in df.columns:
        # Ensure dates are in the correct format
        df['Starting Date'] = pd.to_datetime(df['Starting Date'], errors='coerce')
        df['Ending Date'] = pd.to_datetime(df['Ending Date'], errors='coerce')

        # Calculate the number of Saturdays for each row
        df['Total Saturdays'] = df.apply(
            lambda row: len(pd.date_range(start=row['Starting Date'], end=row['Ending Date'], freq='W-SAT'))
            if pd.notnull(row['Starting Date']) and pd.notnull(row['Ending Date']) else 0,
            axis=1
        )
    else:
        raise KeyError("The DataFrame must contain 'Starting Date' and 'Ending Date' columns.")

    return df

def add_percentage_of_saturdays_column(df):
    """
    Adds the "percentage of saturdays" column to the DataFrame.

    This column calculates the percentage of Saturdays not open based on the formula:
    ((Total Saturdays - Saturdays Open) * 100) / Total Saturdays.

    Args:
    df (pd.DataFrame): The input DataFrame containing 'Total Saturdays' and 'Saturdays Open' columns.

    Returns:
    pd.DataFrame: DataFrame with the added "percentage of saturdays" column.
    """
    if 'Total Saturdays' in df.columns and 'Saturdays Open' in df.columns:
        # Calculate "percentage of saturdays"
        df['percentage of saturdays'] = ((df['Saturdays Open'] / df['Total Saturdays']) * 100)
        df['percentage of saturdays'] = df['percentage of saturdays'].round(2)  # Round to 2 decimal places
    else:
        raise KeyError("The DataFrame must contain 'Total Saturdays' and 'Saturdays Open' columns.")

    return df

def add_binary_saturdays_column(df):
    """
    Adds the "Binary Saturdays" column to the DataFrame.

    This column assigns a binary value (0 or 1) based on the "percentage of saturdays" column:
    If the "percentage of saturdays" is greater than 50%, it assigns 1, otherwise 0.

    Args:
    df (pd.DataFrame): The input DataFrame containing the "percentage of saturdays" column.

    Returns:
    pd.DataFrame: DataFrame with the added "Binary Saturdays" column.
    """
    if 'percentage of saturdays' in df.columns:
        # Calculate "Binary Saturdays"
        df['Binary Saturdays'] = df['percentage of saturdays'].apply(lambda x: 1 if x > 50 else 0)
    else:
        raise KeyError("The DataFrame must contain 'percentage of saturdays' column.")

    return df

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/content/merged_and_organized.xlsx'
df = pd.read_excel(file_path)
df['יום'] = pd.to_datetime(df['יום'], errors='coerce')  # Ensure proper datetime parsing

# Replace all zeros with NaN
df = replace_zeros_with_nan(df)

# Fill down mall names
df['מרכז מסחרי'] = df['מרכז מסחרי'].fillna(method='ffill')

# Filter the dataset for a desired date range
start_date = pd.to_datetime('2010-01-01')
end_date = pd.to_datetime('2024-10-31')
df = df.loc[(df['יום'] >= start_date) & (df['יום'] <= end_date)]
df['Sample stores'] = calculate_sample_stores_average(df)

# Define date ranges to exclude
excluded_ranges = [
    (pd.to_datetime('2020-03-14'), pd.to_datetime('2020-05-07')),
    (pd.to_datetime('2020-09-19'), pd.to_datetime('2021-02-20'))
]

# Generate all Saturdays between the specified dates
saturdays = pd.date_range(start=start_date, end=end_date, freq='W-SAT')

# Add each Saturday as a single-day range to the excluded_ranges
for saturday in saturdays:
    excluded_ranges.append((saturday, saturday))

# Initialize a list to store results
mall_results = []

# Group by 'מרכז מסחרי'
for mall_name, group in df.groupby('מרכז מסחרי'):
    if group.empty:
        continue

    # Ensure the group contains valid dates and non-NaN values in 'סך הכול'
    group = group[group['סך הכול'].notna() | group['יום'].notna()]

    # Determine the first and last date with valid 'סך הכול' values
    if not group.empty:
        sorted_group = group.sort_values(by='יום')
        first_date = sorted_group.loc[sorted_group['סך הכול'].notna(), 'יום'].min().date()  # First valid date
        last_date = sorted_group.loc[sorted_group['סך הכול'].notna(), 'יום'].max().date()  # Last valid date
    else:
        first_date = None
        last_date = None

    # Calculate the total days between first_date and last_date
    total_days = (last_date - first_date).days + 1 if first_date and last_date else 0

    # Adjust total_days based on excluded_ranges
    if first_date and last_date:
        for start, end in excluded_ranges:
            overlap_start = max(pd.Timestamp(first_date), start)
            overlap_end = min(pd.Timestamp(last_date), end)
            if overlap_start <= overlap_end:  # Check if there's an overlap
                total_days -= (overlap_end - overlap_start).days + 1

    # Filter the group for dates between first_date and last_date, excluding excluded_ranges
    valid_group = group[(group['יום'] >= pd.Timestamp(first_date)) & (group['יום'] <= pd.Timestamp(last_date))]
    for start, end in excluded_ranges:
        valid_group = valid_group[~((valid_group['יום'] >= start) & (valid_group['יום'] <= end))]

    # Calculate empty cells count
    empty_cells_count = valid_group['סך הכול'].isna().sum()

    # Calculate actual days
    actual_days = total_days - empty_cells_count

    # Calculate the filled percentage
    filled_percentage = ((actual_days) / total_days * 100) if total_days > 0 else 0

    # Other calculations
    average_total = group[(group['סך הכול'] != 0) & (group['סך הכול'].notna())]['סך הכול'].mean(skipna=True)
    saturday_open_count = group[(group['יום'].dt.dayofweek == 5) & (group['סך הכול'].notna())].shape[0]
    # Count Saturdays for the current group
    if first_date and last_date:  # Ensure dates are valid
        temp_df = pd.DataFrame({
            'Starting Date': [first_date],
            'Ending Date': [last_date]
        })
        temp_df = add_saturdays_count_column(temp_df)
        total_saturdays = temp_df['Total Saturdays'].iloc[0]  # Get the calculated Saturdays
    else:
        total_saturdays = 0  # Set to 0 if dates are invalid

    # Add 'Total Saturdays' column to the group
    group['Total Saturdays'] = total_saturdays

    # Calculate 'Saturdays Open' for the current group
    saturday_open_count = group[(group['יום'].dt.dayofweek == 5) & (group['סך הכול'].notna())].shape[0]
    group['Saturdays Open'] = saturday_open_count

    # Handle cases where 'Total Saturdays' is 0 to avoid division by zero
    if total_saturdays > 0:
        # Add 'percentage of saturdays' column
        group = add_percentage_of_saturdays_column(group)

        # Add 'Binary Saturdays' column
        group = add_binary_saturdays_column(group)
    else:
        # If 'Total Saturdays' is 0, set default values
        group['percentage of saturdays'] = 0
        group['Binary Saturdays'] = 0

    # Calculate metrics for the current group
    percentage_of_saturdays = group['percentage of saturdays'].mean()  # Average percentage of Saturdays
    binary_saturdays = group['Binary Saturdays'].iloc[0]  # Binary indicator (consistent per group)


    # Append results
    mall_results.append({
        'Mall': mall_name,
        'Starting Date': first_date,
        'Ending Date': last_date,
        'Total Days': total_days,
        'Empty Cells': empty_cells_count,
        'Actual Days': actual_days,
        'Saturdays Open': saturday_open_count,
        'Total Saturdays': total_saturdays,
        'Percentage of Saturdays': percentage_of_saturdays,
        'Binary Saturdays': binary_saturdays,
        'Filled Percentage (%)': filled_percentage,
        'Total Weekdays %': ((actual_days - saturday_open_count) * 100 / actual_days) if actual_days > 0 else 0,
        'AVG Revenue per sqm': average_total,
        'AVG Sample stores': group['Sample stores'].mean()
    })

# Convert results to DataFrame
summary_df = pd.DataFrame(mall_results)

# Add the "Keeping_Mall" column
summary_df = add_keeping_mall_column(summary_df)

# Save the summary to an Excel file
output_file = "mall_summary_updated.xlsx"
summary_df.to_excel(output_file, index=False)

# Debug: Check the summary DataFrame
print("Updated Summary DataFrame:")
print(summary_df.head())

# Visualization
if not summary_df.empty:
    # Filled Percentage Comparison
    plt.figure(figsize=(10, 6))
    summary_df.set_index('Mall')['Filled Percentage (%)'].sort_values().plot(kind='bar', color='skyblue')
    plt.title('Filled Percentage (%) Comparison by Mall')
    plt.ylabel('Filled Percentage (%)')
    plt.xlabel('Mall')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("No valid data to display.")