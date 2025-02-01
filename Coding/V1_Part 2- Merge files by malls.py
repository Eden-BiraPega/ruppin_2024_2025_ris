import pandas as pd
from datetime import datetime, timedelta

def load_and_prepare_data(file_path, category_column):
    """Load data from a file and add columns specific to the category."""
    df = pd.read_excel(file_path)

    # Add binary category column
    df[category_column] = 1

    # Create open Saturday column
    open_saturday_column = f"{category_column} open saturday"
    df[open_saturday_column] = df['Saturdays Open'].notna().astype(int)

    # Add average revenue per square meter column
    average_column = f"AVG Revenue per sqm_{category_column}"
    df[average_column] = df['AVG Revenue per sqm']

    return df

def merge_and_group_data(dfs):
    """Merge multiple DataFrames and group by 'Mall'."""
    merged_df = pd.concat(dfs, ignore_index=True)
    grouped_df = merged_df.groupby("Mall").first().reset_index()

    # Compute averages for סך הכל columns after grouping
    for col in [col for col in merged_df.columns if col.startswith("AVG Revenue per sqm")]:
        grouped_df[col] = merged_df.groupby("Mall")[col].mean().reset_index(drop=True)

    return grouped_df

# Function to fill missing values and convert columns to binary
def fill_missing_and_convert_to_binary(df, columns):
    """Fill missing values and convert specified columns to binary."""
    for column in columns:
        df[column] = df[column].fillna(0).astype(int)
    return df

# Function to save DataFrame to an Excel file
def save_to_excel(df, output_file):
    """Save the DataFrame to an Excel file."""
    df.to_excel(output_file, index=False)
    print(f"Merged file saved as {output_file}")

def add_columns_by_category(df, file_paths, categories):
    """
    Adds relevant columns (e.g., 'Total Days') from each category file to the main DataFrame,
    while keeping 'Mall', 'Starting date', and 'Ending date' unchanged.

    Parameters:
        df (pd.DataFrame): The main DataFrame to update.
        file_paths (dict): A dictionary mapping category names to their file paths.
        categories (list): A list of category names.

    Returns:
        pd.DataFrame: Updated DataFrame with the new columns.
    """
    for category in categories:
        # Load the respective category file
        file_path = file_paths[category]
        category_df = pd.read_excel(file_path)

        # Ensure the key column for merging exists in both DataFrames
        merge_key = "Mall"  # Adjust if your key column is different

        # Select relevant columns
        relevant_columns = ["Mall", "Total Days"]
        if "Starting date" in category_df.columns:
            relevant_columns.append("Starting date")
        if "Ending date" in category_df.columns:
            relevant_columns.append("Ending date")

        # Rename the relevant columns to include the category name
        category_df = category_df[relevant_columns]
        category_df.rename(columns={"Total Days": f"Total Days_{category}"}, inplace=True)

        # Merge with the main DataFrame
        df = df.merge(category_df, how="left", on=merge_key)

    return df

#Function that add "general mall" column for all the general calculate that isnt specific mall
def add_general_mall_column(df, mall_column):
    """
    Add a binary column 'general mall' to indicate if the mall is in the specified list.
    """
    general_malls = [
        'ממוצע ארצי',
        'ממוצע קניונים אזוריים',
        'ממוצע פאוור סנטרס',
        'מרכזים שכונתיים',
        'מרכזים סגורים',
        'מרכזים פתוחים'
    ]
    df['general mall'] = df[mall_column].apply(lambda mall: 1 if mall in general_malls else 0)
    return df

def add_open_stores_columns(df, categories):
    """
    Add columns for each category representing the value of 'מספר חנויות'.
    """
    for category in categories:
        column_name = f"AVG Sample stores_{category}"
        df[column_name] = df.apply(
            lambda row: row['AVG Sample stores'] if row[category] == 1 else 0, axis=1
        )
    return df

def add_mall_size_column(df, average_column):
    """
    Add a column 'Mall_Size' based on the value of 'AVG Revenue per sqm'.
    """
    def classify_size(value):
        if value <= 25:
            return 'Small'
        elif 25 < value <= 100:
            return 'Medium'
        elif 100 < value <= 150:
            return 'Big'
        else:
            return 'Outlier'

    df['Mall_Size'] = df[average_column].apply(classify_size)
    return df

#Function that add 3 column from previous files, Actual Days column from each category
def add_actual_days_opened_columns(df, file_paths, categories):
    """
    Adds "Actual Days_<category name>" columns to the DataFrame.

    Parameters:
        df (pd.DataFrame): The combined DataFrame.
        file_paths (dict): A dictionary mapping category names to their file paths.
        categories (list): A list of category names.

    Returns:
        pd.DataFrame: Updated DataFrame with the new columns.
    """
    for category in categories:
        # Load the respective Excel file
        file_path = file_paths[category]
        category_df = pd.read_excel(file_path)

        # Ensure the key column for merging exists in both DataFrames
        merge_key = "Mall"  # Change "Mall" if your merging key column is different

        # Merge the "Actual Days" column
        df = df.merge(category_df[[merge_key, "Actual Days"]],
                      how="left",
                      on=merge_key,
                      suffixes=("", f"_{category}"))

        # Rename the column to include the category name
        new_column_name = f"Actual Days_{category}"
        df.rename(columns={"Actual Days": new_column_name}, inplace=True)

    return df

def add_total_weekdays_columns(df, file_paths, categories):
    """
    Adds "Total Weekdays %_<category>" columns to the DataFrame from external files.

    Parameters:
        df (pd.DataFrame): The combined DataFrame.
        file_paths (dict): A dictionary mapping category names to their file paths.
        categories (list): A list of category names.

    Returns:
        pd.DataFrame: Updated DataFrame with the new columns.
    """
    for category in categories:
        # Load the respective file for the category
        file_path = file_paths[category]
        category_df = pd.read_excel(file_path)

        # Ensure the key column for merging exists in both DataFrames
        merge_key = "Mall"  # Adjust this if your key column is different

        # Merge the "Total Weekdays %" column
        df = df.merge(category_df[[merge_key, "Total Weekdays %"]],
                      how="left",
                      on=merge_key,
                      suffixes=("", f"_{category}"))

        # Rename the column to include the category name
        new_column_name = f"Total Weekdays %_{category}"
        df.rename(columns={"Total Weekdays %": new_column_name}, inplace=True)

    return df

def add_keeping_mall_column(df, categories):
    """
    Add 'Keeping_Mall' columns for each category from the input data.

    Parameters:
        df (pd.DataFrame): The main DataFrame to update.
        categories (list): List of categories for which to add the 'Keeping_Mall' column.

    Returns:
        pd.DataFrame: The updated DataFrame with 'Keeping_Mall' columns added for each category.
    """
    for category in categories:
        column_name = f"Keeping_Mall_{category}"
        df[column_name] = df.apply(
            lambda row: row['Keeping_Mall'] if row[category] == 1 else 0, axis=1
        )
    return df

import pandas as pd

def load_and_prepare_data(file_path, category_name):
    """
    Loads and prepares data by renaming columns with the category name.
    """
    df = pd.read_excel(file_path)
    df.columns = [
        f"{col}_{category_name}" if col not in ["Mall", "Starting Date", "Ending Date"] else col
        for col in df.columns
    ]
    return df

def merge_dataframes(file_paths, categories):
    """
    Merges the dataframes from all categories into one, ensuring unique mall names
    and filling missing data with '0'.
    """
    merged_df = None
    for category, file_path in file_paths.items():
        df = load_and_prepare_data(file_path, categories[category])
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(
                merged_df, df, on=["Mall", "Starting Date", "Ending Date"], how="outer"
            )

    # Fill missing data with '0'
    merged_df.fillna(0, inplace=True)

    # Remove duplicates by consolidating data for each mall
    def consolidate(group):
        for col in group.select_dtypes(include=['datetime64[ns]']):
            group[col] = group[col].iloc[0]
        for col in group.select_dtypes(include=['number']):
            group[col] = group[col].sum()
        for col in group.select_dtypes(include=['object']):
            group[col] = group[col].iloc[0]
        return group.iloc[0]

    merged_df = merged_df.groupby("Mall", as_index=False).apply(consolidate)

    return merged_df

def add_general_mall_column(df):
    """
    Adds a binary column `general mall` to indicate if the mall is general.
    """
    general_malls = [
        'ממוצע ארצי',
        'ממוצע קניונים אזוריים',
        'ממוצע פאוור סנטרס',
        'מרכזים שכונתיים',
        'מרכזים סגורים',
        'מרכזים פתוחים'
    ]
    df['general mall'] = df['Mall'].apply(lambda mall: 1 if mall in general_malls else 0)
    return df

def reorder_columns(df, fixed_columns):
    """
    Reorders columns to ensure fixed columns appear first.
    """
    all_columns = list(df.columns)
    reordered_columns = fixed_columns + [col for col in all_columns if col not in fixed_columns]
    return df[reordered_columns]

def remove_unwanted_columns(df):
    """
    Removes unwanted columns from the DataFrame.
    """
    unwanted_columns = [
        col for col in df.columns
        if any(keyword in col for keyword in ["Saturdays Open", "Total Saturdays", "Percentage of Saturdays"])
    ]
    df.drop(columns=unwanted_columns, inplace=True)
    return df

def main():
    # File paths and categories
    file_paths = {
        "general": "/content/mall_summary general economy.xlsx",
        "shoes_and_fashion": "/content/mall_summary_shoes and fashion.xlsx",
        "other": "/content/mall_summary others.xlsx",
    }

    categories = {
        "general": "General Economy",
        "shoes_and_fashion": "Shoes and Fashion",
        "other": "Other",
    }

    # Merge dataframes
    merged_df = merge_dataframes(file_paths, categories)

    # Add "general mall" column
    merged_df = add_general_mall_column(merged_df)

    # Add "Mall_Size" column based on "AVG Revenue per sqm"
    if "AVG Revenue per sqm_General Economy" in merged_df.columns:
        merged_df = add_mall_size_column(merged_df, "AVG Revenue per sqm_General Economy")

    # Remove unwanted columns
    merged_df = remove_unwanted_columns(merged_df)

    # Reorder columns
    fixed_columns = ["Mall", "Starting Date", "Ending Date"]
    merged_df = reorder_columns(merged_df, fixed_columns)

    # Save the final DataFrame to Excel
    output_file = "merged_mall_data.xlsx"
    merged_df.to_excel(output_file, index=False)
    print(f"Merged data saved to {output_file}")

if __name__ == "__main__":
    main()

