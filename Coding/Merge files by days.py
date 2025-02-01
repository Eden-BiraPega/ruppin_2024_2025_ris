#'pip install hdate', write it in the terminal
import pandas as pd
import numpy as np
from hdate import HDate
from google.colab import files

def read_excel_files(file_paths):
    """Reads multiple Excel files and concatenates them into a single DataFrame."""
    dfs = [pd.read_excel(file) for file in file_paths]
    return pd.concat(dfs, ignore_index=True)

def save_dataframe_to_excel(df, output_file):
    """Saves a DataFrame to an Excel file."""
    df.to_excel(output_file, index=False)
    print(f"File saved to: {output_file}")

def fill_missing_values(df):
    """Fills missing values in the 'מרכז מסחרי' column using forward fill."""
    df['מרכז מסחרי'] = df['מרכז מסחרי'].fillna(method='ffill')
    return df

def pivot_and_expand(df, value_column, new_column_suffix):
    """Creates and expands pivot tables for a specified value column."""
    pivoted = df.pivot_table(
        index='יום',
        columns='מרכז מסחרי',
        values=value_column,
        aggfunc=lambda x: list(x)
    )
    expanded = pivoted.apply(lambda col: col.explode()).reset_index()
    expanded.columns = [
        f"{col}_{new_column_suffix}" if col != 'יום' else col for col in expanded.columns
    ]
    return expanded

def merge_pivoted_tables(expanded_totals, expanded_shops):
    """Merges expanded pivot tables on the 'יום' column."""
    return expanded_totals.merge(
        expanded_shops,
        on='יום',
        suffixes=('_סך הכול', '_מספר חנויות')
    )

def add_date_components(df):
    """Adds Gregorian and Hebrew date components to the DataFrame."""
    df['יום'] = pd.to_datetime(df['יום'], dayfirst=True).dt.date
    df['יום_יום'] = df['יום'].apply(lambda x: x.day)
    df['יום_חודש'] = df['יום'].apply(lambda x: x.month)
    df['יום_שנה'] = df['יום'].apply(lambda x: x.year)
    df['יום_יום בשבוע'] = pd.to_datetime(df['יום']).dt.day_name()

    hebrew_days = {
        "Sunday": "ראשון",
        "Monday": "שני",
        "Tuesday": "שלישי",
        "Wednesday": "רביעי",
        "Thursday": "חמישי",
        "Friday": "שישי",
        "Saturday": "שבת"
    }
    df['יום_יום בשבוע'] = df['יום_יום בשבוע'].replace(hebrew_days)
    df['יום_תאריך עברי'] = df['יום'].apply(lambda x: HDate(x).__str__())
    return df

def reorder_columns(df):
    """Reorders columns to group metrics for each 'מרכז מסחרי' together."""
    columns_order = ['יום', 'יום_יום', 'יום_חודש', 'יום_שנה', 'יום_יום בשבוע', 'יום_תאריך עברי']
    for col in df.columns:
        if col not in columns_order:
            base_name = col.split('_')[0]
            if base_name not in columns_order:
                columns_order.extend([
                    f"{base_name}_סך הכול",
                    f"{base_name}_מספר חנויות"
                ])

    ordered_columns = columns_order + [col for col in df.columns if col not in columns_order]
    df = df[ordered_columns]
    return df.loc[:, ~df.columns.duplicated()]

def replace_zeros_with_nan(df):
    """Replaces all cells with the value 0 with NaN."""
    return df.replace(0, pd.NA)

def drop_rows_with_nan_in_columns(df, columns):
    """Drops rows where any of the specified columns have NaN."""
    return df.dropna(subset=columns)

# Main processing function
def process_file(input_file, output_file):
    """Processes the input file and generates the output file with required transformations."""
    df = pd.read_excel(input_file)
    df = fill_missing_values(df)

    # Replace zeros with NaN
    df = replace_zeros_with_nan(df)

    # Drop rows with NaN in specific columns
    columns_to_check = ['סך הכול', 'מספר חנויות במדגם']
    df = drop_rows_with_nan_in_columns(df, columns_to_check)

    # Create pivot tables
    expanded_totals = pivot_and_expand(df, 'סך הכול', 'סך הכול')
    expanded_shops = pivot_and_expand(df, 'מספר חנויות במדגם', 'מספר חנויות')

    # Merge pivoted tables
    combined_expansion = merge_pivoted_tables(expanded_totals, expanded_shops)

    # Add date components
    combined_expansion = add_date_components(combined_expansion)

    # Sort and reorder columns
    combined_expansion = combined_expansion.sort_values(by='יום').reset_index(drop=True)
    final_table_reordered = reorder_columns(combined_expansion)

    # Save the final DataFrame
    save_dataframe_to_excel(final_table_reordered, output_file)

# Main script execution
uploaded = [
    "/content/משק כללי 2010-2014 עותק.xlsx",
    "/content/משק כללי 2015-2019 עותק.xlsx",
    "/content/משק כללי 2020-2024 עותק.xlsx"
]

combined_file_path = "combined_all_files.xlsx"
final_output_file = "New_Sorted_Combined_Values_with_Hebrew_Dates.xlsx"

# Step 1: Combine files
final_df = read_excel_files(uploaded)
save_dataframe_to_excel(final_df, combined_file_path)

# Step 2: Process combined file
process_file(combined_file_path, final_output_file)

# Step 3: Download the final file
files.download(final_output_file)

