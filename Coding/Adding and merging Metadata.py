import pandas as pd

# ---------------------------
# Step 1. Read the Excel files
# ---------------------------
# Replace 'excel1.xlsx' and 'excel2.xlsx' with your actual file names.
df1 = pd.read_excel('/content/MetaDataRIS.xlsx')  # File 1, which contains the column 'Mall'
df2 = pd.read_excel('/content/דוח מרכזים מסחריים.xlsx')  # File 2, which contains the column 'מרכז מסחרי'

# ---------------------------
# Step 2. Define a matching function
# ---------------------------
def find_matching_row(row):
    """
    For a given row in df1, try to find a row in df2 where the value in the
    'מרכז מסחרי' column is a substring (case-insensitive) of the value in the 'Mall' column.
    If a match is found, return that entire row (as a Series).
    If no match is found, return a Series with NaN for each column in df2.
    """
    mall_name = str(row['Mall']).lower()  # normalize to lowercase for comparison
    for _, row2 in df2.iterrows():
        mall_candidate = str(row2['מרכז מסחרי']).lower()
        # Check if the candidate string is a substring of the mall name in df1
        if mall_candidate in mall_name:
            return row2  # return the first match found
    # If no match is found, return a Series with NaN for each column in df2
    return pd.Series({col: pd.NA for col in df2.columns})

# ---------------------------
# Step 3. Apply the matching function to df1
# ---------------------------
# This will create a new DataFrame (matches) with the same number of rows as df1,
# where each row contains the corresponding matching data from df2.
matches = df1.apply(find_matching_row, axis=1)

# ---------------------------
# Step 4. Combine the data
# ---------------------------
# Resetting indexes to ensure proper alignment, then concatenate side-by-side.
result = pd.concat([df1.reset_index(drop=True), matches.reset_index(drop=True)], axis=1)

# ---------------------------
# Step 5. Remove the 'מרכז מסחרי' column from the merged output
# ---------------------------
if 'מרכז מסחרי' in result.columns:
    result = result.drop(columns=['מרכז מסחרי'])

# ---------------------------
# Step 6. Reorder columns so that the new columns are inserted immediately after 'Mall'
# ---------------------------
# Get the original df1 columns order
df1_cols = list(df1.columns)
# The new columns come from df2 (matches) except for 'מרכז מסחרי'
new_cols = [col for col in matches.columns if col != 'מרכז מסחרי']

if 'Mall' in df1_cols:
    # Find the position of 'Mall'
    mall_index = df1_cols.index('Mall')
    # Create the new column order:
    # - All original df1 columns up to and including 'Mall'
    # - Then all the new columns from df2
    # - Then the remaining df1 columns (if any) that come after 'Mall'
    new_order = df1_cols[:mall_index+1] + new_cols + df1_cols[mall_index+1:]
    # Reorder the columns of the merged result
    result = result[new_order]
else:
    print("The 'Mall' column was not found in the first Excel file. Keeping the default order.")

# ---------------------------
# Step 7. Save the merged result to a new Excel file
# ---------------------------
output_filename = 'merged_output.xlsx'
result.to_excel(output_filename, index=False)
print(f"Merged file saved as {output_filename}")

# ---------------------------
# (Optional) Step 8. Download the file (if using Colab)
# ---------------------------
from google.colab import files
files.download(output_filename)