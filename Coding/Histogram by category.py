import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
file_path = '/content/mall_summary_shoes and fashion.xlsx'  # Update to match your actual file name
try:
    data = pd.read_excel(file_path)
    print("Excel file loaded successfully!")
except Exception as e:
    print(f"Error loading the file: {e}")
    exit()

# Print all column names for debugging
print("\nAll column names in the Excel file:")
for col in data.columns:
    print(f"'{col}'")

# Verify the column exists
column_name = 'AVG Revenue per sqm'
if column_name not in data.columns:
    print(f"\nThe column '{column_name}' was not found in the Excel file. Please check the column names.")
    exit()

# Check for missing or invalid values in the column
missing_values = data[column_name].isnull().sum()
if missing_values > 0:
    print(f"\nColumn '{column_name}' has {missing_values} missing values.")
else:
    print(f"\nColumn '{column_name}' has no missing values.")

# Extract data for the histogram
try:
    revenue_values = data[column_name].dropna()  # Drop missing values
    print(f"\nExtracted {len(revenue_values)} values for the histogram.")
except Exception as e:
    print(f"Error extracting data from column '{column_name}': {e}")
    exit()

# Define bin edges for the histogram
bin_edges = list(range(0, 201, 25))  # Adjust range as needed

# Create a histogram
plt.hist(revenue_values, bins=bin_edges, edgecolor='black')
plt.title('Histogram of Average Revenue per sqm')
plt.xlabel('Average Revenue per sqm')
plt.ylabel('Frequency')
plt.xticks(bin_edges)  # Set x-axis ticks to match bin edges
plt.show()