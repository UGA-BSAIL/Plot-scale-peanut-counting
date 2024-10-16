import pandas as pd

# Load the Excel file
file_path = '/blue/cli2/zhengkun.li/peanut_project/stitching_comparison/comparison_orb-SIFT_LoFTR.xlsx'
data = pd.read_excel(file_path)

# Calculate mean and standard deviation for each column in the original data
mean_values = data.mean()
std_values = data.std()

# Combine the results into a new DataFrame
stats_summary = pd.DataFrame({
    'Mean': mean_values,
    'Standard Deviation': std_values
})

# Display the summary statistics
print(stats_summary)

# remove the abnormal data using Z-score method 
z = (data - data.mean()) / data.std()
data = data[z.abs() < 3].dropna()

# calculate the mean and standard deviation for each column in the cleaned data
mean_values_cleaned = data.mean()
std_values_cleaned = data.std()

# print the summary statistics for the cleaned data
stats_summary_cleaned = pd.DataFrame({
    'Mean': mean_values_cleaned,
    'Standard Deviation': std_values_cleaned
})

print(stats_summary_cleaned)
