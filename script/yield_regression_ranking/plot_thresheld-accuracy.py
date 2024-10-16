import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data from the uploaded Excel file
file_path = r'Y:\zhengkun.li\peanut_project\Analysis_result\LoFTR-stitching\peanut_analysis_tifton_modified.xlsx'
data = pd.read_excel(file_path)

# Calculate mean yield and mean predicted yield for each genotype
mean_values = data.groupby('Name1').agg(
    Mean_Yield=('Yield (lbs/A)', 'mean'),
    Mean_Prediction_yield=('Prediction_yield', 'mean')
).reset_index()

# Define the function to calculate relative rank accuracy within a given range
def calculate_relative_rank_accuracy(mean_values, lower_threshold, upper_threshold):
    n = len(mean_values)
    correct_predictions = [0] * n
    total_comparisons = [0] * n
    
    for i in range(n):
        for j in range(n):
            if i != j:
                yield_diff = abs(mean_values.loc[i, 'Mean_Yield'] - mean_values.loc[j, 'Mean_Yield'])
                lower_bound = (lower_threshold / 100) * mean_values.loc[i, 'Mean_Yield']
                upper_bound = (upper_threshold / 100) * mean_values.loc[i, 'Mean_Yield']
                if lower_bound <= yield_diff <= upper_bound:
                    total_comparisons[i] += 1
                    if ((mean_values.loc[i, 'Mean_Yield'] > mean_values.loc[j, 'Mean_Yield']) == 
                        (mean_values.loc[i, 'Mean_Prediction_yield'] > mean_values.loc[j, 'Mean_Prediction_yield'])):
                        correct_predictions[i] += 1
    
    relative_rank_accuracy = [correct / total if total > 0 else None for correct, total in zip(correct_predictions, total_comparisons)]
    return relative_rank_accuracy

# Define the thresholds
thresholds = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40]
accuracy_results = {}

# Calculate relative rank accuracy for each threshold range
for i in range(len(thresholds) - 1):
    lower_threshold = thresholds[i]
    upper_threshold = thresholds[i + 1]
    accuracy_results[f'{upper_threshold}%'] = calculate_relative_rank_accuracy(mean_values, lower_threshold, upper_threshold)

# Combine results into a dataframe manually
accuracy_data = {
    'Name1': mean_values['Name1']
}
for threshold_range in accuracy_results.keys():
    accuracy_data[threshold_range] = accuracy_results[threshold_range]

accuracy_df = pd.DataFrame(accuracy_data)
accuracy_df.set_index('Name1', inplace=True)

# Calculate average accuracy for each threshold range, ignoring None values
average_accuracy = accuracy_df.apply(lambda x: np.nanmean([val for val in x if val is not None]), axis=0)

# Plot the influence of the thresholds on the accuracy
plt.figure(figsize=(10, 6))
plt.plot(average_accuracy.index, average_accuracy.values, marker='o', linestyle='-', color='b')
plt.xlabel('Threshold Range of Yield Difference (%)', fontsize=18)
plt.ylabel('Average Relative Rank Accuracy', fontsize=18)
# plt.title('Influence of Yield Difference Range on Relative Rank Accuracy', fontsize=24)
plt.grid(True)


# Modify tick parameters
plt.xticks(fontsize=14)  # Increase font size for x-axis tick labels
plt.yticks(fontsize=14)  # Increase font size for y-axis tick labels


plt.show()

# Display the dataframe
print(accuracy_df)
