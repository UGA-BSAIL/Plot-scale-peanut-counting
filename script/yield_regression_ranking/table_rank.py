import pandas as pd
import numpy as np

# Load the CSV file
file_path = r'Y:\zhengkun.li\peanut_project\Analysis_result\LoFTR-stitching\peanut_analysis_ranks.csv'
peanut_analysis_df = pd.read_csv(file_path)

# Extract relevant columns
genotypes = peanut_analysis_df['Genotypes']
mean_prediction_yield = peanut_analysis_df['Mean_Prediction_yield']
mean_yield = peanut_analysis_df['Mean_Yield (lbs/A)']

# Initialize confusion matrix and accuracy calculation
confusion_matrix = np.zeros((len(genotypes), len(genotypes)))
correct_predictions = 0
total_comparisons = 0

# Create confusion matrix
for i in range(len(genotypes)):
    for j in range(len(genotypes)):
        if i != j:
            total_comparisons += 1
            if ((mean_prediction_yield[i] > mean_prediction_yield[j]) and (mean_yield[i] > mean_yield[j])) or \
               ((mean_prediction_yield[i] < mean_prediction_yield[j]) and (mean_yield[i] < mean_yield[j])):
                confusion_matrix[i, j] = 1  # Correct prediction
                correct_predictions += 1
            else:
                confusion_matrix[i, j] = -1  # Incorrect prediction

# Calculate relative rank accuracy for each genotype
relative_rank_accuracy = np.sum(confusion_matrix == 1, axis=1) / (len(genotypes) - 1)

# Prepare the data for the table
table_data = {
    'Genotypes': peanut_analysis_df['Genotypes'],
    'Manual estimation - Rank': peanut_analysis_df['Rank_Yield (lbs/A)'],
    'Manual estimation - Yield': peanut_analysis_df['Mean_Yield (lbs/A)'],
    'Algorithm prediction (ours) - Rank': peanut_analysis_df['Rank_Prediction_yield'],
    'Algorithm prediction (ours) - Yield': peanut_analysis_df['Mean_Prediction_yield'],
    'Relative ranking accuracy': relative_rank_accuracy
}

# Create a DataFrame for the table
table_df = pd.DataFrame(table_data)

# Save the DataFrame to a CSV file
output_file_path = 'Y:\zhengkun.li\peanut_project\Analysis_result\LoFTR-stitching\peanut_analysis_accuracy.csv'
table_df.to_csv(output_file_path, index=False)

print("The comparison table has been saved to:", output_file_path)
