import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# Plot the confusion matrix with grid lines and 45-degree rotated genotype labels
plt.figure(figsize=(12, 12))
sns.heatmap(confusion_matrix, annot=False, cmap='RdYlGn', xticklabels=genotypes, yticklabels=genotypes, cbar=False, linewidths=.4, linecolor='black')
plt.title('Confusion Matrix: Predicted vs Actual Yield Rankings')
# plt.xlabel('Genotypes')
# plt.ylabel('Genotypes')
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=45, va='top', fontsize=12)

plt.show()

# Display relative rank accuracy
relative_rank_accuracy_df = pd.DataFrame({
    'Genotype': genotypes,
    'Relative Rank Accuracy': relative_rank_accuracy
})

print(relative_rank_accuracy_df)

# # Display relative rank accuracy
# relative_rank_accuracy_df = pd.DataFrame({
#     'Genotype': genotypes,
#     'Relative Rank Accuracy': relative_rank_accuracy
# })

# import ace_tools as tools; tools.display_dataframe_to_user(name="Relative Rank Accuracy", dataframe=relative_rank_accuracy_df)
