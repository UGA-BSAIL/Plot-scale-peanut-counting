import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the provided Excel file
file_path = r'Y:\zhengkun.li\peanut_project\Analysis_result\LoFTR-stitching\peanut_analysis_tifton_modified.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Melt the dataframe to have a single column for the yields (predicted and actual)
melted_df = df.melt(id_vars=['Name1'], value_vars=['Prediction_yield', 'Yield (lbs/A)'],
                    var_name='Yield Type', value_name='Yield')

# Sort the dataframe by the mean of 'Yield (lbs/A)' for each genotype
sorted_genotypes = df.groupby('Name1')['Yield (lbs/A)'].mean().sort_values().index

# Update the melted dataframe to have the genotypes sorted by the mean yield
melted_df['Name1'] = pd.Categorical(melted_df['Name1'], categories=sorted_genotypes, ordered=True)

# Create the boxplot with dashed lines for whiskers
plt.figure(figsize=(14, 6))
sns.boxplot(x='Name1', y='Yield', hue='Yield Type', data=melted_df,
            boxprops=dict(linewidth=1.5),
            whiskerprops=dict(linestyle='--', linewidth=1.5),
            capprops=dict(linewidth=1.5),
            medianprops=dict(linewidth=2.5))

plt.xticks(rotation=45) # 45 degree rotation
plt.xlabel('Genotypes')
plt.ylabel('Yield(lbs/A)')
plt.title('Boxplot of Prediction Yield and Actual Yield for Different Genotypes (Sorted)')
# plt.legend(title='Yield Type')
plt.legend()
plt.tight_layout()

# Display the plot
plt.show()
