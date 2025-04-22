import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
# Step 1: Load the California Housing Dataset
data = fetch_california_housing(as_frame=True)
housing_df = data.frame
# Step 2: Compute the correlation matrix
correlation_matrix = housing_df.corr()
# Step 3: Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix,annot=True,linewidths=0.5)
plt.title('Correlation Matrix of California Housing Features')
plt.show()
# Step 4: Create a pair plot to visualize pairwise relationships
sns.pairplot(housing_df, diag_kind='kde')

plt.show()