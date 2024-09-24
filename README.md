# CryptoClustering
## Overview
In this project, we perform a clustering analysis on cryptocurrencies using unsupervised learning methods. The goal is to identify if cryptocurrencies are affected by short-term price changes (24-hour and 7-day periods) and how they can be grouped based on their price change behaviors.

We will leverage clustering algorithms like K-Means and dimensionality reduction techniques such as Principal Component Analysis (PCA) to optimize the clusters. The project will culminate in a visual comparison of clustering results using the original data and the PCA-reduced data.

## Files
Crypto_Clustering.ipynb: Jupyter notebook containing all code for the analysis and clustering process.
crypto_market_data.csv: Dataset with historical cryptocurrency data including price change percentages over various time frames.
## Instructions
1. Load the Data
The dataset (crypto_market_data.csv) is loaded into a Pandas DataFrame.
Initial exploratory analysis is performed to understand the structure of the data using summary statistics and visualizations.
2. Prepare the Data
Data is normalized using the StandardScaler from sklearn to standardize features by removing the mean and scaling to unit variance.
A new DataFrame is created with the scaled data while retaining the original coin_id as the index.
3. Find the Best Value for k (Elbow Method) Using the Scaled DataFrame
The elbow method is applied to determine the optimal number of clusters (k).
A list of k values ranging from 1 to 11 is created.
For each k, the inertia (sum of squared distances to the nearest cluster center) is calculated.
A line chart is generated to visualize the elbow curve and identify the optimal k (the point where the curve bends).
The best value for k is determined based on the elbow curve.
4. Cluster Cryptocurrencies Using the Scaled DataFrame
The K-Means algorithm is used to cluster the cryptocurrencies based on the scaled data and the best value for k.
A new DataFrame is created to store the cluster predictions.
A scatter plot is created using hvPlot to visualize the clustering of cryptocurrencies based on their 24-hour and 7-day price change percentages.
5. Optimize Clusters with Principal Component Analysis (PCA)
Principal Component Analysis (PCA) is performed on the scaled DataFrame to reduce the features to three principal components.
The explained variance of the three components is calculated to assess how much of the dataset's information is retained after dimensionality reduction.
A new DataFrame is created with the PCA-transformed data, with the original coin_id index retained.
6. Find the Best Value for k Using the PCA-Reduced Data
The elbow method is repeated on the PCA-transformed DataFrame to determine the optimal value of k.
The elbow curve is visualized using a line chart to identify the optimal value of k.
7. Cluster Cryptocurrencies Using the PCA DataFrame
K-Means clustering is applied to the PCA-reduced data using the optimal k.
A scatter plot is generated using hvPlot to visualize the clusters in the PCA-reduced feature space, with PC1 and PC2 as the axes.
8. Visualize and Compare Results
Two composite plots are created to compare the results:
One plot contrasts the elbow curves for the original and PCA-reduced data.
Another plot contrasts the clusters generated using the original data versus the PCA-reduced data.
The impact of using fewer features (PCA) on the clustering results is analyzed.
Key Findings
Best Value for k (Original Data and PCA-Reduced Data):

The elbow method suggests that the optimal value for k is 4 for both the original and PCA-reduced data.
This suggests that using PCA did not alter the best value for k.
Impact of PCA on Clustering:

By reducing the dimensionality of the data using PCA, the clustering process became more efficient while retaining most of the important features.
The cluster patterns in the PCA-reduced data are visually simpler, indicating that fewer features can still yield meaningful clustering.
Setup
Dependencies
Python 3.x
Jupyter Notebook
Required Libraries:
pandas
hvplot
sklearn
matplotlib
Instructions to Run
Clone the repository:
bash
Copy code
git clone <your-repo-url>
Navigate to the project directory:
bash
Copy code
cd CryptoClustering
Install the necessary dependencies (if using a virtual environment, activate it first):
bash
Copy code
pip install -r requirements.txt
Launch the Jupyter Notebook:
bash
Copy code
jupyter notebook
Open and run the Crypto_Clustering.ipynb notebook to execute the analysis and view the results.
Project Structure
Crypto_Clustering.ipynb: The main Jupyter notebook containing the code and outputs.
crypto_market_data.csv: Dataset used in the project.
README.md: Detailed instructions and overview of the project.
requirements.txt: List of dependencies for running the project.

