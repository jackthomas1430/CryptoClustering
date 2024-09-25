# CryptoClustering

## Overview

For this project, we perform clustering analysis on cryptocurrencies using unsupervised learning to identify if cryptocurrencies are affected by short-term price changes (24-hour and 7-day periods) and how they can be grouped based on their price change behaviors.

We use clustering algorithms like K-Means and dimensionality reduction techniques such as Principal Component Analysis (PCA) to optimize the clusters. Ultimately we create visual comparison of clustering results using the original data and the PCA-reduced data.

## Files
    - Crypto_Clustering<https://github.com/jackthomas1430/CryptoClustering.git> 
    -Crypto_Clustering.ipynb: Jupyter notebook containing all code for the analysis and clustering process.
    -crypto_market_data.csv: Dataset with historical cryptocurrency data including price change percentages over various time frames.

## Objective
1. Load the Data
    The dataset (crypto_market_data.csv) is loaded into a Pandas DataFrame and exploratory analysis is performed to understand the structure of the data. 
2. Prepare the Data
    Data is normalized using the StandardScaler from sklearn and a new DataFrame is created with the scaled data and the original coin_id as the index.
3. Find the Best Value for k (Elbow Method) Using the Scaled DataFrame
    The elbow method is applied to determine the optimal number of clusters (k).
    A list of k values ranging from 1 to 11 is created.
    For each k, the inertia  is calculated.
    A line chart is generated to visualize the elbow curve and identify the best value for k. 
4. Cluster Cryptocurrencies Using the Scaled DataFrame
    The K-Means algorithm is used to cluster the cryptocurrencies based on the scaled data and the best value for k.
    A scatter plot is created using hvPlot to visualize the clustering of cryptocurrencies based on their 24-hour and 7-day price change percentages.
5. Optimize Clusters with Principal Component Analysis (PCA)
    Principal Component Analysis (PCA) is performed on the scaled DataFrame.
    The explained variance of the three components is calculated
6. Find the Best Value for k Using the PCA-Reduced Data
    The elbow method is repeated on the PCA DataFrame to determine the best value of k.
    
7. Cluster Cryptocurrencies Using the PCA DataFrame
    K-Means clustering is applied to the PCA-reduced data using the optimal value of k.
    A scatter plot is generated using hvPlot to visualize the clusters with PCA1 and PCA2 as the axes.
8. Visualize and Compare Results
    Two composite plots are created to compare the results of both elbow curves and cluster plots and to show the impact of using fewer features with the PCA data. 
   
##Key Findings

Best Value for k:
    The best value for k is 4 for both the original and PCA-reduced data.

Impact of PCA on Clustering:
    By reducing the dimensionality of the data using PCA, the clustering process became more efficient while retaining most of the important features.The cluster patterns show that the PCA reduction seemingly eliminates noise and focuses on the most significant aspects of the data, but may miss out on some of the finer details as the there is less separation between clusters. 
## Setup
    ### Dependencies
            Python 3.x
            Jupyter Notebook
    ### Required Libraries:
            pandas
            hvplot
            sklearn
            matplotlib

## Instructions to Run
    1.  Clone the repository: git clone <https://github.com/jackthomas1430/CryptoClustering.git>
    2. Open and run the Crypto_Clustering.ipynb notebook to execute the analysis and view the results.

## File Structure
    -Crypto_Clustering.ipynb: The main Jupyter notebook containing the code and outputs.
    -crypto_market_data.csv: Dataset used in the project.
    -README.md: Detailed instructions and overview of the project.
   

##Acknowledgements
    
    Xpert Learning Assistant was used to answer detailed questions, and assist in debugging.The starter code provided was the base of the report and was modified using course curriculum and activities to fit the requirements of the assignment. The TA and instructor for the course also assisted in adjusting the code during office hours.For more information about the Xpert Learning Assistant, visit [EdX Xpert Learning Assistant](https://www.edx.org/). 

## References
sklearn.decomposition.PCA - Documentation for Principal Component Analysis (PCA) from scikit-learn.
sklearn.cluster.AgglomerativeClustering - Information on Agglomerative Clustering provided by scikit-learn.
Clustering Evaluation Metrics - Overview of clustering algorithms and evaluation metrics like the Calinski-Harabasz Index.
pandas.get_dummies - Pandas function for converting categorical variables into dummy/indicator variables.
sklearn.preprocessing.StandardScaler - Documentation for the StandardScaler used for feature scaling.
sklearn.cluster.KMeans - Documentation for the KMeans clustering algorithm from scikit-learn.
sklearn.preprocessing.LabelEncoder - Information on the LabelEncoder used for encoding labels with values between 0 and n_classes-1.
