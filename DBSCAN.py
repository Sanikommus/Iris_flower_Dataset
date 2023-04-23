

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.optimize import linear_sum_assignment

#Reading the csv files
df = pd.read_csv("C:/Users/manta/Downloads/LAB7-2022/LAB7-2022/Iris.csv")

#dropping the species column
res = df["Species"]
df = df.drop("Species" , axis = 1)

#Reducing the dimensionality
pca = PCA(n_components=2)
new_df = pca.fit_transform(df)

#different combinations of epsilon and minimum number of sample
cob = [[1,4] , [1,10] , [5,4] , [5,10]]
purity_score = []

#using different DBSCAN model for different combinations of epsilon and minimum number of samples
for i in cob:
    #training the model and predicting the value
    dbscan_model=DBSCAN(eps=i[0], min_samples=i[1]).fit(new_df)
    labels = dbscan_model.labels_
    
    #plotting the cluster with different colours
    plt.scatter(new_df[:, 0] , new_df[:, 1], c = labels )
    plt.title(f"scatter plot of DBSCAN for {i[0]} epsilon and {i[1]} min samples" )
    plt.xlabel("Axis-1")
    plt.ylabel("Axis-2")
    plt.show()
    
    #calculating the purity score for different combination of epsilon and minimum number of samples and storing it in a list
    confusion_matrix = metrics.cluster.contingency_matrix(res , labels )
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
    purity = confusion_matrix[row_ind, col_ind].sum() / np.sum(confusion_matrix)
    print("purity score for the {i[0]} epsilon and {i[1]} min samples cluster is: ", purity)
    
    purity_score.append(purity)
