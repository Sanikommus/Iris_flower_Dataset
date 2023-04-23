

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
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

#using the GMM model for K = 3 value
K = 3
gmm = GaussianMixture(n_components = K)

#training the model and predicting the value
gmm.fit(new_df)
labels = gmm.predict(new_df)

new_df_1 = pd.DataFrame(new_df.copy())
new_df_1.insert(2, "Labels", labels , True)

by_class = new_df_1.groupby('Labels')
d0 = by_class.get_group(0)
d1 = by_class.get_group(1)
d2 = by_class.get_group(2)

#plotting the cluster with different colours
plt.scatter(d0[0] , d0[1] , color = "red")
plt.scatter(d1[0] , d1[1] , color = "orange")
plt.scatter(d2[0] , d2[1] , color = "green")

#plotting the centre point of each cluster
plt.scatter(d0[0].mean() , d0[1].mean() , color = "black" , marker = "*" , s = 100)
plt.scatter(d1[0].mean() , d1[1].mean() , color = "black" , marker = "*" , s = 100)
plt.scatter(d2[0].mean() , d2[1].mean() , color = "black" , marker = "*" , s = 100)

plt.xlabel("Axis-0")
plt.ylabel("Axis-1")
plt.title("Scatter plot after Clustering Using GMM and K=3")

plt.show()

#claculating the distotion measure
distortion = gmm.score(new_df)*len(new_df)
print("The Distortion Measure is: " + str(distortion))

#finding the purity score
contingency_matrix=metrics.cluster.contingency_matrix(res , labels)
row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
purity = contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix)

print("The purity score of the data is : " + str(purity))
