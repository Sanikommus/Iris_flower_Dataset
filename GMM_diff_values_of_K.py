

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

#using GMM model for different values of K
K = [2,3,4,5,6,7]
distortion_measure = []
purity_score = []

for i in K:
    gmm = GaussianMixture(n_components = i)
    
    #training the model and predicting the value
    gmm.fit(new_df)
    labels = gmm.predict(new_df)
    
    #claculating the distotion measure and storing it in a list for different values of K
    distortion = gmm.score(new_df)*len(new_df)
    distortion_measure.append(distortion)
    
    #finding the purity score and storinh it in a list for different values of K
    contingency_matrix=metrics.cluster.contingency_matrix(res , labels)
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    purity = contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix)
   
    purity_score.append(purity)
    
    print("K = " + str(i))
    print("The Distortion Measure for is : " + str(distortion))
    print("The purity score for is       : " + str(purity))
    print()
    print()
    
#plotting the distortion measure for Different values of K
plt.plot(K , distortion_measure)
plt.scatter(K , distortion_measure , marker = "*" , s = 100 , color = "Black")
plt.scatter(K[1] , distortion_measure[1] , marker = "*" , s = 200 , color = "Red")
plt.xlabel("K Values")
plt.ylabel("Distortion Measure")
plt.title("Elbow method to choose the value of K")
plt.show()

print("The value K for the optimum purity score is : " + str(K[purity_score.index(max(purity_score))]))

    



