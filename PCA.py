

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#Reading the csv files
df = pd.read_csv("C:/Users/manta/Downloads/LAB7-2022/LAB7-2022/Iris.csv")

#dropping the species column
res = df["Species"]
df = df.drop("Species" , axis = 1)

#Reducing the dimensionality
pca = PCA(n_components=2)
new_df = pca.fit_transform(df)

#getting the covariance matrix
cov1 = pca.get_covariance()
evalue , evect = np.linalg.eig(cov1)

print("Eigen Values : " )

for i in evalue:
    print(i , end =" , ")

#plotting the eigen values
plt.plot([1,2,3,4] , evalue , color = "r")
plt.xlabel("Number")
plt.ylabel("Eigen Values")
plt.title("Eigen Values for the Iris Dataset")
plt.show()

#plotting the reduced data
new_data0 = [i[0] for i in new_df]
new_data1 = [i[1] for i in new_df]

plt.scatter(new_data0 , new_data1 , color = "g")
plt.xlabel("Axis-0")
plt.ylabel("Axis-1")
plt.title("Scatter plot of Reduced dimensional data")
plt.show()
