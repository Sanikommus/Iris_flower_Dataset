# Iris_flower_Dataset

Given with Iris flower dataset file. The file Iris.csv consists of 150, 4-dimensional data
which includes 50 samples from each of the three species of Iris (Iris setose, Iris virginica and Iris
versicolor). Column 1 to 4 of the given file are the four features (attributes) that were measured
from each sample: the length and the width of the sepals and petals (in centimetres) respectively.
Column 5 is the class label (species name) associated with each of the samples of Iris flower.

We had to reduce the data into 2-dimensional data using PCA and then partition (cluster) the
reduced dimensional data using different clustering techniques. While performing the PCA, ignore
the target column.

Target Attribute was to be used to calculate the Purity Score.

The code:
* Loads the dataset into the Spyder Enviornment.
* Imports the Unsupervised Learning Models like k-means, GMM and DBSCAN.
* We form Clusters and then we label them for each of the model seperately and check the purity score for each model.


# Input Dataset

https://www.kaggle.com/datasets/uciml/iris

![image](https://user-images.githubusercontent.com/119813195/228898903-07be9a49-b991-468a-808b-6f7c72e7fcc7.png)

# Output 

PCA on the dataset : 

![image](https://user-images.githubusercontent.com/119813195/228899300-0d7386a1-10f8-47b0-ab5e-550d9054db33.png)

![image](https://user-images.githubusercontent.com/119813195/228899425-b52dd998-d6db-4387-b0cf-484e78db0486.png)

k-Means with number of clusters(K) = 3 :

![image](https://user-images.githubusercontent.com/119813195/228899845-db5a2540-77e9-4ee7-a2fc-c69c45f60d99.png)

k-Means for Different values of K :

![image](https://user-images.githubusercontent.com/119813195/228900079-137c4d46-95a3-4fc6-a1e3-868308ff977c.png)

![image](https://user-images.githubusercontent.com/119813195/228900810-879e29cb-a366-4825-abd2-9586bab612b5.png)

GMM with number of clusters(K) = 3 :

![image](https://user-images.githubusercontent.com/119813195/228901083-468383a2-ca69-4633-acd6-a90d4a691357.png)

GMM with different values of K :

![image](https://user-images.githubusercontent.com/119813195/228901412-aca0dee6-9826-437f-8d95-83eb7c3edf51.png)

![image](https://user-images.githubusercontent.com/119813195/228901503-6402eac7-961d-4a77-80a4-e4ae551a8dd5.png)

DBSCAN :

![image](https://user-images.githubusercontent.com/119813195/228902087-5bb52b45-3c7f-4314-b7ae-c9a4d090dfac.png)

![image](https://user-images.githubusercontent.com/119813195/228902228-eb38fc10-c43f-4674-9480-33eaee6de234.png)





