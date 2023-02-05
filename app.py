import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import datetime as dt
from sklearn.cluster import KMeans

df = pd.read_excel(r'kel08_dataset.xlsx', engine='openpyxl')
df

df_null = round(100*(df.isnull().sum())/len(df), 2)
df_null

plt.scatter(df['nama_provinsi'], df['penduduk_miskin'])
plt.xlabel('nama_provinsi')
plt.ylabel('penduduk_miskin')

df.info()

df_null = round(100*(df.isnull().sum())/len(df), 2)
df_null

df = df.dropna()
df.shape

plt.scatter(df['nama_provinsi'],df['penduduk_miskin'])
plt.xlim(0,5)
plt.ylim(0,40)
plt.show()

x =df.iloc[0:,4:5]
x

X = df[['id', 'penduduk_miskin']].copy()

distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(X)
    distortions.append(kmeanModel.inertia_)

plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

kmeans = KMeans(4)
kmeans.fit(X)
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0).fit(X)

kmeans.labels_
kmeans.inertia_
kmeans.n_iter_
kmeans.cluster_centers_

from collections import Counter
Counter(kmeans.labels_)
# output
Counter({2: 50, 0: 50, 3: 50, 1: 50})

data_with_clusters = df.copy()
data_with_clusters['Clusters'] = kmeans.labels_
plt.scatter(data_with_clusters['penduduk_miskin'],data_with_clusters['nama_provinsi'],c=data_with_clusters['Clusters'],cmap='rainbow')

#teknik clustering GMM(GaussianMixture)
from sklearn.mixture import GaussianMixture
from sklearn.mixture import GaussianMixture
n_clusters = 5
gmm_model = GaussianMixture(n_components=n_clusters)
gmm_model.fit(X)
cluster_labels = gmm_model.predict(X)
X = pd.DataFrame(X)
X['cluster'] = cluster_labels
for k in range(0,n_clusters):
    data = X[X["cluster"]==k]
    plt.scatter(data["id"],data["penduduk_miskin"])
    plt.title("Clusters Identified by Guassian Mixture Model")    
plt.ylabel("Penduduk Miskin")
plt.xlabel("Id Provinsi")
plt.show()

#Teknik Clustering Spectral Clustering
from sklearn.cluster import SpectralClustering
spectral_cluster_model= SpectralClustering(
    n_clusters=5, 
    random_state=25, 
    n_neighbors=8, 
    affinity='nearest_neighbors'
)
X['cluster'] = spectral_cluster_model.fit_predict(X[['id', 'penduduk_miskin']])
fig, ax = plt.subplots()
sns.scatterplot(x='id', y='penduduk_miskin', data=X, hue='cluster', ax=ax)
ax.set(title='Spectral Clustering')
plt.xlabel("Id Provinsi")

import pickle
pickle.dump(kmeans, open('cluster.pkl','wb'))