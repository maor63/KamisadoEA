from sklearn.cluster import KMeans, AgglomerativeClustering
from itertools import permutations
import numpy as np

X = np.array(list(permutations(list(range(8)))))
kmeans = AgglomerativeClustering(n_clusters=3).fit(X)
print(kmeans.cluster_centers_)