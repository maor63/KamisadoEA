from sklearn.cluster import KMeans
from itertools import permutations
import numpy as np

X = np.array(list(permutations(list(range(8)))))
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
print(kmeans.cluster_centers_)