import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Load IRIS dataset
df = pd.read_csv("Datasets/IRIS.csv")
df = df.drop(columns="species").values

K = 4

# Initial model (first random centroids)
kmeans = KMeans(n_clusters=K, init='random', n_init=1, max_iter=1, random_state=42)
kmeans.fit(df)

centroids = kmeans.cluster_centers_

print("Iteration 1 cluster sizes:")
labels = kmeans.labels_
for i in range(K):
    print(f"Cluster {i+1}: {sum(labels == i)} points")

# Run 9 more iterations (total 10)
for it in range(2, 11):
    kmeans = KMeans(
        n_clusters=K,
        init=centroids,   # use previous iteration’s centroids
        n_init=1,
        max_iter=1
    )
    kmeans.fit(df)

    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    print(f"\nIteration {it} cluster sizes:")
    for i in range(K):
        print(f"Cluster {i+1}: {sum(labels == i)} points")

# Final centroids
print("\nFinal cluster means after 10 iterations:")
for i, c in enumerate(centroids, start=1):
    print(f"Cluster {i}: {c}")
