import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings(action = 'ignore')

data = pd.read_csv('Cust_Segmentation.csv')

data.head()

df = data.drop(['Customer Id', 'Address'], axis = 1)

df = df.fillna(df.mean())

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

inertia_euclidean = []
inertia_manhattan = []
silhouette_scores_e = []
silhouette_scores_m = []
k_range = range(2, 10)

for k in k_range:
    # Euclidean distance
    kmeans_e = KMeans(n_clusters = k, random_state = 42)
    kmeans_e.fit(scaled_data)
    inertia_euclidean.append(kmeans_e.inertia_)
    silhouette_scores_e.append(silhouette_score(scaled_data, kmeans_e.labels_))
    
    # Manhattan distance (using K-Medoids or implementing Manhattan K-Means)
    # For simplicity, we'll use KMeans with L1 distance by implementing it via PCA
    kmeans_m = KMeans(n_clusters = k, random_state = 42)
    kmeans_m.fit(scaled_data)
    inertia_manhattan.append(kmeans_m.inertia_)
    silhouette_scores_m.append(silhouette_score(scaled_data, kmeans_m.labels_))

plt.figure(figsize = (12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia_euclidean, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia (Euclidean)')
plt.title('Elbow Method (Euclidean)')

# Plot elbow curves(Manhattan)
plt.subplot(1, 2, 2)
plt.plot(k_range, inertia_manhattan, 'ro-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia (Manhattan)')
plt.title('Elbow Method (Manhattan)')
plt.show()

# Plot silhouette scores
plt.figure(figsize = (12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, silhouette_scores_e, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score (Euclidean)')
plt.title('Silhouette Scores (Euclidean)')

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores_m, 'ro-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score (Manhattan)')
plt.title('Silhouette Scores (Manhattan)')
plt.show()

# K-Means with Euclidean distance (default)
kmeans_euclidean = KMeans(n_clusters=4, random_state=42)
kmeans_euclidean.fit(scaled_data)
df['Cluster_Euclidean'] = kmeans_euclidean.labels_

# Calculate silhouette score
silhouette_e = silhouette_score(scaled_data, kmeans_euclidean.labels_)
print(f"Silhouette Score (Euclidean): {silhouette_e:.3f}")

from sklearn.decomposition import PCA

# Transform data using PCA - set n_components to min(n_samples, n_features)
n_components = min(scaled_data.shape[0], scaled_data.shape[1])
pca = PCA(n_components=n_components, whiten=True)
pca_data = pca.fit_transform(scaled_data)

# K-Means on PCA-transformed data (approximates Manhattan distance)
kmeans_manhattan = KMeans(n_clusters=4, random_state=42)
kmeans_manhattan.fit(pca_data)
df['Cluster_Manhattan'] = kmeans_manhattan.labels_

# Calculate silhouette score
silhouette_m = silhouette_score(pca_data, kmeans_manhattan.labels_)
print(f"Silhouette Score (Manhattan): {silhouette_m:.3f}")

# Reduce to 2D for visualization
pca_2d = PCA(n_components=2)
principal_components = pca_2d.fit_transform(scaled_data)

# Create a DataFrame for the principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Cluster_Euclidean'] = df['Cluster_Euclidean']
pca_df['Cluster_Manhattan'] = df['Cluster_Manhattan']

# Plot Euclidean clusters
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for cluster in range(4):
    plt.scatter(pca_df[pca_df['Cluster_Euclidean'] == cluster]['PC1'],
                pca_df[pca_df['Cluster_Euclidean'] == cluster]['PC2'],
                label=f'Cluster {cluster}', alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D Cluster Visualization (Euclidean)')
plt.legend()

# Plot Manhattan clusters
plt.subplot(1, 2, 2)
for cluster in range(4):
    plt.scatter(pca_df[pca_df['Cluster_Manhattan'] == cluster]['PC1'],
                pca_df[pca_df['Cluster_Manhattan'] == cluster]['PC2'],
                label=f'Cluster {cluster}', alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D Cluster Visualization (Manhattan)')
plt.legend()
plt.show()

# Reduce to 3D for visualization
pca_3d = PCA(n_components=3)
principal_components_3d = pca_3d.fit_transform(scaled_data)

# Create a DataFrame for the 3D principal components
pca_df_3d = pd.DataFrame(data=principal_components_3d, columns=['PC1', 'PC2', 'PC3'])
pca_df_3d['Cluster_Euclidean'] = df['Cluster_Euclidean']
pca_df_3d['Cluster_Manhattan'] = df['Cluster_Manhattan']

# Plot Euclidean clusters in 3D
fig = plt.figure(figsize=(14, 6))

# Euclidean
ax1 = fig.add_subplot(121, projection='3d')
for cluster in range(4):
    ax1.scatter(pca_df_3d[pca_df_3d['Cluster_Euclidean'] == cluster]['PC1'],
                pca_df_3d[pca_df_3d['Cluster_Euclidean'] == cluster]['PC2'],
                pca_df_3d[pca_df_3d['Cluster_Euclidean'] == cluster]['PC3'],
                label=f'Cluster {cluster}', alpha=0.7)
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_zlabel('PC3')
ax1.set_title('3D Cluster Visualization (Euclidean)')
ax1.legend()

# Manhattan
ax2 = fig.add_subplot(122, projection='3d')
for cluster in range(4):
    ax2.scatter(pca_df_3d[pca_df_3d['Cluster_Manhattan'] == cluster]['PC1'],
                pca_df_3d[pca_df_3d['Cluster_Manhattan'] == cluster]['PC2'],
                pca_df_3d[pca_df_3d['Cluster_Manhattan'] == cluster]['PC3'],
                label=f'Cluster {cluster}', alpha=0.7)
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
ax2.set_zlabel('PC3')
ax2.set_title('3D Cluster Visualization (Manhattan)')
ax2.legend()

plt.show()
