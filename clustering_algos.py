import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

n_samples = 1000
X, _ = make_moons(n_samples=n_samples, noise=0.1, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Set up the plot
fig, axs = plt.subplots(2, 2, figsize=(15, 15))
axs = axs.ravel()

# Function to plot decision boundaries for KMeans, GMM, and Agglomerative
def plot_decision_boundary(ax, model, X, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Spectral)
    ax.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap=plt.cm.Spectral)
    ax.set_title(title)

# Function to plot DBSCAN results from Scikit site
def plot_dbscan(ax, dbscan, X, title):
    labels = dbscan.labels_
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]  # Black for noise

        class_member_mask = (labels == k)

        # Plot core samples
        xy = X[class_member_mask & core_samples_mask]
        ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=10)

        # Plot non-core samples
        xy = X[class_member_mask & ~core_samples_mask]
        ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6)
    
    ax.set_title(title)

# K-Means
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_scaled)
plot_decision_boundary(axs[0], kmeans, X_scaled, 'K-Means')

# DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)
dbscan.fit(X_scaled)
plot_dbscan(axs[1], dbscan, X_scaled, 'DBSCAN')

# Gaussian Mixture Model
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(X_scaled)
gmm.labels_ = gmm.predict(X_scaled)
plot_decision_boundary(axs[2], gmm, X_scaled, 'Gaussian Mixture Model')

# Agglomerative Clustering
agglomerative = AgglomerativeClustering(n_clusters=2)
agglomerative.fit(X_scaled)
# Use KNeighborsClassifier to approximate decision boundary
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_scaled, agglomerative.labels_)
knn.labels_ = knn.predict(X_scaled)
plot_decision_boundary(axs[3], knn, X_scaled, 'Agglomerative Clustering')

plt.tight_layout()
plt.show()