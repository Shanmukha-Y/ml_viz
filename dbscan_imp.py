import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

n_samples = 1000
X, _ = make_moons(n_samples=n_samples, noise=0.1, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

eps_values = [0.1, 0.2, 0.3, 0.4, 0.5]
fig, axs = plt.subplots(1, len(eps_values), figsize=(20, 4))

for i, eps in enumerate(eps_values):
    dbscan = DBSCAN(eps=eps, min_samples=5)
    labels = dbscan.fit_predict(X_scaled)
    
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
        xy = X_scaled[class_member_mask]
        axs[i].plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)
    
    axs[i].set_title(f'DBSCAN: eps={eps}')
    axs[i].set_xlim(X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5)
    axs[i].set_ylim(X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5)

plt.tight_layout()
plt.show()

for eps in eps_values:
    dbscan = DBSCAN(eps=eps, min_samples=5)
    labels = dbscan.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"eps={eps}: {n_clusters} clusters")