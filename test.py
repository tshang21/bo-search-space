import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors

def load(function_name, dim, bo):
    X = np.load(f"results_{bo}/{function_name}_{dim}d/X.npy")
    fX = np.load(f"results_{bo}/{function_name}_{dim}d/fX.npy")
    return X, fX

def pairwise_dist(X):
    avg_dist = np.mean(pdist(X))
    return avg_dist

def knn_distance(X, k=5):
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    dists, _ = nbrs.kneighbors(X)
    return np.mean(dists[:, 1:])

def compare_pca(function_name, dim):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    bos = ["turbo", "saasbo"]
    pca = PCA(n_components=2)

    X_turbo, fX_turbo = load(function_name, dim, "turbo")
    X_proj_turbo = pca.fit_transform(X_turbo)

    X_saasbo, fX_saasbo = load(function_name, dim, "saasbo")
    X_proj_saasbo = pca.fit_transform(X_saasbo)

    X_proj = [X_proj_turbo, X_proj_saasbo]
    fX = [fX_turbo, fX_saasbo]

    x_min = min(X_proj[0][:, 0].min(), X_proj[1][:, 0].min())
    x_max = max(X_proj[0][:, 0].max(), X_proj[1][:, 0].max())
    y_min = min(X_proj[0][:, 1].min(), X_proj[1][:, 1].min())
    y_max = max(X_proj[0][:, 1].max(), X_proj[1][:, 1].max())

    for i, bo in enumerate(bos):
        ax = axes[i]
        scatter = ax.scatter(X_proj[i][:, 0], X_proj[i][:, 1], c=fX[i], cmap="viridis", s=30)
        ax.set_title(f"{bo.upper()} — {function_name} {dim}D\n")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_xlim(x_min-0.2, x_max+0.2)
        ax.set_ylim(y_min-0.2, y_max+0.2)
        fig.colorbar(scatter, ax=ax, label="f(x)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{function_name}_{dim}d_comparison.png", dpi=300)
    plt.close()

def plot_avg_pairwise(function_name, dims):
    turbo_dists = []
    saasbo_dists = []

    for dim in dims:
        for bo, dist_list in zip(["turbo", "saasbo"], [turbo_dists, saasbo_dists]):
            X, _ = load(function_name, dim, bo)
            d = pdist(X)
            dist_list.append(np.mean(d))

    plt.figure(figsize=(8, 5))
    plt.plot(dims, turbo_dists, marker='o', label='TuRBO')
    plt.plot(dims, saasbo_dists, marker='s', label='SAASBO')
    plt.xlabel("Dimension")
    plt.ylabel("Average Pairwise Distance")
    plt.title(f"Average Pairwise Distance vs Dimension: {function_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{function_name}_avg_pairwise_vs_dim.png", dpi=300)
    plt.close()

def pairwise_histogram(function_name, dim):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    bos = ["turbo", "saasbo"]
    all_heights = []
    all_xmax = []

    for bo in bos:
        X, _ = load(function_name, dim, bo)
        dists = pdist(X)
        counts, _ = np.histogram(dists, bins=30)
        all_heights.append(np.max(counts))
        all_xmax.append(np.max(dists))

    y_max = max(all_heights)
    x_max = max(all_xmax)

    for i, bo in enumerate(bos):
        X, _ = load(function_name, dim, bo)
        dists = pdist(X)

        ax = axes[i]
        ax.hist(dists, bins=30, color="skyblue", edgecolor="black")
        ax.set_title(f"{bo.upper()} — {function_name} {dim}D\n"
                    f"average pairwise distance={dists[i]:.3f}")
        ax.set_xlabel("Pairwise Distance")
        ax.set_ylabel("Frequency")
        ax.set_ylim(0, y_max+500)
        ax.set_xlim(0, x_max+0.2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{function_name}_{dim}d_pairwise_hist.png", dpi=300)
    plt.close()

def plot_bestfx(functions, dims):
    for func in functions:
        turbo_fx, saas_fx = [], []

        for dim in dims:
            _, fX_turbo = load(func, dim, 'turbo')
            _, fX_saasbo = load(func, dim, 'saasbo')
            turbo_fx.append(np.min(fX_turbo))
            saas_fx.append(np.min(fX_saasbo))

        plt.figure(figsize=(8, 5))
        plt.plot(dims, turbo_fx, marker='o', label='TuRBO')
        plt.plot(dims, saas_fx, marker='s', label='SAASBO')
        plt.xlabel('Dimension')
        plt.ylabel('Best f(x)')
        plt.title(f'Best f(x) vs Dimension: {func}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'best_fx_{func}.png', dpi=300)
    

def plot_knn(functions, dims):
    for func in functions:
        turbo_knn, saas_knn = [], []

        for dim in dims:
            X_turbo, _ = load(func, dim, 'turbo')
            X_saasbo, _ = load(func, dim, 'saasbo')

            turbo_knn.append(knn_distance(X_turbo))
            saas_knn.append(knn_distance(X_saasbo))
        
        plt.figure(figsize=(8, 5))
        plt.plot(dims, turbo_knn, marker='o', label='TuRBO')
        plt.plot(dims, saas_knn, marker='s', label='SAASBO')
        plt.xlabel('Dimension')
        plt.ylabel('Avg KNN Distance')
        plt.title(f'KNN Distance vs Dimension: {func}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'knn_distance_{func}.png', dpi=300)

if __name__ == "__main__":
    function_names = ["ackley", "rastrigin"]
    dims = [5, 10, 20, 30, 50, 75, 100, 125, 150]

    for function_name in function_names:
        plot_avg_pairwise(function_name, dims)
        for dim in dims:
            compare_pca(function_name, dim)
            pairwise_histogram(function_name, dim)
    
    plot_bestfx(function_names, dims)
    plot_knn(function_names, dims)