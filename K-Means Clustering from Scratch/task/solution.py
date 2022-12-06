import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.spatial import distance
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler


# scroll down to the bottom to implement your solution


def plot_comparison(data: np.ndarray, predicted_clusters: np.ndarray, true_clusters: np.ndarray = None,
                    centers: np.ndarray = None, show: bool = True):
    # Use this function to visualize the results on Stage 6.

    if true_clusters is not None:
        plt.figure(figsize=(20, 10))

        plt.subplot(1, 2, 1)
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=predicted_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.grid()

        plt.subplot(1, 2, 2)
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=true_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.grid()
    else:
        plt.figure(figsize=(10, 10))
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=predicted_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.grid()

    plt.savefig('Visualization.png', bbox_inches='tight')
    if show:
        plt.show()


if __name__ == '__main__':

    # Load data
    data = load_wine(as_frame=True, return_X_y=True)
    X_full, y_full = data

    # Permutate it to make things more interesting
    rnd = np.random.RandomState(42)
    permutations = rnd.permutation(len(X_full))
    X_full = X_full.iloc[permutations]
    y_full = y_full.iloc[permutations]

    # From dataframe to ndarray
    X_full = X_full.values
    y_full = y_full.values

    # Scale data
    scaler = MinMaxScaler()
    X_full = scaler.fit_transform(X_full)

    # write your code here
    """Stage 1"""

    centers = X_full[:3, :]
    vectors = X_full[-10:, :]


    def find_nearest(centers_, vectors_):
        """Returns the nearest center for vectors (the cluster of a vector)"""
        clusters = []
        for v in vectors_:
            distances = []
            for c in centers_:
                distances.append(distance.euclidean(c, v))
            min_value = min(distances)
            index = distances.index(min_value)
            clusters.append(index)
        return clusters


    # print(find_nearest(centers, vectors))

    """Stage 2"""


    def calculate_new_centers(centers_, vectors_):
        """Calculates new centers for clusters"""
        cluster_of_vectors = find_nearest(centers_, vectors_)
        unique_cluster_ids = set(cluster_of_vectors)
        clusters = {}
        for u in unique_cluster_ids:
            clusters[u] = []
            for cluster, vector in zip(cluster_of_vectors, vectors_):
                if u == cluster:
                    clusters[u].append(list(vector))
        new_centers = []
        for u in unique_cluster_ids:
            means_of_classes = []
            cluster = np.array(clusters[u])
            for n in range(len(cluster[0])):
                mean_of_class = np.mean(cluster[:, n])
                means_of_classes.append(mean_of_class)
            new_centers.append(means_of_classes)
        return np.array(new_centers)


    # print(calculate_new_centers(centers, X_full).flatten())

    """Stage 3"""


    class CustomKMeans:
        def __init__(self, k=1):
            self.k = k
            self.centers = None

        def fit(self, X, eps=1e-6):
            """Calculates new centers for clusters"""
            self.centers = X[:self.k, :]  # initial centers

            while True:
                cluster_of_vectors = self.predict(X)  # finds clusters with initial centers
                unique_cluster_ids = set(cluster_of_vectors)
                clusters = {}
                for u in unique_cluster_ids:
                    clusters[u] = []
                    for cluster, vector in zip(cluster_of_vectors, X):
                        if u == cluster:
                            clusters[u].append(list(vector))
                new_centers = []
                for u in unique_cluster_ids:
                    means_of_classes = []
                    cluster = np.array(clusters[u])
                    for n in range(len(cluster[0])):
                        mean_of_class = np.mean(cluster[:, n])
                        means_of_classes.append(mean_of_class)
                    new_centers.append(means_of_classes)

                # checking if new center is close to old center, if not we continue iteration
                # and calculate new centers
                is_close = []
                for c1, c2 in zip(new_centers, self.centers):
                    dist = distance.euclidean(c1, c2)
                    is_close.append(dist < eps)
                self.centers = new_centers
                if all(is_close): break

        def predict(self, X):
            clusters = []
            for v in X:
                distances = []
                for c in self.centers:
                    distances.append(distance.euclidean(c, v))
                min_value = min(distances)
                index = distances.index(min_value)
                clusters.append(index)
            return clusters  # returns clusters for new centers

        def calc_error(self, X_data, clusters):
            errors = {index: [] for index, center in enumerate(self.centers)}
            for vector, cluster in zip(X_data, clusters):
                distance_ = vector - self.centers[cluster]
                errors[cluster].append(distance_)

            norm_errors = []
            for error in errors.values():
                norm_errors.append(np.linalg.norm(error))

            return sum(norm_errors) / len(self.centers)

        def find_appropriate_k(self, X, p=0.2):
            error_k_prev = float('inf')
            k = 1
            while k < 1000:
                self.k = k
                self.fit(X)
                predict = self.predict(X)
                error_k = self.calc_error(X, predict)
                if k > 1:
                    p = abs(error_k_prev - error_k) / error_k_prev
                    if p < 0.2:
                        self.k = k - 1
                        return k - 1
                error_k_prev = error_k
                k += 1


    # c = CustomKMeans(2)
    # c.fit(X_full)
    # print(c.predict(X_full[:10, :]))

    """Stage 4"""

    # error_list = []
    # for k in range(1, 100):
    #     c = CustomKMeans(k)
    #     c.fit(X_full)
    #     predict = c.predict(X_full)
    #     r = c.calc_error(X_full, predict)
    #     error_list.append(r)
    # print(error_list)

    """Stage 5"""
    c = CustomKMeans()
    r = c.find_appropriate_k(X_full, 0.2)
    print(r)  # result = 3

    """Stage 6"""
    c = CustomKMeans()
    c.find_appropriate_k(X_full)
    c.fit(X_full)
    r = c.predict(X_full)
    print(r[:20])
