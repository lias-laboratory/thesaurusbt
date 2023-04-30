import numpy as np
from numpy.linalg import norm
import math

class Balanced_Kmeans:
    '''Implementing balanced Kmeans algorithm.'''

    def __init__(self, n_clusters, max_iter=100, random_state=123,k_means_type='balanced'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.k_means_type = k_means_type #can be balanced of classical

    def initializ_centroids(self, X):
        np.random.RandomState(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.n_clusters]]
        return centroids

    def compute_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            centroids[k, :] = np.mean(X[labels == k, :], axis=0)
        return centroids

    def compute_distance(self, X, centroids):
        distance = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            row_norm = norm(X - centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)
        return distance

    def find_closest_cluster(self, distance,y):
        
        if self.k_means_type == 'balanced':
        
            ## Initialization of constants
            number_of_element_in_sample = distance.shape[0]
            size_of_cluster = math.ceil(number_of_element_in_sample/self.n_clusters)
            
            ## Initialization of results
            res = np.full(distance.shape, -1.0)

            ## Looping across elements of the dataset 
            for i in range(len(y)):
                finished = False
                row_index = i
                while finished == False:
                    # Get the closest cluster of this element
                    j= np.argmin(distance, axis=1)[row_index]
                    # Assign this element to the cluster with the closest centroid
                    res[row_index,j]=distance[row_index,j]
                    # If because of this new element assigned to the cluster
                    if len(res[res[:,j]!=-1]) > size_of_cluster:
                        # Get the index of the element of the cluster with de biggest distance to the centroid
                        index_of_element_of_the_cluster_with_max_distance = np.argmax(res[:,j])
                        # Unassign this element to the cluster
                        res[index_of_element_of_the_cluster_with_max_distance,j]=-1
                        # Set the distance of this element to the centroid of this cluster to infinity
                        distance[index_of_element_of_the_cluster_with_max_distance,j]= float('inf')
                        # Set the row_index with the index of the element just deleted in order to assign it to another cluster
                        row_index = index_of_element_of_the_cluster_with_max_distance
                    # If the element was assigned to this cluster and the cluster is not overfull
                    else:
                        # Going forward with de next element
                        finished=True
            res[res!=-1] = 1
            res[res == -1] = 0
            res = np.array([np.argmax(element) for element in res],dtype=np.uint8)
            return res
        else:
            return np.argmin(distance, axis=1)

    def compute_sse(self, X, labels, centroids):
        distance = np.zeros(X.shape[0])
        for k in range(self.n_clusters):
            distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1)
        return np.sum(np.square(distance))
    
    def fit(self, X, y):
        self.centroids = self.initializ_centroids(X)
        for i in range(self.max_iter):
            old_centroids = self.centroids
            distance = self.compute_distance(X, old_centroids)
            # print(distance)
            self.labels = self.find_closest_cluster(distance,y)
            self.centroids = self.compute_centroids(X, self.labels)
            if np.all(old_centroids == self.centroids):
                break
        self.error = self.compute_sse(X, self.labels, self.centroids)
    
    def predict(self, X, y):
        distance = self.compute_distance(X, self.centroids)
        return self.find_closest_cluster(distance,y)
