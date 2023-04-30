from sklearn.multioutput import MultiOutputClassifier
import numpy as np
from numpy import  array_equal
from tqdm.auto import tqdm


class NodeHomerTree(object):
    def __init__(self, value, clusters=None):
        self.key = value
        self.clusters = clusters


class HomerClassifier(object):
    def __init__(self, clusturing_classifier, n_max=10, n_cluster=2,multi_label_classifier=None):
        self.clusturing_classifier = clusturing_classifier
        self.multi_label_classifier = multi_label_classifier
        self.n_max = n_max
        self.n_cluster = n_cluster
        self.fitted_data = None
        self.number_of_nodes = 0
        
    def recursive_kmeans(self,D_train,L_train):
        self.number_of_nodes +=1
        
        # If there is only 1 element in node of there is only one tag combinaison or the node is smaller than n_max
        if len(L_train)<2 or  len(np.unique(L_train,axis=0))<2 or len(L_train)<=self.n_max:
            #We stop the clustering of this node
            return NodeHomerTree({'data':D_train,'labels':L_train})
        # Else we continue the clustering
        else:
            
            classifier = self.clusturing_classifier(self.n_cluster)
            classifier.fit(np.array(D_train,dtype=np.float64),L_train)
            pred = classifier.predict(np.array(D_train,dtype=np.float64),L_train)
            # If the prediction of the clustering algorithm contains only one cluster (so the clustering algorithm failed to create clusters)
            if len(np.unique(pred))==1:
                #We stop the clustering of this node
                return NodeHomerTree({'data':D_train,'labels':L_train})
            else:
                # Else we continue the clustering
                return NodeHomerTree({'data':D_train,'labels':L_train},[self.recursive_kmeans(D_train[pred == cluster_index],L_train[pred == cluster_index]) for cluster_index in np.unique(pred)])
    
    def recursive_train(self,root,bar):
        bar.update(1)
        
        classifier = self.multi_label_classifier(max_depth=10, random_state=42,n_estimators=1000)
        # classifier = self.multi_label_classifier(n_neighbors=3)
        multi_target_forest = MultiOutputClassifier(classifier)
        multi_target_forest.fit(root.key['data'], root.key['labels'])
        root.key['classifier_fitted']=multi_target_forest
            
        # If the root is a leaf
        if root.clusters is not None:
            for cluster in root.clusters:
                self.recursive_train(cluster,bar)
        
        
    def resursive_prediction(self,root,instance):
        prediction = root.key['classifier_fitted'].predict(instance.reshape(1, -1))
        # root.key['prediction']= prediction
        # If the root is not a leaf
        if root.clusters is not None:
            # print("prediction",prediction)
            # print("labels",root.key['labels'])
            # For each child clusters of this node
            res_clusters = []
            for cluster in root.clusters:
                # print("cluster labels",cluster.key['labels'])
                # If the child cluster constrains data with the same combination of labels compare to prediction
                if next((True for elem in cluster.key['labels'] if array_equal(elem.reshape(1, -1), prediction)), False):
                    # We continue the prediction in the child cluster
                    child_prediction = self.resursive_prediction(cluster,instance)
                    # If the prediction of the child is not none (prediction == None is because the prediction in the child doesn't belong to any cluster)
                    if child_prediction is not None:
                        res_clusters.append(child_prediction)
                        
            # print("list of pred fils",res_clusters)
            
            if len(res_clusters)>0:
                # print("concatenated list",np.concatenate(res_clusters))
                return np.concatenate(res_clusters)
            

        
        else:
            return prediction
                    
        
    def fit(self,D_train, L_train):
        print('start recursive balanced K-means')
        clustered_data_tree = self.recursive_kmeans(D_train, L_train)
        print('start recursive train')
        bar = tqdm(total=self.number_of_nodes)
        self.recursive_train(clustered_data_tree,bar)
        self.fitted_data = clustered_data_tree
        return self
    
    def predict(self,D_test):
        print('Start the prediction')
        # For each element to predict
        predictions = []
        
        for instance_index in tqdm(range(len(D_test))):
            # Make the prediction of this element in each concerned nodes of the tree
            predictions_for_instance = self.resursive_prediction(self.fitted_data,D_test[instance_index])
            res = np.zeros(self.fitted_data.key['labels'][0].shape,dtype=int)
            if predictions_for_instance is not None:
                for prediction in predictions_for_instance:
                    res = np.bitwise_or(res,prediction,dtype=int,casting='unsafe')
            predictions.append(res)
        return predictions