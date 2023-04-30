import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = dir_path.split('thesaurusBT/experiments')[0]+'thesaurusBT/'
repo_path = dir_path.split('thesaurusBT/experiments')[0]
sys.path.insert(1, src_path)
from experiments.lib.functions import prepare_labellised_dataset
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from algorithms.catboost_features_selection import get_bests_features_for_each_label
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import time
import pandas as pd
import numpy as np
import argparse



      


def main(preprocessed_data_file_name,preprocessing_technic,caboosted):
      
    print("Execution of : ",os.path.realpath(__file__))


    ####################################################################
    ######################## Data Loading ##############################
    ####################################################################


    # Preprocessed sales data loading (resulting from Occurrence preprocessing script)
    print("*** Start of data loading ***")
    print("Prepocessing technic : ",preprocessing_technic)
    vectors= pd.read_parquet(repo_path+'outputs/preprocessing/'+preprocessed_data_file_name)

    # Loading of labelled stores dataset
    test_dataset = pd.read_json(repo_path+'datasets/validation_dataset.json', lines=True)

    start_general_time_of_execution = time.time()

    # Merge labelled store labels and pre-processed sales data
    test_dataset = test_dataset.merge(vectors, how='inner',on='store_id')
    print("*** End of data loading ***")


    ####################################################################
    ###################### Data Preparation ############################
    ####################################################################


    # Label encoding
    print("*** Start of data preparation ***")
    test_dataset = prepare_labellised_dataset(test_dataset)  

    # Train and test samples preparation
    scaler = MinMaxScaler()
    features = [*test_dataset.columns[3:]]
    # features = [element for element in features if len(np.unique(test_dataset[element].values))>1]
    X =np.concatenate([test_dataset.store_id.values.reshape(-1,1),scaler.fit_transform(np.vstack(test_dataset.vector.values).T).T],axis=1)
    y = test_dataset[features].values  #test_dataset.is_presse.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    print("*** End of data preparation ***")


    ####################################################################
    ########### CatBOOST FEATURE SELECTION (IF ASKED)###################
    ####################################################################

    if caboosted:
        optimized_nb_feature = {'thesaurusBT':9,
                                'pivot':8,
                                'minhash':7}
        # CatBOOST feature selection, select n best features for each label
        print("*** Start of CatBOOST feature selection ***")
        start_time_catboost = time.time()
        df_bests_features_for_each_label = get_bests_features_for_each_label(X_train,y_train,dir_path,optimized_nb_feature[preprocessing_technic])

        # Samples transformations considering CatBOOST feature selection results
        res = np.array([],dtype=int)
        for column in df_bests_features_for_each_label.columns:
                res = np.concatenate([res,df_bests_features_for_each_label[column].values])
        indexes_selected_by_catboost = np.insert((np.unique(res)+1),values=0,obj=0)
        print("*** End of CatBOOST feature selection ***")
        print("Input shape before CatBOOST : ",X_train.shape)
        X_train = X_train[:,indexes_selected_by_catboost]
        X_test = X_test[:,indexes_selected_by_catboost]
        end_time_catboost = time.time()
        print("Input shape after CatBOOST : ",X_train.shape)
        

    ####################################################################
    ######################## Multi-Labelling ###########################
    ####################################################################

    optimized_C = 100

    # Multi-labelling classification with binary relevance model and SVM as base classifier
    print("*** Start of multi-labelling ***")
    start_time_multi_labelling = time.time()
    clf = SVC(C=optimized_C)
    multi_target_SVC = MultiOutputClassifier(clf)
    y_pred = multi_target_SVC.fit(X_train[:,1:], y_train).predict(X_test[:,1:])
    end_time_multi_labelling = time.time()


    ####################################################################
    ##################### Results exportation ##########################
    ####################################################################


    #Export predictions
    if not os.path.exists(repo_path+'outputs'):
            os.makedirs(repo_path+'outputs')
    if not os.path.exists(repo_path+'outputs/processing'):
            os.makedirs(repo_path+'outputs/processing')
    if caboosted:
        output_file_name = preprocessing_technic+'_cb'
    else:
        output_file_name = preprocessing_technic
    np.savetxt(repo_path+"outputs/processing/"+output_file_name+"_svm_pred.csv", y_pred, delimiter=";")
    np.savetxt(repo_path+"outputs/processing/"+output_file_name+"_svm_test.csv", y_test, delimiter=";")
    end_general_time_of_execution = time.time()
    print("*** End of multi-labelling ***")

    # Print times of execution
    print("*** End of processing ***")
    print("Time of execution for processing step : ",end_general_time_of_execution - start_general_time_of_execution)
    if caboosted:
        print("CatBOOST time of execution : ",end_time_catboost - start_time_catboost)
    print("BR SVM multi-labelling time of execution : ",end_time_multi_labelling - start_time_multi_labelling)




if __name__ == '__main__':

    ####################################################################
    ####################### ARGUMENT MANAGER ###########################
    ####################################################################


    argParser = argparse.ArgumentParser()
    argParser.add_argument("-pre", "--preprocessing", help="Preprocessing results to use as input, can be 'thesaurusBT', 'pivot' or 'minhash'.")
    argParser.add_argument("-cb", "--catboost", help="Include feature selection or not, can be 'on', or 'off'.")


    args = argParser.parse_args()
    # print("Preproce=%s" % args.preprocessing)


    preprocessed_data_file_name = None
    
    
    if (args.preprocessing is None or args.preprocessing not in ['thesaurusBT','pivot','minhash']):
        print('Required argument -pre')
        argParser.print_help()
        sys.exit(2)
    else:
        filnames = {'thesaurusBT':'thesaurusBT_data.parquet',
                    'pivot':'clean_pivot_data.parquet',
                    'minhash':'clean_minhash_data.parquet'}
        preprocessed_data_file_name = filnames[args.preprocessing]
        
    caboosted = False


    if (args.catboost is None or args.catboost not in ['on','off']):
        print('Invalid argument for -cb')
        argParser.print_help()
        sys.exit(2)
    elif args.catboost == 'on':
        caboosted = True
        
        
    main(preprocessed_data_file_name,args.preprocessing,caboosted)