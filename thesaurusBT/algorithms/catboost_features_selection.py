import catboost as cb
import pandas as pd
import numpy as np
import shutil


def get_bests_features_for_each_label(X_train,y_train,base_dir,nb_features=10):
    res_df = pd.DataFrame()
    for label_index in range(y_train.shape[1]):
        if len(np.unique(y_train[:,label_index]))>1:
            model = cb.CatBoostClassifier(loss_function='Logloss', eval_metric='Accuracy',train_dir=base_dir+'/catboost_info')
            model.fit(X_train[:,1:],y_train[:,label_index], verbose_eval=False)
            res_df['feature_importance_label'+str(label_index)] = model.feature_importances_
        else:
            res_df['feature_importance_label'+str(label_index)] = [0]*X_train[:,1:].shape[1]
    df2=pd.DataFrame()
    for columns in res_df.columns:
        df2[str(nb_features)+'_bests_'+columns] = res_df[columns].sort_values(ascending=False)[:nb_features].index.values
    shutil.rmtree(base_dir+'/catboost_info')
    
    return df2
