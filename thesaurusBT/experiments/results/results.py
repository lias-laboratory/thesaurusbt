# ## Loading and score calculation of processing outputs
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = dir_path.split('thesaurusBT/experiments')[0]+'thesaurusBT/'
repo_path = dir_path.split('thesaurusBT/experiments')[0]
sys.path.insert(1, src_path)
import numpy as np
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score,confusion_matrix,ConfusionMatrixDisplay
import pandas as pd


def main():
    entries = os.listdir(repo_path+'outputs/processing')

    models = []
    accuracy = []
    macro_pres= []
    micro_pres = []
    example_pres = []
    macro_rec = []
    micro_rec = []
    example_rec = []
    macro_f1 = []
    micro_f1 = []
    example_f1 = []

    def rename_entry(entry):
        res=[]
        entry = entry.replace("_test","").replace(".csv","").lower()
        entry = entry.split("_")
        if "thesaurusbt" in entry:
            res.append("th")
        if "minhash" in entry:
            res.append("mh-pv")
        if "pivot" in entry:
            res.append("pv")
        if "cb" in entry:
            res.append("cb")
        if "rf" in entry:
            res.append("rf")
        if "svm" in entry:
            res.append("svm")
        return "-".join(res)

    for element in entries:
        if element[-8:]=='test.csv':
            y_test = np.genfromtxt(repo_path+'outputs/processing/'+element, delimiter=";")
            y_pred = np.genfromtxt(repo_path+'outputs/processing/'+element.replace('test','pred'), delimiter=";")
            print("========================================================================")
            print(rename_entry(element))
            models.append(rename_entry(element))
            print("accuracy : ", accuracy_score(y_test, y_pred))
            accuracy.append(accuracy_score(y_test, y_pred))
            print("macro-precision : ", precision_score(y_test, y_pred,average='macro',zero_division=0))
            macro_pres.append(precision_score(y_test, y_pred,average='macro',zero_division=0))
            print("micro-precision : ", precision_score(y_test, y_pred,average='micro',zero_division=0))
            micro_pres.append(precision_score(y_test, y_pred,average='micro',zero_division=0))
            print("precision examples : ", precision_score(y_test, y_pred,average='samples',zero_division=0))
            example_pres.append(precision_score(y_test, y_pred,average='samples',zero_division=0))
            print("macro-recall : ", recall_score(y_test, y_pred,average='macro',zero_division=0))
            macro_rec.append(recall_score(y_test, y_pred,average='macro',zero_division=0))
            print("micro-recall : ", recall_score(y_test, y_pred,average='micro',zero_division=0))
            micro_rec.append(recall_score(y_test, y_pred,average='micro',zero_division=0))
            print("recall examples : ", recall_score(y_test, y_pred,average='samples',zero_division=0))
            example_rec.append(recall_score(y_test, y_pred,average='samples',zero_division=0))
            print("macro-f1 : ", f1_score(y_test, y_pred,average='macro',zero_division=0))
            macro_f1.append(f1_score(y_test, y_pred,average='macro',zero_division=0))
            print("micro-f1 : ", f1_score(y_test, y_pred,average='micro',zero_division=0))
            micro_f1.append(f1_score(y_test, y_pred,average='micro',zero_division=0))
            print("f1 examples : ", f1_score(y_test, y_pred,average='samples',zero_division=0))
            example_f1.append(f1_score(y_test, y_pred,average='samples',zero_division=0))
    
            



    # ## Dataframe building from outputs
    results = pd.DataFrame(zip(models,accuracy,macro_pres,micro_pres,example_pres,macro_rec,micro_rec,example_rec,macro_f1,micro_f1,example_f1),columns=['model','accuracy','macro_pres','micro_pres','example_pres','macro_rec','micro_rec','example_rec','macro_f1','micro_f1','example_f1'])

    def find_preprocessing_methods(model):
        if 'th' in model:
            return 'th'
        elif 'mh-pv' in model:
            return 'mh-pv'
        elif 'pv' in model:
            return 'pv'
        
    results['pre-processing method'] = results.model.apply(find_preprocessing_methods)
    results['catboost method'] = results.model.apply(lambda val: 'used' if 'cb' in val else 'unused')
    results['multi-labelling method'] = results.model.apply(lambda val:  val.split('-')[-1])
    results[['model','accuracy','macro_f1']]

    if not os.path.exists(repo_path+'outputs/results'):
        os.makedirs(repo_path+'outputs/results')

    results.to_csv(repo_path+'outputs/results/results.csv')

if __name__ == '__main__':
    main()
