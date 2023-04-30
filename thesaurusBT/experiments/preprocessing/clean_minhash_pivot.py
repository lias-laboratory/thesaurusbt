import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = dir_path.split('thesaurusBT/experiments')[0]+'thesaurusBT/'
repo_path = dir_path.split('thesaurusBT/experiments')[0]
sys.path.insert(1, src_path)
import pandas as pd
from experiments.lib.functions import clean_label
from algorithms.minhash import minhash
import numpy as np
import time
import os
import numpy as np

def main():
    print("Execution of : ",os.path.realpath(__file__))

    # Loading sales data from the businesses to be studied
    # sales_data = pd.read_parquet(repo_path+'datasets/tmp_jeu_de_donnee_ventes_2021.parquet')
    sales_data = pd.read_parquet(repo_path+'datasets/bimedia_sales_dataset.parquet')

    # Data cleaning (family labels)
    print("*** Start of data cleaning ***")
    start_time_for_cleaning = time.time()
    sales_data['family_label_clean'] = sales_data.family_label.apply(clean_label)
    end_time_for_cleaning = time.time()
    print("*** End of data cleaning ***")
    print("Time for cleaning : ",end_time_for_cleaning - start_time_for_cleaning)

    # Data transformation to vectors : Minhash
    print("*** Start of MinHash data transformation ***")
    start_time_for_minhash = time.time()


    sales_data['min_hash_family_label'] = sales_data.apply(lambda row: minhash(set(row.family_label_clean.split(' '))),axis=1)
    sales_data['min_hash_str'] = sales_data.min_hash_family_label.apply(str)
    print("Number of variables in original the dataframe (after cleaning) : ",len(np.unique(sales_data.family_label_clean)))
    print("Number of variables after minhash : ",len(np.unique(sales_data.min_hash_str)))

    # Vector building from minhash results
    features = sales_data.min_hash_str.unique()
    df = sales_data.pivot_table(index=['store_id','min_hash_str'], values='quantity',aggfunc='sum')
    vectors = pd.DataFrame()
    vectors['store_id'] = df.reset_index().store_id.unique()
    vectors['vector'] = df.quantity.unstack().fillna(0).to_numpy().tolist()
    vectors['vector']=vectors['vector'].apply(lambda x: np.array(x))
    end_time_for_minhash = time.time()
    print("*** End of MinHash data transformation ***")
    print("Time for MinHash : ",end_time_for_minhash - start_time_for_minhash)



    # Export results
    if not os.path.exists(repo_path+'outputs'):
            os.makedirs(repo_path+'outputs')
    if not os.path.exists(repo_path+'outputs/preprocessing'):
            os.makedirs(repo_path+'outputs/preprocessing')
    vectors.to_parquet(repo_path+'outputs/preprocessing/clean_minhash_pivot_data.parquet')
    print("Length of the inputs (number of variables) : ",len(vectors.iloc[0].vector))
    print("*** Results exported to : "+repo_path+"outputs/preprocessing/clean_minhash_pivot_data.parquet"+" ***")

    print("*** End of clean + minhash + pivot pre-processing ***")

if __name__ == '__main__':
    main()
