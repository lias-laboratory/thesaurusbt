import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = dir_path.split('thesaurusBT/experiments')[0]+'thesaurusBT/'
repo_path = dir_path.split('thesaurusBT/experiments')[0]
sys.path.insert(1, src_path)
import pandas as pd
from experiments.lib.functions import clean_label
import numpy as np
import time

def main():
    print("Execution of : ",os.path.realpath(__file__))

    # Loading sales data from the businesses to be studied
    sales_data = pd.read_parquet(repo_path+'datasets/bimedia_sales_dataset.parquet')

    # Data cleaning (family labels)
    print("*** Start of data cleaning ***")
    start_time_for_cleaning = time.time()
    sales_data['family_label_clean'] = sales_data.family_label.apply(clean_label)
    end_time_for_cleaning = time.time()
    print("*** End of data cleaning ***")
    print("Time for cleaning : ",end_time_for_cleaning - start_time_for_cleaning)

    # Data transformation to vectors : Pivot
    print("*** Start of data pivoting ***")
    start_time_for_pivoting = time.time()
    features = sales_data.family_label_clean.unique()
    df = sales_data.pivot_table(index=['store_id','family_label_clean'], values='quantity',aggfunc='sum')
    vectors = pd.DataFrame()
    vectors['store_id'] = df.reset_index().store_id.unique()
    vectors['vector'] = df.quantity.unstack().fillna(0).to_numpy().tolist()
    vectors['vector']=vectors['vector'].apply(lambda x: np.array(x))
    end_time_for_pivoting = time.time()
    print("*** End of data pivoting ***")
    print("Time for pivoting : ",end_time_for_pivoting - start_time_for_pivoting)

    # Export results
    if not os.path.exists(repo_path+'outputs'):
            os.makedirs(repo_path+'outputs')
    if not os.path.exists(repo_path+'outputs/preprocessing'):
            os.makedirs(repo_path+'outputs/preprocessing')
    vectors.to_parquet(repo_path+'outputs/preprocessing/clean_pivot_data.parquet')
    print("Length of the inputs (number of variables) : ",len(vectors.iloc[0].vector))
    print("*** Results exported to : "+repo_path+"outputs/preprocessing/clean_pivot_data.parquet"+" ***")

    print("*** End of clean + pivot pre-processing ***")


if __name__ == '__main__':
      main()
