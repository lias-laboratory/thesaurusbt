import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = dir_path.split('thesaurusBT/experiments')[0]+'thesaurusBT/'
repo_path = dir_path.split('thesaurusBT/experiments')[0]
sys.path.insert(1, src_path)
import pandas as pd
from experiments.lib.functions import get_type_of_family,extract_generated_barcode,generate_new_unique_product_identifier,clean_label,clean_stop_words,max_occurence_vat,search_in_df,deep_search_in_df,ambigious_processing,final_predicted_type,predicted_type_level_1,prepare_stop_words_list
import numpy as np
from collections import Counter
from statistics import mode
import logging
import time
import json


def main():
      dir_path = os.path.dirname(os.path.realpath(__file__))

      print("Execution of : ",os.path.realpath(__file__))

      # Loading sales data from the businesses to be studied
      print("*** Start of data loading ***")
      products_sold = pd.read_parquet(repo_path+'datasets/bimedia_sales_dataset.parquet')

      print("*** Start of thesaurusBT script ***")

      start_time = time.time()

      base_dir= dir_path

      #########################################################
      ########## PART 1 : Data preparation ####################
      #########################################################

      print("Data preparation step 1")

      # Product family type computation
      print('Product family type computation')
      products_sold['family_type'] = get_type_of_family(products_sold,family_dictionnary_path=base_dir+'/input/thesaurus_ressources/dictionnaire_familles.json')

      # Export pre-processing step to save ram memory
      if not os.path.exists(base_dir+'/tmp'):
            os.makedirs(base_dir+'/tmp')
      products_sold.to_pickle(base_dir+'/tmp/tmp_pre_processed_data.pkl')

      # Separate data, keeping only non fixed family products for data processing (the other are defined by bimedia and are not to be identified)
      products_sold_to_identify = products_sold[products_sold.family_type == 'unfixed_category'].copy()
      del products_sold

      # Identification of generated barcodes (not scanned on products but generated by the system or invented by the store owner)
      products_sold_to_identify = products_sold_to_identify[['barcode','store_id','family_id','product_label','family_label', 'vat','family_type']].copy()
      products_sold_to_identify['generated'] = products_sold_to_identify.apply(lambda row : extract_generated_barcode(row.barcode,row.family_id), axis=1)
      # print("Number of unique barcode before generated barcode identification : "+ str(len(np.unique(products_sold_to_identify.barcode))))

      # Regenerate barcodes if necessary
      fr_stop =  prepare_stop_words_list()
      products_sold_to_identify['unique_product_identifier'] = products_sold_to_identify.apply(lambda row: generate_new_unique_product_identifier(row.product_label,row.generated,row.barcode,fr_stop),axis=1)
      # print("Number of unique barcode after generated barcode identification : "+ str(len(np.unique(products_sold_to_identify.unique_product_identifier))))
      del fr_stop

      # Export of the dictionary of unique identifiers for the products to be studied with the thesaurusBT method (to save memory)
      if not os.path.exists(base_dir+'/tmp'):
            os.makedirs(base_dir+'/tmp')
      products_sold_to_identify[['store_id','family_id','barcode','unique_product_identifier']].to_pickle(base_dir+'/tmp/dictionnary_product_unique_identifer.pkl')


      print("Data preparation step 2")

      # Textual data cleaning
      dataset_sample_to_identify = products_sold_to_identify.pivot_table(index=['unique_product_identifier','store_id','family_id'], values=['product_label','family_label','vat'] ,aggfunc={"product_label":mode,'family_label':mode,'vat':mode}).reset_index()
      del products_sold_to_identify
      dataset_sample_to_identify['family_label'] = dataset_sample_to_identify.apply(lambda row: clean_label(row.family_label,True),axis=1)
      dataset_sample_to_identify['product_label'] = dataset_sample_to_identify.apply(lambda row: clean_label(row.product_label,True),axis=1)
      fr_stop =  prepare_stop_words_list()
      dataset_sample_to_identify['family_label_sans_stop'] = dataset_sample_to_identify.apply(lambda row: clean_stop_words(row.family_label,fr_stop),axis=1)
      dataset_sample_to_identify['family_label_split'] = dataset_sample_to_identify.apply(lambda row: row.family_label_sans_stop.split(' '),axis=1)
      del fr_stop
      len_before_supp = len(dataset_sample_to_identify)
      dataset_sample_to_identify = dataset_sample_to_identify[dataset_sample_to_identify.family_label != '']
      print("Number of product deleted from the dataframe because empty name after cleaning : "+str(len_before_supp - len(dataset_sample_to_identify))+" soit : "+str((len_before_supp - len(dataset_sample_to_identify))/len(dataset_sample_to_identify)*100)+"%")


      #########################################################
      ########## PART 2 : Data processing  ####################
      #########################################################

      print("Data processing step 1")

      # Pivoting data by product unique identifier generated
      dataset_sample_to_identify = dataset_sample_to_identify.pivot_table(index=['unique_product_identifier'], values=['family_label','family_label_split','product_label','vat'], aggfunc={'family_label':list, 'family_label_split':'sum', 'product_label':list, 'vat':list})

      # Words occurrence calculation
      dataset_sample_to_identify['len_familly_list'] = dataset_sample_to_identify.apply(lambda row: len(row.family_label),axis=1)
      dataset_sample_to_identify['occurence'] = dataset_sample_to_identify.apply(lambda row: Counter(row.family_label_split),axis=1)
      dataset_sample_to_identify['max_occurence'] = dataset_sample_to_identify.apply(lambda row: row.occurence.most_common()[:3],axis=1)
      dataset_sample_to_identify['max_occurence_count_ratio'] = dataset_sample_to_identify.apply(lambda row: row.max_occurence[0][1]/len(row.family_label_split),axis=1)
      dataset_sample_to_identify['occurence_vat'] = dataset_sample_to_identify.apply(lambda row: Counter(row.vat),axis=1)
      dataset_sample_to_identify['max_occurence_vat']= dataset_sample_to_identify.apply(lambda row: max_occurence_vat(row.occurence_vat), axis=1)


      # Loading of the thesaurusBT words dictionary necessary for its execution
      with open(base_dir+'/input/thesaurus_ressources/identify_type.json') as f:
            data = f.read()
      dico_families = json.loads(data)

      print("Data processing step 2")

      # Product type prediction by word occurrence processing
      dataset_sample_to_identify['predicted_type_dico1']= dataset_sample_to_identify.apply(lambda row: search_in_df(row.max_occurence,dico_families), axis=1)
      dataset_sample_to_identify['predicted_type_dico2']= dataset_sample_to_identify.apply(lambda row: deep_search_in_df(row.predicted_type_dico1,row.occurence, dico_families), axis=1)
      dataset_sample_to_identify['predicted_type_dico3']= dataset_sample_to_identify.apply(lambda row: ambigious_processing(row.predicted_type_dico2,row.max_occurence_vat,row.occurence, dico_families), axis=1)
      dataset_sample_to_identify['predicted_type_level_2']= dataset_sample_to_identify.apply(lambda row: final_predicted_type(row.predicted_type_dico3), axis=1)
      del dico_families

      # Identification results (coverage)
      print("Number of identified products : ",len(dataset_sample_to_identify[dataset_sample_to_identify.predicted_type_level_2 != 'inconnu']))
      print("Percentage of identified products : ",len(dataset_sample_to_identify[dataset_sample_to_identify.predicted_type_level_2 != 'inconnu'])/len(dataset_sample_to_identify)*100)
      print("Percentage of non ambiguous identified products : ",len(dataset_sample_to_identify[(dataset_sample_to_identify.predicted_type_level_2 != 'inconnu') & (dataset_sample_to_identify.predicted_type_level_2.str.contains("ambigious")==False)])/len(dataset_sample_to_identify)*100)
      print("Number of non identified products : ",len(dataset_sample_to_identify[dataset_sample_to_identify.predicted_type_level_2 == 'inconnu'])+len(dataset_sample_to_identify[dataset_sample_to_identify.predicted_type_level_2.isnull()]))
      print("Percentage of non identified products : ",len(dataset_sample_to_identify[dataset_sample_to_identify.predicted_type_level_2 == 'inconnu'])/len(dataset_sample_to_identify)*100)

      print("Data processing step 3")

      # Product level 1 identification based on level 2 identification
      with open(base_dir+'/input/thesaurus_ressources/identify_type_level.json') as f:
            data = f.read()
      dico_level = json.loads(data)
      dataset_sample_to_identify['predicted_type_level_1'] = dataset_sample_to_identify.apply(lambda row: predicted_type_level_1(row.predicted_type_level_2,dico_level,row.predicted_type_dico3),axis=1)
      print("Percentage of non ambiguous identified products on level 1 : ",len(dataset_sample_to_identify[dataset_sample_to_identify.predicted_type_level_1 == 'Ambigious'])/len(dataset_sample_to_identify)*100)

      # Reloading unique identifier dictionnary to mappe predictions to the products
      dictionnary_product_unique_identifier = pd.read_pickle(base_dir+'/tmp/dictionnary_product_unique_identifer.pkl')
      os.remove(base_dir+'/tmp/dictionnary_product_unique_identifer.pkl')
      dataset_sample_to_identify_processed = dictionnary_product_unique_identifier.merge(dataset_sample_to_identify.reset_index()[['unique_product_identifier','predicted_type_level_1','predicted_type_level_2']],how='inner',on='unique_product_identifier')[['store_id','family_id','barcode','predicted_type_level_1','predicted_type_level_2']]
      del dictionnary_product_unique_identifier
      del dataset_sample_to_identify

      # Ram reload preprocessed data export earlier in tmp folder
      dataset_sample = pd.read_pickle(base_dir+'/tmp/tmp_pre_processed_data.pkl')
      os.remove(base_dir+'/tmp/tmp_pre_processed_data.pkl')

      # Merging fixed and non-fixed family data
      logging.info("Merging data of fixed and unfixed categories") 
      export = dataset_sample.merge(dataset_sample_to_identify_processed,how='left',on=['store_id','family_id','barcode'])

      # Construction of level 2 for fixed families thanks to the dictionary of fixed families
      export['type_level_2']= export.apply(lambda row: row.predicted_type_level_2 if row.family_type == 'unfixed_category' else row.family_type,axis=1)

      # Construction of level 1 on fixed families
      export['type_level_1'] = export.apply(lambda row: predicted_type_level_1(row.type_level_2,dico_level) if row.family_type != 'unfixed_category' else row.predicted_type_level_1 ,axis=1)
      del dico_level

      #########################################################
      ############ PART 3 : Results preparation ###############
      #########################################################
      print("Results preparation step 1")


      # Data preparation for exportation
      print("Data preparation for exportation")
      export.rename(columns={'type_level_1':'type_product_predicted_1','type_level_2':'type_product_predicted_2'},inplace=True)
      export.set_index(['store_id','barcode'],drop=True,inplace=True)
      export['type_product_predicted_1'] = export['type_product_predicted_1'].str.replace("\'", "\''")
      export['type_product_predicted_2'] = export['type_product_predicted_2'].str.replace("\'", "\''")
      export = export[['type_product_predicted_1','type_product_predicted_2']]

      print("Results preparation step 2")

      # Merge between raw data and thesaurusBT results
      products_sold = pd.read_parquet(repo_path+'datasets/bimedia_sales_dataset.parquet')
      products_sold = products_sold.merge(export.reset_index(),how='left',on=['store_id','barcode'])
      products_sold = products_sold[~products_sold.type_product_predicted_2.isnull()]
      print("End of occurence data processing, time of execution : "+str(time.time() - start_time)) 

      # Delete tmp directory
      os.rmdir(base_dir+'/tmp')

      # Vector building
      features =  products_sold.type_product_predicted_2.unique()
      df = products_sold.pivot_table(index=['store_id','type_product_predicted_2'],values='quantity',aggfunc='sum')
      vectors = pd.DataFrame()
      vectors['store_id'] = df.reset_index().store_id.unique()
      vectors['vector'] = df.quantity.unstack().fillna(0).to_numpy().tolist()
      vectors['vector']=vectors['vector'].apply(lambda x: np.array(x))

      # Result exportation
      if not os.path.exists(repo_path+'outputs'):
            os.makedirs(repo_path+'outputs')
      if not os.path.exists(repo_path+'outputs/preprocessing'):
            os.makedirs(repo_path+'outputs/preprocessing')
      vectors.to_parquet(repo_path+'outputs/preprocessing/clean_thesaurusbt_pivot_data.parquet')

      print("Length of the inputs (number of variables) : ",len(vectors.iloc[0].vector))
      print("*** Results exported to : "+repo_path+"outputs/preprocessing/clean_thesaurusbt_pivot_data.parquet"+" ***")

      print("*** End of clean + thesaurusBT + pivot pre-processing ***")

if __name__ == '__main__':
      main()


