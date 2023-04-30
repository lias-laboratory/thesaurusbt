import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = dir_path.split('thesaurusBT')[0]+'thesaurusBT/'
sys.path.insert(1, src_path)
from experiments.preprocessing.clean_pivot import main as clean_pivot
from experiments.preprocessing.clean_minhash_pivot import main as clean_minhash_pivot
from experiments.preprocessing.clean_thesaurusbt_pivot import main as clean_thesaurusbt_pivot
from experiments.processing.labelling_br_rf import main as br_rf
from experiments.processing.labelling_br_svm import main as br_svm
from experiments.processing.labelling_homer import main as homer
from experiments.results.results import main as results_calculation
import argparse
import time


def main():
    ####################################################################
    ####################### ARGUMENT MANAGER ###########################
    ####################################################################


    argParser = argparse.ArgumentParser()
    argParser.add_argument("-all", "--allcombinations", action='store_true', help="Execute all the combinations of workflow steps.")
    argParser.add_argument("-pre", "--preprocessing", help="Preprocessing method to use for data transformation, can be 'thesaurusBT', 'pivot' or 'minhash'.")
    argParser.add_argument("-pro", "--processing", help="Processing method to use for multi-labelling, can be 'br_rf', 'br_svm' or 'homer'.")
    argParser.add_argument("-cb", "--catboost", help="Include feature selection or not, can be 'on', or 'off'.")
    argParser.add_argument("-res", "--results",action='store_true', help="Enable score reporting.")



    args = argParser.parse_args()
    # print("Preproce=%s" % args.preprocessing)


    preprocessed_data_file_name = None

    if args.allcombinations==True:
        # Execute all the preprocessing technics:
        clean_thesaurusbt_pivot()
        clean_pivot()
        clean_minhash_pivot()
        
        #Execute all the processing methods with all the preprocessing results and with or whithout catboost
        br_svm('clean_thesaurusbt_pivot_data.parquet','thesaurusBT',caboosted=False)
        br_svm('clean_pivot_data.parquet','pivot',caboosted=False)
        br_svm('clean_minhash_pivot_data.parquet','minhash',caboosted=False)
        br_svm('clean_thesaurusbt_pivot_data.parquet','thesaurusBT',caboosted=True)
        br_svm('clean_pivot_data.parquet','pivot',caboosted=True)
        br_svm('clean_minhash_pivot_data.parquet','minhash',caboosted=True)
        br_rf('clean_thesaurusbt_pivot_data.parquet','thesaurusBT',caboosted=False)
        br_rf('clean_pivot_data.parquet','pivot',caboosted=False)
        br_rf('clean_minhash_pivot_data.parquet','minhash',caboosted=False)
        br_rf('clean_thesaurusbt_pivot_data.parquet','thesaurusBT',caboosted=True)
        br_rf('clean_pivot_data.parquet','pivot',caboosted=True)
        br_rf('clean_minhash_pivot_data.parquet','minhash',caboosted=True)
    
    else:
        datatransformation_technic = None
    
        if args.results==False and args.allcombinations==False and (args.preprocessing is None or args.preprocessing not in ['thesaurusBT','pivot','minhash']):
            print('Invalid argument for -pre, -pre argument must be specified if -all flag not included')
            argParser.print_help()
            sys.exit(2)
        elif args.preprocessing is not None and args.preprocessing in ['thesaurusBT','pivot','minhash']:
            datatransformation_technics = {'thesaurusBT':clean_thesaurusbt_pivot,
                        'pivot':clean_pivot,
                        'minhash':clean_minhash_pivot}
            datatransformation_technic = datatransformation_technics[args.preprocessing]
            filnames = {'thesaurusBT':'clean_thesaurusbt_pivot_data.parquet',
                        'pivot':'clean_pivot_data.parquet',
                        'minhash':'clean_minhash_pivot_data.parquet'}
            preprocessed_data_file_name = filnames[args.preprocessing]


        caboosted = False


        if args.results is None and args.allcombinations is None and (args.catboost is None or args.catboost not in ['on','off']):
            print('Invalid argument for -cb, -cb argument must be specified if -all flag not included')
            argParser.print_help()
            sys.exit(2)
        elif args.catboost == 'on':
            caboosted = True
            
            
        multi_labelling_technic = None  
            
                    
        if args.results==False and args.allcombinations==False and (args.processing is None or args.processing not in ['br_rf','br_svm','homer']):
            print('Invalid argument for -pro, -pro argument must be specified if -all flag not included')
            argParser.print_help()
            sys.exit(2)
        elif args.processing is not None and args.processing in ['br_rf','br_svm','homer']:
            multi_labelling_technics = {'br_rf':br_rf,
                        'br_svm':br_svm,
                        'homer':homer}
            multi_labelling_technic = multi_labelling_technics[args.processing]
            

        if datatransformation_technic is not None and multi_labelling_technic is not None:
            start_general_time_of_execution = time.time()
            datatransformation_technic()
            multi_labelling_technic(preprocessed_data_file_name,args.preprocessing,caboosted)
            end_general_time_of_execution = time.time()
            print("Time of execution for complete workflow : ",end_general_time_of_execution - start_general_time_of_execution)
        
    if args.results==True:
        results_calculation()
        
    
if __name__ == '__main__':
    main()