import os
import pandas as pd
import re
import hashlib
import unicodedata
from w3lib.html import replace_entities
import string
import json
import os
import numpy as np
from functools import reduce


def extract_barcode_with_numbers_only(inputString):
    return bool(re.search(r'^[0-9]+$', inputString))            
            

def extract_barcode_generated_310000_en13(inputString):
    if bool(re.search(r'^[0-9]+$', inputString)) and len(inputString)==13:
        if inputString[:8]== '31000000':
            return True
    return False

def extract_generated_barcode(inputString,family_id):
    if len (inputString) < 6:
        return True
    else : 
        is_31XX_generated = extract_barcode_generated_310000_en13(inputString)
        if is_31XX_generated:
            return True
        elif str(inputString) == str(family_id):
            return True
        else:
            if extract_barcode_with_numbers_only(str(inputString).strip()):
                if len(inputString)<8:
                    return True
                else:
                    return False
            else: 
                return False
        

def strip_accents(text):
    try:
        text = unicode(text, 'utf-8')
    except (TypeError, NameError): # unicode is a default on python 3 
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)

def prepare_stop_words_list():
    
    fr_stop = ['ml','mg','g','l','cl','litres','litre',
               'unite','unites','gr','en','le','la','les',
               'mon','ma','mes','ton','ta','tes','a','de','des','et','avec','sans','autre','article','autres','articles','du','au']
    return fr_stop
    
    
def clean_stop_words(label,fr_stop):
    
    tokens = label.split(" ")
    tokens_filtered= [word for word in tokens if not word in fr_stop]
    return (" ").join(tokens_filtered)
    

def prepare_labellised_dataset(df):
    types = list(reduce(lambda i, j: set(i) | set(j), df.type))
    print(types)
    for unique_type in types:
        df[unique_type] = df.type.apply(lambda x: 1 if unique_type in x else 0)
    return df

def clean_label(label,remove_digits=False):
    try:
        label = bytearray(
            label, 'Windows-1252').decode('unicode_escape', "strict")
        label =  replace_entities(label)

        label = strip_accents(label)
        #delete dot
        label = label.replace('.', '')
        #delete dot
        label = label.replace('.', '')
        #delete coma
        label = label.replace(',', '')
        # Remove punctuations
        label = re.sub('[%s]' % re.escape(string.punctuation), ' ', label)
        # # Remove numbers
        if remove_digits:
            label = re.sub(r'\d', '', label)
        #lower case
        label = label.lower()
        # Remove multiple spaces
        label = re.sub('\s{2,}', " ", label)

        if label.strip() == "c b d" :
            label = "cbd"
        if label.strip() == "p q r":
            label = "pqr"
        if label.strip() == "p q n":
            label = "pqn"
    
        # label = re.sub('(^|\s)(sans|non|e|ss)\s',r'\1\2-' ,label)
        return label.strip()
    except:
        return ""
    



def generate_new_unique_product_identifier(inputString, isGenerated, orginalBarcode,fr_stop=None):
    
    if isGenerated:
        if fr_stop is None:
            inputString = clean_label(inputString)
        else:
            inputString = clean_stop_words(clean_label(inputString),fr_stop)
        return hashlib.sha224(inputString.encode('utf-8')).hexdigest()
    return orginalBarcode




def get_type_of_family_func(row,fixed_category_dict):
    telephonie = ['orange rd','orange tva', 'orange','sfr','simyo','ortel','symacom','telephonie','ticket telephone rechargeable','vectone','adn','pack telephone','virgin','universal mobile','libon','kertel','gt mobile','impact sfr','intercall''laposte mobile','lebara','bandyou','bouygues','bouygues t l com','bouygues tel com','bouygues tel m','bouygues telacom','bouygues telecom','bouygues tilicom','corpedia','lyca mobile','lycamobile','mobiho','mobisud','delta m','delta multimedia','france t l com','france tel','france tel m','france telacom','france telecom','france tilicom','ft cabine','sfr sl fdj']
    autre = ['jouet','la poste','ouibus','papeterie','presse','test','ticket premium','um']
    moyen_de_paiement = ['spotify','steam','keplerk by moneyclic','kobo','sfr paycard','moneyclic','my pocket','neosurf','netflix','nintendo','adduniti','itunes','google','fnac darty','amazon','auchan','carte cadeau mode','cashlib','vp','winamax','wonderbox carte cadeau','xbox','zalando''ticketcash','ticketsurf''toneo','toneo first','transcash','transcashrex','twitch','ukash','paysafecard','pcs','pcs direct','pcs phy','playstation']

    type_of_family = fixed_category_dict.get(str(row['family_id']), {'type_famille' : 'unfixed_category'})['type_famille']
    if type_of_family == 'Téléphonie / Moyens de paiement':
        clean_family_label = clean_label(row.family_label)
        if clean_family_label in telephonie:
            return 'Telephonie'
        elif clean_family_label in moyen_de_paiement:
            return 'Moyen de paiement'
        else:
            return 'unfixed_category'
    return type_of_family

def get_type_of_family(df,family_dictionnary_path):
    with open(family_dictionnary_path) as f:
        data = f.read()
    fixed_category_dict = json.loads(data)
    family_type_serie = df.apply(lambda row : get_type_of_family_func(row,fixed_category_dict), axis=1)
    return family_type_serie


############################## FONCTIONS DE LA METHODE OCCURENCE ###################################

def particular_prediction_processing(result_prediction,max_occurence,dico_families):
    if 'Fumeur' in result_prediction['prediction'] or 'Cigarette' in result_prediction['prediction'] or 'Cigare' in result_prediction['prediction'] or 'Chicha' in result_prediction['prediction'] or 'Article fumeur' in result_prediction['prediction']:
        if len(max_occurence)>1:
            second_max_occurence_word = max_occurence[1][0]
            res2 = search_word_in_dict_func(second_max_occurence_word,dico_families)
            if len(res2['prediction'])>0:
                if 'Vape' in res2['prediction'] or 'E cigarette' in res2['prediction'] or 'E liquide' in res2['prediction'] or 'CBD' in res2['prediction'] or 'CBD a fumer' in res2['prediction']:
                    return res2
        if len(max_occurence)>2:
            third_max_occurence_word = max_occurence[2][0]
            res3 = search_word_in_dict_func(third_max_occurence_word,dico_families)
            if len(res3['prediction'])>0:
                if 'Vape' in res3['prediction'] or 'E cigarette' in res3['prediction'] or 'E liquide' in res3['prediction'] or 'CBD' in res3['prediction'] or 'CBD a fumer' in res3['prediction']:
                    return res3
    return result_prediction

def search_word_in_dict_func(word,dico_families):
    res= {"prediction": [], "ambigious": []}
    for key in dico_families.keys():
        if word in dico_families[key]['ambigious_words']:
            res["ambigious"].append(key)
    for key in dico_families.keys():
        if word in dico_families[key]['ambigious_words_plural']:
            res["ambigious"].append(key)
    for key in dico_families.keys():
        if word in dico_families[key]['keywords']:
            res["prediction"].append(key)
    for key in dico_families.keys():
        if word in dico_families[key]['keywords_plural']:
            res["prediction"].append(key)
    return res

def prediction_word(max_occurence,dico_families):
    max_occurence_word = max_occurence[0][0]
    res = search_word_in_dict_func(max_occurence_word,dico_families)
    if len(res['prediction']) == 0 and len(max_occurence) >1: #Si aucune prédiction n'a été trouvée avec le premier mot (donc il peut y avoir des ambigues ou pas)
            second_max_occurence_word = max_occurence[1][0]
            res2 = search_word_in_dict_func(second_max_occurence_word,dico_families)
            if len(res2['prediction'])>0 and len(res['ambigious'])>0: #Si une prédiction a été trouvé alors qu'un ambigue avait été trouvé, on regarde si la prédiction se trouve dans l'ambigue de l'origine
                if res2['prediction'] in res['ambigious']:
                    return particular_prediction_processing(res2,max_occurence,dico_families)
                else:
                    return particular_prediction_processing(res,max_occurence,dico_families)
            elif len(res2['prediction'])>0 and len(res['ambigious'])==0:
                return particular_prediction_processing(res2,max_occurence,dico_families)
            elif len(res2['ambigious'])>0 and len(res['ambigious'])==0:
                return particular_prediction_processing(res2,max_occurence,dico_families)
    return particular_prediction_processing(res,max_occurence,dico_families)

def search_duo_in_dict_func(duo,dico_families):
    res= {"prediction": [], "ambigious": []}
    for key in dico_families.keys():
        if duo in dico_families[key]['ambigious_tuples']:
            res["ambigious"].append(key)
    for key in dico_families.keys():
        if duo in dico_families[key]['ambigious_tuples_plural']:
            res["ambigious"].append(key)
    for key in dico_families.keys():
        if duo in dico_families[key]['keyword_tuples']:
            res["prediction"].append(key)
    for key in dico_families.keys():
        if duo in dico_families[key]['keyword_tuples_plural']:
            res["prediction"].append(key)
    return res


def prediction_duo(duo,dico_families):
        res = search_duo_in_dict_func(duo,dico_families)
        if len(res['prediction']) == 0:
            reversed_duo = list(reversed(duo))
            res2 = search_duo_in_dict_func(reversed_duo,dico_families)
            if len(res2['prediction']) >0 or (len(res2['ambigious'])>0 and len(res['ambigious'])==0):
                return res2
        return res

def search_in_df(max_occurence,dico_families):
        if len(max_occurence)>1: # Search for tuple in dictionnary
            res = prediction_duo([max_occurence[0][0],max_occurence[1][0]],dico_families)
            if len(res['prediction']) == 0:
                if len(max_occurence) >2:
                    res2 = prediction_duo([max_occurence[0][0],max_occurence[2][0]],dico_families)
                    if len(res2['prediction']) == 0:
                        res3 = prediction_duo([max_occurence[1][0],max_occurence[2][0]],dico_families)
                        if len(res3['prediction']) > 0:
                            return particular_prediction_processing(res3,max_occurence,dico_families)
                    else:
                        return particular_prediction_processing(res2,max_occurence,dico_families)
            else:
                return particular_prediction_processing(res,max_occurence,dico_families)

        return prediction_word(max_occurence,dico_families)
        
    
def deep_search_in_df(predicted_type_dico, occurence,dico_families,keep_ambigious=True):
    if (len(predicted_type_dico['prediction']) + len(predicted_type_dico['ambigious'])) == 0:
        occurence_sorted = dict(sorted(occurence.items(), key=lambda item: item[1],reverse=True))
        res= {"prediction": [], "ambigious": []}
        for key in occurence_sorted.keys():
            res = search_word_in_dict_func(key,dico_families)
            if keep_ambigious:
                if (len(res['prediction']) + len(res['ambigious'])) >0:
                    return res
            else:
                if len(res['prediction']) >0:
                    return res
    return predicted_type_dico

def max_occurence_vat(occurence_vat):
        occurence_sorted = dict(sorted(occurence_vat.items(), key=lambda item: item[1],reverse=True))
        for key in occurence_sorted.keys():
            if not pd.isna(key) :
                return key


def ambigious_processing(predicted_type_dico,max_occurence_vat,occurence, dico_families):
    if len(predicted_type_dico["ambigious"]) >0 and not pd.isna(max_occurence_vat):
        res= {"prediction": [], "ambigious": []}
        for key in predicted_type_dico["ambigious"]:
            if max_occurence_vat in dico_families[key]['vat']:
                res["prediction"].append(key)
        if len(res["prediction"]) == 1: #Si avec les vat l'algo trouve plusieurs predictions, alors ça reste ambiguë
            return res 
        else: # sinon c'est qu'il n'a rien trouvé ou que c'est ambigue
            res2 = deep_search_in_df({"prediction": [], "ambigious": []},occurence,dico_families,False)
            if len(res2["prediction"]) < 1:
                return predicted_type_dico
            else:
                return res2
    return predicted_type_dico

def final_predicted_type(predicted_type_dico):
    if len(predicted_type_dico["ambigious"]) >0:
        res = 'ambigious'
        for type in predicted_type_dico["ambigious"]:
            res = res+' '+type
        return res
    elif len(predicted_type_dico["prediction"]) >0 :
        return predicted_type_dico["prediction"][0]
    
    return 'inconnu'

def predicted_type_level_1(predicted_type_level_2,dico_level,predicted_type_dico3=None):
        if predicted_type_level_2 == 'inconnu':
                return "inconnu"
        for key in dico_level.keys():
                if predicted_type_level_2 in dico_level[key]["sublevel"]:
                        return key
                elif predicted_type_dico3 is not None and 'ambigious' in  predicted_type_level_2:
                        if set(predicted_type_dico3['ambigious']).issubset(set(dico_level[key]["sublevel"])):
                                return key
                        else:
                                return "Ambigious"

        return "Unknown level 2 type"
    
    
    
    



