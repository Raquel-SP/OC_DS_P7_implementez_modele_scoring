""" 
    Personal library containing preprocessing fonctions.
"""

#! /usr/bin/env python3
# coding: utf-8

# ====================================================================
# PREPROCESSING TOOLS -  projet 7 Openclassrooms
# Version : 0.0.0 - Created by RSP 08/06/2023
# ====================================================================
# from IPython.core.display import display
# from datetime import datetime
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.utils import check_random_state
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.feature_selection import RFECV
from pprint import pprint

# --------------------------------------------------------------------
# -- VERSION
# --------------------------------------------------------------------
__version__ = '0.0.0'


# --------------------------------------------------------------------
# -- REDUCE MEMORY USAGE
# --------------------------------------------------------------------

def reduce_mem_usage(data, verbose=True):
    # source: https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
    '''
    This function is used to reduce the memory usage by converting the datatypes of a pandas
    DataFrame withing required limits.
    '''

    start_mem = data.memory_usage().sum() / 1024**2
    if verbose:
        print('-' * 79)
        print('Memory usage of dataframe is: {:.2f} MB'.format(start_mem))

    for col in data.columns:
        col_type = data[col].dtype

        #  Float et int
        if col_type != object:
            c_min = data[col].min()
            c_max = data[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(
                        np.int8).min and c_max < np.iinfo(
                        np.int8).max:
                    data[col] = data[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    data[col] = data[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    data[col] = data[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    data[col] = data[col].astype(np.int64)
            else:
                if c_min > np.finfo(
                        np.float16).min and c_max < np.finfo(
                        np.float16).max:
                    data[col] = data[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    data[col] = data[col].astype(np.float32)
                else:
                    data[col] = data[col].astype(np.float64)


    end_mem = data.memory_usage().sum() / 1024**2
    if verbose:
        print('Memory usage after optimization is : {:.2f} MB'.format(end_mem))
        print('Reduction of {:.1f}%'.format(
            100 * (start_mem - end_mem) / start_mem))
        print('-' * 79)

    return data


def convert_types(dataframe, print_info=False):

    original_memory = dataframe.memory_usage().sum()

    # Iterate through each column
    for c in dataframe:

        # Convert ids and booleans to integers
        if ('SK_ID' in c):
            dataframe[c] = dataframe[c].fillna(0).astype(np.int32)

        # Convert objects to category
        elif (dataframe[c].dtype == 'object') and (dataframe[c].nunique() < dataframe.shape[0]):
            dataframe[c] = dataframe[c].astype('category')

        # Booleans mapped to integers
        elif list(dataframe[c].unique()) == [1, 0]:
            dataframe[c] = dataframe[c].astype(bool)

        # Float64 to float32
        elif dataframe[c].dtype == float:
            dataframe[c] = dataframe[c].astype(np.float32)

        # Int64 to int32
        elif dataframe[c].dtype == int:
            dataframe[c] = dataframe[c].astype(np.int32)

    new_memory = dataframe.memory_usage().sum()

    if print_info:
        print(
            f'Origin memory Usage : {round(original_memory / 1e9, 2)} Gb.')
        print(
            f'Memory Usage after data type modification: {round(new_memory / 1e9, 2)} Gb.')

    return dataframe



# --------------------------------------------------------------------
# -- TRANSLATION OF GROUP/SUBGROUP NAMES
# --------------------------------------------------------------------

def transl_values(dataframe, column_translate, dictionary):
    """
    Translate the transmitted dataframe column values by the dictionary value
    ----------
    @param IN : dataframe : DataFrame
                column_translate : column whose values we want to translate
                dictionary : dictionary key=to replace,
                             value = the required replacement text
    @param OUT :None
    """
    for key, value in dictionary.items():
        dataframe[column_translate] = dataframe[column_translate].replace(key, value)
        
        
def transl_values_variables(dataframe, list_columns_translate, dictionary):
    """
    Traduire les valeurs de la colonne du dataframe transmis par la valeur du dictionnaire
    ----------
    @param IN : dataframe : DataFrame
                list_columns_translate : list of columns whose values we want to translate
                dictionary : dictionary key=to replace,
                             value = the required replacement text
    @param OUT :None
    """
    for col in list_columns_translate:
        for key, value in dictionary.items():
            dataframe[col] = dataframe[col].replace(key, value)
        