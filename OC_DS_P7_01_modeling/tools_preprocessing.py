""" 
    Personal library containing preprocessing functions.
"""

# ! /usr/bin/env python3
# coding: utf-8

# ====================================================================
# PREPROCESSING TOOLS -  project 7 Openclassrooms
# Version : 0.0.0 - Created by RSP 08/06/2023
# ====================================================================

import datetime
import pickle
import numpy as np
from xgboost import XGBRegressor

# --------------------------------------------------------------------
# -- VERSION
# --------------------------------------------------------------------
__version__ = '0.0.0'


# --------------------------------------------------------------------
# -- REDUCE MEMORY USAGE
# --------------------------------------------------------------------

def reduce_mem_usage(data, verbose=True):
    # source: https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
    """
    This function is used to reduce the memory usage by converting the datatypes of a pandas
    DataFrame withing required limits.
    """

    start_mem = data.memory_usage().sum() / 1024 ** 2
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
                        np.int8).min and c_max < np.iinfo(np.int8).max:
                    data[col] = data[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    data[col] = data[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    data[col] = data[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    data[col] = data[col].astype(np.int64)
            else:
                if c_min > np.finfo(
                        np.float16).min and c_max < np.finfo(np.float16).max:
                    data[col] = data[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    data[col] = data[col].astype(np.float32)
                else:
                    data[col] = data[col].astype(np.float64)

    end_mem = data.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('Memory usage after optimization is : {:.2f} MB'.format(end_mem))
        print('Reduction of {:.1f}%'.format(
            100 * (start_mem - end_mem) / start_mem))
        print('-' * 79)

    return data


# --------------------------------------------------------------------
# -- TRANSLATION OF GROUP/SUBGROUP NAMES
# --------------------------------------------------------------------

def transl_values(dataframe, column_translate, dictionary):
    """
    Translate the transmitted dataframe column values by the dictionary value
    ----------
    @parameters : dataframe : DataFrame
                column_translate : column whose values we want to translate
                dictionary : dictionary key=to replace,
                             value = the required replacement text
    @returns :None
    """
    for key, value in dictionary.items():
        dataframe[column_translate] = dataframe[column_translate].replace(key, value)


# --------------------------------------------------------------------
# -- PREDICT MISSING VALUES OF EXT_SOURCE FEATURES
# --------------------------------------------------------------------

def ext_source_values_predictor(application_train, application_test):
    """
    Function to predict the missing values of EXT_SOURCE features

    Inputs: application_train, application_test

    Returns:  None
    """
    file_directory = "../P7_scoring_credit/preprocessing/"

    start = datetime.datetime.now()
    print("\nPredicting the missing values of EXT_SOURCE columns...")

    # predicting the EXT_SOURCE missing values
    # using only numeric columns for predicting the EXT_SOURCES
    columns_for_modelling = list(set(application_test.dtypes[application_test.dtypes != 'object'].index.tolist())
                                 - {'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'SK_ID_CURR'})
    with open(file_directory + 'columns_for_ext_values_predictor.pkl', 'wb') as f:
        pickle.dump(columns_for_modelling, f)

    # we'll train an XGB Regression model for predicting missing EXT_SOURCE values
    # we will predict in the order of least number of missing value columns to max.
    for ext_col in ['EXT_SOURCE_2', 'EXT_SOURCE_3', 'EXT_SOURCE_1']:
        # X_model - datapoints which do not have missing values of given column
        # Y_train - values of column trying to predict with non missing values
        # X_train_missing - datapoints in application_train with missing values
        # X_test_missing - datapoints in application_test with missing values
        X_model, X_train_missing, X_test_missing, Y_train = application_train[~application_train[ext_col].isna()][
                                                                columns_for_modelling], \
                                                            application_train[application_train[ext_col].isna()][
                                                                columns_for_modelling], \
                                                            application_test[application_test[ext_col].isna()][
                                                                columns_for_modelling], \
                                                            application_train[ext_col][
                                                                ~application_train[ext_col].isna()]
        xg = XGBRegressor(n_estimators=1000, max_depth=3, learning_rate=0.1, n_jobs=-1, random_state=59)
        xg.fit(X_model, Y_train)
        # dumping the model to pickle file
        with open(file_directory + f'nan_{ext_col}_xgbr_model.pkl', 'wb') as f:
            pickle.dump(xg, f)
        application_train[ext_col][application_train[ext_col].isna()] = xg.predict(X_train_missing)
        application_test[ext_col][application_test[ext_col].isna()] = xg.predict(X_test_missing)

        # adding the predicted column to columns for modelling for next column's prediction
        columns_for_modelling = columns_for_modelling + [ext_col]

    print("Done.")
    print(f"Time elapsed = {datetime.datetime.now() - start}")


# ---------------------------------------------------------------------------------
# -- MERGE ALL THE TABLES TOGETHER WITH THE APPLICATION_TRAIN AND APPLICATION_TEST
# ---------------------------------------------------------------------------------

def merge_all_tables(application_train_ML, application_test_ML,
                     bureau_ML,
                     previous_application_ML, installments_payments_ML,
                     POS_CASH_balance_ML, cc_balance_ML):
    """
    Function to merge all the tables together with the application_train_ML and application_test_ML tables
    on SK_ID_CURR.

    Inputs:
        All the previously pre-processed Tables.

    Returns:
        Single merged tables, one for training data and one for test data
    """

    # merging application_train_ML and application_test_ML with Aggregated bureau table
    app_train_merged = application_train_ML.merge(bureau_ML, on='SK_ID_CURR', how='left')
    app_test_merged = application_test_ML.merge(bureau_ML, on='SK_ID_CURR', how='left')
    # merging with aggregated previous_applications
    app_train_merged = app_train_merged.merge(previous_application_ML, on='SK_ID_CURR', how='left')
    app_test_merged = app_test_merged.merge(previous_application_ML, on='SK_ID_CURR', how='left')
    # merging with aggregated installments tables
    app_train_merged = app_train_merged.merge(installments_payments_ML, on='SK_ID_CURR', how='left')
    app_test_merged = app_test_merged.merge(installments_payments_ML, on='SK_ID_CURR', how='left')
    # merging with aggregated POS_Cash balance table
    app_train_merged = app_train_merged.merge(POS_CASH_balance_ML, on='SK_ID_CURR', how='left')
    app_test_merged = app_test_merged.merge(POS_CASH_balance_ML, on='SK_ID_CURR', how='left')
    # merging with aggregated credit card table
    app_train_merged = app_train_merged.merge(cc_balance_ML, on='SK_ID_CURR', how='left')
    app_test_merged = app_test_merged.merge(cc_balance_ML, on='SK_ID_CURR', how='left')

    return reduce_mem_usage(app_train_merged), reduce_mem_usage(app_test_merged)

