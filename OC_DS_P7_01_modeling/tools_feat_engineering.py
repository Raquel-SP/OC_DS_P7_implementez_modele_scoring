""" 
    Personal library containing feature engineering fonctions.
"""

# ! /usr/bin/env python3
# coding: utf-8

# ====================================================================
# PREPROCESSING TOOLS -  projet 7 Openclassrooms
# Version : 0.0.0 - Created by RSP 08/06/2023
# ====================================================================

from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pprint import pprint

from sklearn.feature_selection import RFECV
from sklearn.neighbors import KNeighborsClassifier

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# --------------------------------------------------------------------
# -- VERSION
# --------------------------------------------------------------------
__version__ = '0.0.0'


# --------------------------------------------------------------------
# -- FEATURE ENGINEERING FOR APPLICATION_TRAIN / TEST DATASETS
# --------------------------------------------------------------------

class feat_eng_application_train_test:
    """
    Preprocess the application_train and application_test tables.
    Contains 11 member functions:
        1. init method
        2. load_dataframe method
        3. ext_source_values_predictor method
        4. numeric_feature_engineering method
        5. neighbors_EXT_SOURCE_feature method
        6. categorical_interaction_features method
        7. response_fit method
        8. response_transform method
        9. cnt_payment_prediction method
        10. main method
    """

    def __init__(self, file_directory='', verbose=True, dump_to_pickle=False):
        """
        This function is used to initialize the class members

        Inputs:
            self
            file_directory: Path, str, default = ''
                The path where the file exists. Include a '/' at the end of the path in input
            verbose: bool, default = True
                Whether to enable verbosity or not
            dump_to_pickle: bool, default = False
                Whether to pickle the final preprocessed table or not

        Returns:
            None
        """

        self.verbose = verbose
        self.dump_to_pickle = dump_to_pickle
        self.file_directory = "../P7_scoring_credit/preprocessing/"
        self.file_directory_temp = "../P7_scoring_credit/preprocessing/temp/"
        self.path_sav_appltrain_wonan = "../P7_scoring_credit/preprocessing/application_train_wo_nan.pkl"
        self.path_sav_appltest_wonan = "../P7_scoring_credit/preprocessing/application_test_wo_nan.pkl"

    def load_dataframes(self):
        """
        Function to load the application_train.csv and application_test.csv DataFrames.

        Inputs:
            self

        Returns:
            None
        """

        if self.verbose:
            self.start = datetime.now()
            print('###################################################')
            print('#        Pre-processing application_train         #')
            print('#        Pre-processing application_test          #')
            print('###################################################')
            print("\nLoading the DataFrames into memory...")

        with open(self.path_sav_appltrain_wonan, 'rb') as f:
            self.application_train = pickle.load(f)
        with open(self.path_sav_appltest_wonan, 'rb') as f:
            self.application_test = pickle.load(f)
        self.initial_shape = self.application_train.shape

        if self.verbose:
            print("Loaded application_train and application_test")
            print(f"Time Taken to load = {datetime.now() - self.start}")

    def ext_source_values_predictor(self):
        """
        Function to predict the missing values of EXT_SOURCE features

        Inputs: self
        Returns: None
        """

        if self.verbose:
            start = datetime.now()
            print("\nPredicting the missing values of EXT_SOURCE columns...")

        # predicting the EXT_SOURCE missing values
        # using only numeric columns for predicting the EXT_SOURCES
        columns_for_modelling = list(
            set(self.application_test.dtypes[self.application_test.dtypes != 'object'].index.tolist())
            - {'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'SK_ID_CURR'})
        with open(self.file_directory_temp + 'columns_for_ext_values_predictor.pkl', 'wb') as f:
            pickle.dump(columns_for_modelling, f)

        # we'll train an XGB Regression model for predicting missing EXT_SOURCE values
        # we will predict in the order of least number of missing value columns to max.
        for ext_col in ['EXT_SOURCE_2', 'EXT_SOURCE_3', 'EXT_SOURCE_1']:
            # X_model - datapoints which do not have missing values of given column
            # Y_train - values of column trying to predict with non missing values
            # X_train_missing - datapoints in application_train with missing values
            # X_test_missing - datapoints in application_test with missing values
            X_model, X_train_missing, X_test_missing, Y_train = \
                self.application_train[~self.application_train[ext_col].isna()][columns_for_modelling], \
                self.application_train[
                    self.application_train[ext_col].isna()][columns_for_modelling], self.application_test[
                    self.application_test[ext_col].isna()][columns_for_modelling], self.application_train[
                    ext_col][~self.application_train[ext_col].isna()]
            xg = XGBRegressor(n_estimators=1000, max_depth=3, learning_rate=0.1, n_jobs=-1, random_state=59)
            xg.fit(X_model, Y_train)
            # dumping the model to pickle file
            with open(self.file_directory_temp + f'nan_{ext_col}_xgbr_model.pkl', 'wb') as f:
                pickle.dump(xg, f)

            self.application_train[ext_col][self.application_train[ext_col].isna()] = xg.predict(X_train_missing)
            self.application_test[ext_col][self.application_test[ext_col].isna()] = xg.predict(X_test_missing)

            # adding the predicted column to columns for modelling for next column's prediction
            columns_for_modelling = columns_for_modelling + [ext_col]

        if self.verbose:
            print("Done.")
            print(f"Time elapsed = {datetime.now() - start}")

    def numeric_feature_engineering(self, data):
        """
        Function to perform feature engineering on numeric columns based on domain knowledge.

        Inputs:
            self
            data: DataFrame
                The tables of whose features are to be generated

        Returns:
            None
        """

        # income and credit features
        data['CREDIT_INCOME_RATIO'] = data['AMT_CREDIT'] / (data['AMT_INCOME_TOTAL'] + 0.00001)
        data['CREDIT_ANNUITY_RATIO'] = data['AMT_CREDIT'] / (data['AMT_ANNUITY'] + 0.00001)
        data['ANNUITY_INCOME_RATIO'] = data['AMT_ANNUITY'] / (data['AMT_INCOME_TOTAL'] + 0.00001)
        data['INCOME_ANNUITY_DIFF'] = data['AMT_INCOME_TOTAL'] - data['AMT_ANNUITY']
        data['CREDIT_GOODS_RATIO'] = data['AMT_CREDIT'] / (data['AMT_GOODS_PRICE'] + 0.00001)
        data['CREDIT_GOODS_DIFF'] = data['AMT_CREDIT'] - data['AMT_GOODS_PRICE'] + 0.00001
        data['GOODS_INCOME_RATIO'] = data['AMT_GOODS_PRICE'] / (data['AMT_INCOME_TOTAL'] + 0.00001)
        data['INCOME_EXT_RATIO'] = data['AMT_INCOME_TOTAL'] / (data['EXT_SOURCE_3'] + 0.00001)
        data['CREDIT_EXT_RATIO'] = data['AMT_CREDIT'] / (data['EXT_SOURCE_3'] + 0.00001)
        # age ratios and diffs
        data['AGE_EMPLOYED_DIFF'] = data['DAYS_BIRTH'] - data['DAYS_EMPLOYED']
        data['EMPLOYED_TO_AGE_RATIO'] = data['DAYS_EMPLOYED'] / (data['DAYS_BIRTH'] + 0.00001)
        # car ratios
        data['CAR_EMPLOYED_DIFF'] = data['OWN_CAR_AGE'] - data['DAYS_EMPLOYED']
        data['CAR_EMPLOYED_RATIO'] = data['OWN_CAR_AGE'] / (data['DAYS_EMPLOYED'] + 0.00001)
        data['CAR_AGE_DIFF'] = data['DAYS_BIRTH'] - data['OWN_CAR_AGE']
        data['CAR_AGE_RATIO'] = data['OWN_CAR_AGE'] / (data['DAYS_BIRTH'] + 0.00001)
        # flag contacts sum
        data['FLAG_CONTACTS_SUM'] = data['FLAG_MOBIL'] + data['FLAG_EMP_PHONE'] + data['FLAG_WORK_PHONE'] + data[
            'FLAG_CONT_MOBILE'] + data['FLAG_PHONE'] + data['FLAG_EMAIL']

        data['HOUR_PROCESS_CREDIT_MUL'] = data['AMT_CREDIT'] * data['HOUR_APPR_PROCESS_START']
        # family members
        data['CNT_NON_CHILDREN'] = data['CNT_FAM_MEMBERS'] - data['CNT_CHILDREN']
        data['CHILDREN_INCOME_RATIO'] = data['CNT_CHILDREN'] / (data['AMT_INCOME_TOTAL'] + 0.00001)
        data['PER_CAPITA_INCOME'] = data['AMT_INCOME_TOTAL'] / (data['CNT_FAM_MEMBERS'] + 1)
        # region ratings
        data['REGIONS_RATING_INCOME_MUL'] = (data['REGION_RATING_CLIENT'] + data['REGION_RATING_CLIENT_W_CITY']) * data[
            'AMT_INCOME_TOTAL'] / 2
        data['REGION_RATING_MAX'] = [max(ele1, ele2) for ele1, ele2 in
                                     zip(data['REGION_RATING_CLIENT'], data['REGION_RATING_CLIENT_W_CITY'])]
        data['REGION_RATING_MAX'] = [min(ele1, ele2) for ele1, ele2 in
                                     zip(data['REGION_RATING_CLIENT'], data['REGION_RATING_CLIENT_W_CITY'])]
        data['REGION_RATING_MEAN'] = (data['REGION_RATING_CLIENT'] + data['REGION_RATING_CLIENT_W_CITY']) / 2
        data['REGION_RATING_MUL'] = data['REGION_RATING_CLIENT'] * data['REGION_RATING_CLIENT_W_CITY']
        # flag regions
        data['FLAG_REGIONS'] = data['REG_REGION_NOT_LIVE_REGION'] + data['REG_REGION_NOT_WORK_REGION'] + data[
            'LIVE_REGION_NOT_WORK_REGION'] + data[
                                   'REG_CITY_NOT_LIVE_CITY'] + data['REG_CITY_NOT_WORK_CITY'] + data[
                                   'LIVE_CITY_NOT_WORK_CITY']
        # ext_sources
        data['EXT_SOURCE_MEAN'] = (data['EXT_SOURCE_1'] + data['EXT_SOURCE_2'] + data['EXT_SOURCE_3']) / 3
        data['EXT_SOURCE_MUL'] = data['EXT_SOURCE_1'] * data['EXT_SOURCE_2'] * data['EXT_SOURCE_3']
        data['EXT_SOURCE_MAX'] = [max(ele1, ele2, ele3) for ele1, ele2, ele3 in
                                  zip(data['EXT_SOURCE_1'], data['EXT_SOURCE_2'], data['EXT_SOURCE_3'])]
        data['EXT_SOURCE_MIN'] = [min(ele1, ele2, ele3) for ele1, ele2, ele3 in
                                  zip(data['EXT_SOURCE_1'], data['EXT_SOURCE_2'], data['EXT_SOURCE_3'])]
        data['EXT_SOURCE_VAR'] = [np.var([ele1, ele2, ele3]) for ele1, ele2, ele3 in
                                  zip(data['EXT_SOURCE_1'], data['EXT_SOURCE_2'], data['EXT_SOURCE_3'])]
        data['WEIGHTED_EXT_SOURCE'] = data.EXT_SOURCE_1 * 2 + data.EXT_SOURCE_2 * 3 + data.EXT_SOURCE_3 * 4
        # apartment scores
        data['APARTMENTS_SUM_AVG'] = data['APARTMENTS_AVG'] + data['BASEMENTAREA_AVG'] + data[
            'YEARS_BEGINEXPLUATATION_AVG'] + data[
                                         'YEARS_BUILD_AVG'] + data['ELEVATORS_AVG'] + data['ENTRANCES_AVG'] + data[
                                         'FLOORSMAX_AVG'] + data['FLOORSMIN_AVG'] + data['LANDAREA_AVG'] + data[
                                         'LIVINGAREA_AVG'] + data['NONLIVINGAREA_AVG']

        data['APARTMENTS_SUM_MODE'] = data['APARTMENTS_MODE'] + data['BASEMENTAREA_MODE'] + data[
            'YEARS_BEGINEXPLUATATION_MODE'] + data[
                                          'YEARS_BUILD_MODE'] + data['ELEVATORS_MODE'] + data['ENTRANCES_MODE'] + data[
                                          'FLOORSMAX_MODE'] + data['FLOORSMIN_MODE'] + data['LANDAREA_MODE'] + data[
                                          'LIVINGAREA_MODE'] + data['NONLIVINGAREA_MODE'] + data['TOTALAREA_MODE']

        data['APARTMENTS_SUM_MEDI'] = data['APARTMENTS_MEDI'] + data['BASEMENTAREA_MEDI'] + data[
            'YEARS_BEGINEXPLUATATION_MEDI'] + data[
                                          'YEARS_BUILD_MEDI'] + data['ELEVATORS_MEDI'] + data['ENTRANCES_MEDI'] + data[
                                          'FLOORSMAX_MEDI'] + data['FLOORSMIN_MEDI'] + data['LANDAREA_MEDI'] + data[
                                          'LIVINGAREA_MEDI'] + data['NONLIVINGAREA_MEDI']
        data['INCOME_APARTMENT_AVG_MUL'] = data['APARTMENTS_SUM_AVG'] * data['AMT_INCOME_TOTAL']
        data['INCOME_APARTMENT_MODE_MUL'] = data['APARTMENTS_SUM_MODE'] * data['AMT_INCOME_TOTAL']
        data['INCOME_APARTMENT_MEDI_MUL'] = data['APARTMENTS_SUM_MEDI'] * data['AMT_INCOME_TOTAL']
        # OBS And DEF
        data['OBS_30_60_SUM'] = data['OBS_30_CNT_SOCIAL_CIRCLE'] + data['OBS_60_CNT_SOCIAL_CIRCLE']
        data['DEF_30_60_SUM'] = data['DEF_30_CNT_SOCIAL_CIRCLE'] + data['DEF_60_CNT_SOCIAL_CIRCLE']
        data['OBS_DEF_30_MUL'] = data['OBS_30_CNT_SOCIAL_CIRCLE'] * data['DEF_30_CNT_SOCIAL_CIRCLE']
        data['OBS_DEF_60_MUL'] = data['OBS_60_CNT_SOCIAL_CIRCLE'] * data['DEF_60_CNT_SOCIAL_CIRCLE']
        data['SUM_OBS_DEF_ALL'] = data['OBS_30_CNT_SOCIAL_CIRCLE'] + data['DEF_30_CNT_SOCIAL_CIRCLE'] + data[
            'OBS_60_CNT_SOCIAL_CIRCLE'] + data['DEF_60_CNT_SOCIAL_CIRCLE']
        data['OBS_30_CREDIT_RATIO'] = data['AMT_CREDIT'] / (data['OBS_30_CNT_SOCIAL_CIRCLE'] + 0.00001)
        data['OBS_60_CREDIT_RATIO'] = data['AMT_CREDIT'] / (data['OBS_60_CNT_SOCIAL_CIRCLE'] + 0.00001)
        data['DEF_30_CREDIT_RATIO'] = data['AMT_CREDIT'] / (data['DEF_30_CNT_SOCIAL_CIRCLE'] + 0.00001)
        data['DEF_60_CREDIT_RATIO'] = data['AMT_CREDIT'] / (data['DEF_60_CNT_SOCIAL_CIRCLE'] + 0.00001)
        # Flag Documents combined
        data['SUM_FLAGS_DOCUMENTS'] = data['FLAG_DOCUMENT_3'] + data['FLAG_DOCUMENT_5'] + data['FLAG_DOCUMENT_6'] + \
                                data['FLAG_DOCUMENT_7'] + data['FLAG_DOCUMENT_8'] + data['FLAG_DOCUMENT_9'] + \
                                data['FLAG_DOCUMENT_11'] + data['FLAG_DOCUMENT_13'] + data['FLAG_DOCUMENT_14'] + \
                                data['FLAG_DOCUMENT_15'] + data['FLAG_DOCUMENT_16'] + data['FLAG_DOCUMENT_17'] + \
                                data['FLAG_DOCUMENT_18'] + data['FLAG_DOCUMENT_19'] + data['FLAG_DOCUMENT_21']
        # details change
        data['DAYS_DETAILS_CHANGE_SUM'] = data['DAYS_LAST_PHONE_CHANGE'] + data['DAYS_REGISTRATION'] + data[
            'DAYS_ID_PUBLISH']
        # enquires
        data['AMT_ENQ_SUM'] = data['AMT_REQ_CREDIT_BUREAU_HOUR'] + data['AMT_REQ_CREDIT_BUREAU_DAY'] + data[
            'AMT_REQ_CREDIT_BUREAU_WEEK'] + data[
                                  'AMT_REQ_CREDIT_BUREAU_MON'] + data['AMT_REQ_CREDIT_BUREAU_QRT'] + data[
                                  'AMT_REQ_CREDIT_BUREAU_YEAR']
        data['ENQ_CREDIT_RATIO'] = data['AMT_ENQ_SUM'] / (data['AMT_CREDIT'] + 0.00001)

        cnt_payment = self.cnt_payment_prediction(data)
        data['EXPECTED_CNT_PAYMENT'] = cnt_payment
        data['EXPECTED_INTEREST'] = data['AMT_ANNUITY'] * data['EXPECTED_CNT_PAYMENT'] - data['AMT_CREDIT']
        data['EXPECTED_INTEREST_SHARE'] = data['EXPECTED_INTEREST'] / (data['AMT_CREDIT'] + 0.00001)
        data['EXPECTED_INTEREST_RATE'] = 2 * 12 * data['EXPECTED_INTEREST'] / (
                data['AMT_CREDIT'] * (data['EXPECTED_CNT_PAYMENT'] + 1))

        return data

    def neighbors_EXT_SOURCE_feature(self):
        """
        Function to generate a feature which contains the means of TARGET of 500 neighbors of a particular row.

        Inputs:
            self

        Returns:
            None
        """

        # https://www.kaggle.com/c/home-credit-default-risk/discussion/64821
        # imputing the mean of 500 nearest neighbor's target values for each application
        # neighbors are computed using EXT_SOURCE feature and CREDIT_ANNUITY_RATIO

        knn = KNeighborsClassifier(500, n_jobs=-1)

        train_data_for_neighbors = self.application_train[
            ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'CREDIT_ANNUITY_RATIO']].fillna(0)
        # saving the training data for neighbors
        with open(self.file_directory_temp + 'TARGET_MEAN_500_Neighbors_training_data.pkl', 'wb') as f:
            pickle.dump(train_data_for_neighbors, f)
        train_target = self.application_train.TARGET
        test_data_for_neighbors = self.application_test[
            ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'CREDIT_ANNUITY_RATIO']].fillna(0)

        knn.fit(train_data_for_neighbors, train_target)
        # pickling the knn model
        with open(self.file_directory_temp + 'KNN_model_TARGET_500_neighbors.pkl', 'wb') as f:
            pickle.dump(knn, f)

        train_500_neighbors = knn.kneighbors(train_data_for_neighbors)[1]
        test_500_neighbors = knn.kneighbors(test_data_for_neighbors)[1]

        # adding the means of targets of 500 neighbors to new column
        self.application_train['TARGET_NEIGHBORS_500_MEAN'] = [self.application_train['TARGET'].iloc[ele].mean() for ele
                                                               in train_500_neighbors]
        self.application_test['TARGET_NEIGHBORS_500_MEAN'] = [self.application_train['TARGET'].iloc[ele].mean() for ele
                                                              in test_500_neighbors]

    def categorical_interaction_features(self, train_data, test_data):
        """
        Function to generate some features based on categorical groupings.

        Inputs:
            self
            train_data, test_data : DataFrames
                train and test dataframes

        Returns:
            Train and test datasets, with added categorical interaction features.
        """

        # now we will create features based on categorical interactions
        columns_to_aggregate_on = [
            ['NAME_CONTRACT_TYPE', 'NAME_INCOME_TYPE', 'OCCUPATION_TYPE'],
            ['CODE_GENDER', 'NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE'],
            ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE'],
            ['NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE', 'OCCUPATION_TYPE'],
            ['OCCUPATION_TYPE', 'ORGANIZATION_TYPE'],
            ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']

        ]
        aggregations = {
            'AMT_ANNUITY': ['mean', 'max', 'min'],
            'ANNUITY_INCOME_RATIO': ['mean', 'max', 'min'],
            'AGE_EMPLOYED_DIFF': ['mean', 'min'],
            'AMT_INCOME_TOTAL': ['mean', 'max', 'min'],
            'APARTMENTS_SUM_AVG': ['mean', 'max', 'min'],
            'APARTMENTS_SUM_MEDI': ['mean', 'max', 'min'],
            'EXT_SOURCE_MEAN': ['mean', 'max', 'min'],
            'EXT_SOURCE_1': ['mean', 'max', 'min'],
            'EXT_SOURCE_2': ['mean', 'max', 'min'],
            'EXT_SOURCE_3': ['mean', 'max', 'min']
        }

        # extracting values
        for group in columns_to_aggregate_on:
            # grouping based on categories
            grouped_interactions = train_data.groupby(group).agg(aggregations)
            grouped_interactions.columns = ['_'.join(ele).upper() + '_AGG_' + '_'.join(group) for ele in
                                            grouped_interactions.columns]
            # saving the grouped interactions to pickle file
            group_name = '_'.join(group)
            with open(self.file_directory_temp + f'Application_train_grouped_interactions_{group_name}.pkl', 'wb') as f:
                pickle.dump(grouped_interactions, f)
            # merging with the original data
            train_data = train_data.join(grouped_interactions, on=group)
            test_data = test_data.join(grouped_interactions, on=group)

        return train_data, test_data

    def response_fit(self, data, column):
        """
        Response Encoding Fit Function
        Function to create a vocabulary with the probability of occurrence of each category for categorical features
        for a given class label.

        Inputs:
            self
            data: DataFrame
                training Dataset
            column: str
                the categorical column for which vocab is to be generated

        Returns:
            Dictionary of probability of occurrence of each category in a particular class label.
        """

        dict_occurrences = {1: {}, 0: {}}
        for label in [0, 1]:
            dict_occurrences[label] = dict(
                (data[column][data.TARGET == label].value_counts() / data[column].value_counts()).fillna(0))

        return dict_occurrences

    def response_transform(self, data, column, dict_mapping):
        """
        Response Encoding Transform Function
        Function to transform the categorical feature into two features, which contain the probability
        of occurrence of that category for each class label.

        Inputs:
            self
            data: DataFrame
                DataFrame whose categorical features are to be encoded
            column: str
                categorical column whose encoding is to be done
            dict_mapping: dict
                Dictionary obtained from Response Fit function for that particular column

        Returns:
            None
        """

        data[column + '_0'] = data[column].map(dict_mapping[0])
        data[column + '_1'] = data[column].map(dict_mapping[1])

    def cnt_payment_prediction(self, data_to_predict):
        """
        Function to predict the Count_payments on Current Loans using data from previous loans.

        Inputs:
            self
            data_to_predict: DataFrame
                the values using which the model would predict the Count_payments on current applications

        Returns:
            Predicted Count_payments of the current applications.
        """

        # https://www.kaggle.com/c/home-credit-default-risk/discussion/64598
        previous_application = pd.read_csv('previous_application.csv')
        train_data = previous_application[['AMT_CREDIT', 'AMT_ANNUITY', 'CNT_PAYMENT']].dropna()
        train_data['CREDIT_ANNUITY_RATIO'] = train_data['AMT_CREDIT'] / (train_data['AMT_ANNUITY'] + 1)
        # value to predict is our CNT_PAYMENT
        train_value = train_data.pop('CNT_PAYMENT')

        # test data would be our application_train data
        test_data = data_to_predict[['AMT_CREDIT', 'AMT_ANNUITY']].fillna(0)
        test_data['CREDIT_ANNUITY_RATIO'] = test_data['AMT_CREDIT'] / (test_data['AMT_ANNUITY'] + 1)

        lgbmr = LGBMRegressor(max_depth=9, n_estimators=5000, n_jobs=-1, learning_rate=0.3,
                              random_state=125, verbose=-100)
        lgbmr.fit(train_data, train_value)
        # dumping the model to pickle file
        with open(self.file_directory_temp + 'cnt_payment_predictor_lgbmr.pkl', 'wb') as f:
            pickle.dump(lgbmr, f)
        # predicting the CNT_PAYMENT for test_data
        cnt_payment = lgbmr.predict(test_data)

        return cnt_payment

    def main(self):
        """
        Function to be called for complete preprocessing of application_train and application_test tables.

        Inputs:
            self

        Returns:
            Final pre=processed application_train and application_test tables.
        """

        # loading the DataFrames first
        self.load_dataframes()
        # predicting the missing values of EXT_SOURCE columns
        self.ext_source_values_predictor()

        # doing the feature engineering
        if self.verbose:
            start = datetime.now()
            print("\nStarting Feature Engineering...")
            print("\nCreating Domain Based Features on Numeric Data")
        # Creating Numeric features based on domain knowledge
        self.application_train = self.numeric_feature_engineering(self.application_train)
        self.application_test = self.numeric_feature_engineering(self.application_test)
        # 500 Neighbors Target mean
        self.neighbors_EXT_SOURCE_feature()
        if self.verbose:
            print("Done.")
            print(f"Time Taken = {datetime.now() - start}")

        if self.verbose:
            start = datetime.now()
            print("Creating features based on Categorical Interactions on some Numeric Features")
        # creating features based on categorical interactions
        self.application_train, self.application_test = self.categorical_interaction_features(self.application_train,
                                                                                              self.application_test)
        if self.verbose:
            print("Done.")
            print(f"Time taken = {datetime.now() - start}")

        # using response coding on categorical features, to keep the dimensionality in check
        # categorical columns to perform response coding on
        categorical_columns_application = self.application_train.dtypes[
            self.application_train.dtypes == 'object'].index.tolist()
        for col in categorical_columns_application:
            # extracting the dictionary with values corresponding to TARGET variable 0 and 1 for each of the categories
            mapping_dictionary = self.response_fit(self.application_train, col)
            # saving the mapping dictionary to pickle file
            with open(self.file_directory_temp + f'Response_coding_dict_{col}.pkl', 'wb') as f:
                pickle.dump(mapping_dictionary, f)
            # mapping this dictionary with our DataFrame
            self.response_transform(self.application_train, col, mapping_dictionary)
            self.response_transform(self.application_test, col, mapping_dictionary)
            # removing the original categorical columns
            _ = self.application_train.pop(col)
            _ = self.application_test.pop(col)

        if self.verbose:
            print('Done preprocessing appplication_train and application_test.')
            print(f"\nInitial Size of application_train: {self.initial_shape}")
            print(
                f'Size of application_train after Pre-Processing and Feature Engineering: {self.application_train.shape}')
            print(f'\nTotal Time Taken = {datetime.now() - self.start}')

        if self.dump_to_pickle:
            if self.verbose:
                print(
                    '\nPickling application_train and application_test after feature engineering to application_train_aft_feat_eng.pkl and application_test_aft_feat_eng, respectively.')
            with open(self.file_directory + 'application_train_aft_feat_eng.pkl', 'wb') as f:
                pickle.dump(self.application_train, f)
            with open(self.file_directory + 'application_test_aft_feat_eng.pkl', 'wb') as f:
                pickle.dump(self.application_test, f)
            if self.verbose:
                print('Done.')
        if self.verbose:
            print('-' * 100)

        return self.application_train, self.application_test


# --------------------------------------------------------------------
# -- FEATURE ENGINEERING FOR BUREAU_BALANCE
# --------------------------------------------------------------------    

class feat_eng_bureau_balance:
    """
    Feature engineering of bureau_balance
    Contains 4 member functions:
        1. init method
        2. feat_eng_bureau_balance method
        4. main method
    """

    def __init__(self, file_directory='', verbose=True, dump_to_pickle=False):
        """
        This function is used to initialize the class members

        Inputs:
            self
            file_directory: Path, str, default = ''
                The path where the file exists. Include a '/' at the end of the path in input
            verbose: bool, default = True
                Whether to enable verbosity or not
            dump_to_pickle: bool, default = False
                Whether to pickle the final preprocessed table or not

        Returns:
            None
        """

        self.verbose = verbose
        self.dump_to_pickle = dump_to_pickle
        self.start = datetime.now()
        self.file_directory = "../P7_scoring_credit/preprocessing/"
        self.path_sav_bureaubal_aftproc = "../P7_scoring_credit/preprocessing/bureaubal_aftproc.pkl"

    def feat_eng_bureau_balance(self):
        """
        Function to transform and create variables from bureau_balance
        This function first loads the table into memory, does some feature engineering

        Inputs:
            self

        Returns:
            bureau_balance table after feature engineering
        """

        if self.verbose:
            print('########################################################')
            print('#          Feature engineering bureau_balance          #')
            print('########################################################')
            print("\nLoading the DataFrame, bureau_balance, into memory...")

        with open(self.path_sav_bureaubal_aftproc, 'rb') as f:
            bureau_balance = pickle.load(f)

        if self.verbose:
            print("Loaded bureau_balance.csv")
            print(f"Time Taken to load = {datetime.now() - self.start}")
            print("\nStarting Feature Engineering...")

        # sorting the bureau_balance in ascending order of month and by the bureau SK_ID
        # this is done so as to make the rolling exponential average easily for previous months till current month
        bureau_balance = bureau_balance.sort_values(by=['SK_ID_BUREAU', 'MONTHS_BALANCE'], ascending=[0, 0])
        # we will do exponential weighted average on the encoded status
        # this is because if a person had a bad status 2 years ago, it should be given less weightage today
        # we keep the latent variable alpha = 0.8 
        # doing this for both weighted status and the status itself
        bureau_balance['EXP_WEIGHTED_STATUS'] = bureau_balance.groupby('SK_ID_BUREAU')['WEIGHTED_STATUS'].transform(
            lambda x: x.ewm(alpha=0.8).mean())
        bureau_balance['EXP_ENCODED_STATUS'] = bureau_balance.groupby('SK_ID_BUREAU')['STATUS'].transform(
            lambda x: x.ewm(alpha=0.8).mean())

        if self.verbose:
            print("Halfway through. A little bit more patience...")
            print(f"Total Time Elapsed = {datetime.now() - self.start}")

        # we can see that these datapoints are for 96 months i.e. 8 years.
        # so we will extract the means, and exponential averages for each year separately
        # first we convert month to year
        bureau_balance['MONTHS_BALANCE'] = bureau_balance['MONTHS_BALANCE'] // 12

        # defining our aggregations
        aggregations_basic = {
            'MONTHS_BALANCE': ['mean', 'max'],
            'STATUS': ['mean', 'max', 'first'],
            'WEIGHTED_STATUS': ['mean', 'sum', 'first'],
            'EXP_ENCODED_STATUS': ['last'],
            'EXP_WEIGHTED_STATUS': ['last']}

        # we will be finding aggregates for each year too
        aggregations_for_year = {
            'STATUS': ['mean', 'max', 'last', 'first'],
            'WEIGHTED_STATUS': ['mean', 'max', 'first', 'last'],
            'EXP_WEIGHTED_STATUS': ['last'],
            'EXP_ENCODED_STATUS': ['last']}

        # aggregating over whole dataset first
        aggregated_bureau_balance = bureau_balance.groupby(['SK_ID_BUREAU']).agg(aggregations_basic)
        aggregated_bureau_balance.columns = ['_'.join(ele).upper() for ele in aggregated_bureau_balance.columns]

        # aggregating some of the features separately for latest 2 years
        aggregated_bureau_years = pd.DataFrame()
        for year in range(2):
            year_group = bureau_balance[bureau_balance['MONTHS_BALANCE'] == year].groupby('SK_ID_BUREAU').agg(
                aggregations_for_year)
            year_group.columns = ['_'.join(ele).upper() + '_YEAR_' + str(year) for ele in year_group.columns]

            if year == 0:
                aggregated_bureau_years = year_group
            else:
                aggregated_bureau_years = aggregated_bureau_years.merge(year_group, on='SK_ID_BUREAU', how='outer')

        # aggregating for rest of the years
        aggregated_bureau_rest_years = bureau_balance[bureau_balance.MONTHS_BALANCE > year].groupby(
            ['SK_ID_BUREAU']).agg(aggregations_for_year)
        aggregated_bureau_rest_years.columns = ['_'.join(ele).upper() + '_YEAR_REST' for ele in
                                                aggregated_bureau_rest_years.columns]

        # merging with rest of the years
        aggregated_bureau_years = aggregated_bureau_years.merge(aggregated_bureau_rest_years, on='SK_ID_BUREAU',
                                                                how='outer')
        aggregated_bureau_balance = aggregated_bureau_balance.merge(aggregated_bureau_years, on='SK_ID_BUREAU',
                                                                    how='inner')

        if self.dump_to_pickle:
            if self.verbose:
                print('\nPickling bureau_balance after feature engineering to bureau_balance_aft_feat_eng.pkl')
            with open(self.file_directory + 'bureau_balance_aft_feat_eng.pkl', 'wb') as f:
                pickle.dump(aggregated_bureau_balance, f)
            if self.verbose:
                print('Done.')

        return aggregated_bureau_balance

    def main(self):
        """
        Function to be called for feature engineering of bureau_balance table.

        Inputs:
            self

        Returns:
            Final burea_balance table after feature engineering
        """

        # preprocessing the bureau_balance first
        aggregated_bureau_balance = self.feat_eng_bureau_balance()

        return aggregated_bureau_balance


# --------------------------------------------------------------------
# -- FEATURE ENGINEERING FOR BUREAU
# --------------------------------------------------------------------

class feat_eng_bureau:
    """
    Preprocess the tables bureau.
    Contains 4 member functions:
        1. init method
        3. feat_eng_bureau method
        4. main method
    """

    def __init__(self, file_directory='', verbose=True, dump_to_pickle=False):
        """
        This function is used to initialize the class members

        Inputs:
            self
            file_directory: Path, str, default = ''
                The path where the file exists. Include a '/' at the end of the path in input
            verbose: bool, default = True
                Whether to enable verbosity or not
            dump_to_pickle: bool, default = False
                Whether to pickle the final preprocessed table or not

        Returns:
            None
        """

        self.verbose = verbose
        self.dump_to_pickle = dump_to_pickle
        self.start = datetime.now()
        self.file_directory = "../P7_scoring_credit/preprocessing/"
        self.path_sav_bureau_aftproc = '../P7_scoring_credit/preprocessing/bureau_aftproc.pkl'
        self.path_sav_bureaubal_aft_feateng = '../P7_scoring_credit/preprocessing/bureau_balance_aft_feat_eng.pkl'

    def feat_eng_bureau(self):
        """
        Function to preprocess the bureau table and merge it with the result of feature engineering of bureau_balance
        table. Finally aggregates the data over SK_ID_CURR for it to be merged with application_train table.

        Inputs:
            self

        Returns:
            Final preprocessed, merged and aggregated bureau table
        """

        if self.verbose:
            start2 = datetime.now()
            print('###############################################')
            print('#          Feature engineering bureau         #')
            print('###############################################')
            print("\nLoading the DataFrame, bureau, into memory...")

        with open(self.path_sav_bureau_aftproc, 'rb') as f:
            bureau = pickle.load(f)

        if self.verbose:
            print("Loaded bureau")
            print(f"Time Taken to load = {datetime.now() - start2}")
            print("\nStarting Data Cleaning and Feature Engineering...")

        with open(self.path_sav_bureaubal_aft_feateng, 'rb') as f:
            aggr_bureau_balance = pickle.load(f)

        # merging it with aggregated bureau_balance on 'SK_ID_BUREAU'
        bureau_merged = bureau.merge(aggr_bureau_balance, on='SK_ID_BUREAU', how='left')

        # engineering some features based on domain knowledge
        bureau_merged['CREDIT_DURATION'] = np.abs(bureau_merged['DAYS_CREDIT'] - bureau_merged['DAYS_CREDIT_ENDDATE'])
        bureau_merged['FLAG_OVERDUE_RECENT'] = [0 if ele == 0 else 1 for ele in bureau_merged['CREDIT_DAY_OVERDUE']]
        bureau_merged['MAX_AMT_OVERDUE_DURATION_RATIO'] = bureau_merged['AMT_CREDIT_MAX_OVERDUE'] / (
                bureau_merged['CREDIT_DURATION'] + 0.00001)
        bureau_merged['CURRENT_AMT_OVERDUE_DURATION_RATIO'] = bureau_merged['AMT_CREDIT_SUM_OVERDUE'] / (
                bureau_merged['CREDIT_DURATION'] + 0.00001)
        bureau_merged['AMT_OVERDUE_DURATION_LEFT_RATIO'] = bureau_merged['AMT_CREDIT_SUM_OVERDUE'] / (
                bureau_merged['DAYS_CREDIT_ENDDATE'] + 0.00001)
        bureau_merged['CNT_PROLONGED_MAX_OVERDUE_MUL'] = bureau_merged['CNT_CREDIT_PROLONG'] * bureau_merged[
            'AMT_CREDIT_MAX_OVERDUE']
        bureau_merged['CNT_PROLONGED_DURATION_RATIO'] = bureau_merged['CNT_CREDIT_PROLONG'] / (
                bureau_merged['CREDIT_DURATION'] + 0.00001)
        bureau_merged['CURRENT_DEBT_TO_CREDIT_RATIO'] = bureau_merged['AMT_CREDIT_SUM_DEBT'] / (
                bureau_merged['AMT_CREDIT_SUM'] + 0.00001)
        bureau_merged['CURRENT_CREDIT_DEBT_DIFF'] = bureau_merged['AMT_CREDIT_SUM'] - bureau_merged[
            'AMT_CREDIT_SUM_DEBT']
        bureau_merged['AMT_ANNUITY_CREDIT_RATIO'] = bureau_merged['AMT_ANNUITY'] / (
                bureau_merged['AMT_CREDIT_SUM'] + 0.00001)
        bureau_merged['CREDIT_ENDDATE_UPDATE_DIFF'] = np.abs(
            bureau_merged['DAYS_CREDIT_UPDATE'] - bureau_merged['DAYS_CREDIT_ENDDATE'])

        # now we will be aggregating the bureau_merged df with respect to 'SK_ID_CURR' so as to merge it with application_train later
        # firstly we will aggregate the columns based on the category of CREDIT_ACTIVE
        aggregations_CREDIT_ACTIVE = {
            'DAYS_CREDIT': ['mean', 'min', 'max', 'last'],
            'CREDIT_DAY_OVERDUE': ['mean', 'max'],
            'DAYS_CREDIT_ENDDATE': ['mean', 'max'],
            'DAYS_ENDDATE_FACT': ['mean', 'min'],
            'AMT_CREDIT_MAX_OVERDUE': ['max', 'sum'],
            'CNT_CREDIT_PROLONG': ['max', 'sum'],
            'AMT_CREDIT_SUM': ['sum', 'max'],
            'AMT_CREDIT_SUM_DEBT': ['sum'],
            'AMT_CREDIT_SUM_LIMIT': ['max', 'sum'],
            'AMT_CREDIT_SUM_OVERDUE': ['max', 'sum'],
            'DAYS_CREDIT_UPDATE': ['mean', 'min'],
            'AMT_ANNUITY': ['mean', 'sum', 'max'],
            'CREDIT_DURATION': ['max', 'mean'],
            'FLAG_OVERDUE_RECENT': ['sum'],
            'MAX_AMT_OVERDUE_DURATION_RATIO': ['max', 'sum'],
            'CURRENT_AMT_OVERDUE_DURATION_RATIO': ['max', 'sum'],
            'AMT_OVERDUE_DURATION_LEFT_RATIO': ['max', 'mean'],
            'CNT_PROLONGED_MAX_OVERDUE_MUL': ['mean', 'max'],
            'CNT_PROLONGED_DURATION_RATIO': ['mean', 'max'],
            'CURRENT_DEBT_TO_CREDIT_RATIO': ['mean', 'min'],
            'CURRENT_CREDIT_DEBT_DIFF': ['mean', 'min'],
            'AMT_ANNUITY_CREDIT_RATIO': ['mean', 'max', 'min'],
            'CREDIT_ENDDATE_UPDATE_DIFF': ['max', 'min'],
            'STATUS_MEAN': ['mean', 'max'],
            'WEIGHTED_STATUS_MEAN': ['mean', 'max']
        }

        # we saw from EDA that the two most common type of CREDIT ACTIVE were 'Closed' and 'Active'.
        # So we will aggregate them two separately and the remaining categories separately.
        categories_to_aggregate_on = ['Closed', 'Active']
        bureau_merged_aggregated_credit = pd.DataFrame()
        for i, status in enumerate(categories_to_aggregate_on):
            group = bureau_merged[bureau_merged['CREDIT_ACTIVE'] == status].groupby('SK_ID_CURR').agg(
                aggregations_CREDIT_ACTIVE)
            group.columns = ['_'.join(ele).upper() + '_CREDITACTIVE_' + status.upper() for ele in group.columns]

            if i == 0:
                bureau_merged_aggregated_credit = group
            else:
                bureau_merged_aggregated_credit = bureau_merged_aggregated_credit.merge(group, on='SK_ID_CURR',
                                                                                        how='outer')
        # aggregating for remaining categories
        bureau_merged_aggregated_credit_rest = bureau_merged[(bureau_merged['CREDIT_ACTIVE'] != 'Active') &
                                                             (bureau_merged['CREDIT_ACTIVE'] != 'Closed')].groupby(
            'SK_ID_CURR').agg(aggregations_CREDIT_ACTIVE)
        bureau_merged_aggregated_credit_rest.columns = ['_'.join(ele).upper() + 'CREDIT_ACTIVE_REST' for ele in
                                                        bureau_merged_aggregated_credit_rest.columns]

        # merging with other categories
        bureau_merged_aggregated_credit = bureau_merged_aggregated_credit.merge(bureau_merged_aggregated_credit_rest,
                                                                                on='SK_ID_CURR', how='outer')

        # Encoding the categorical columns in one-hot form
        currency_ohe = pd.get_dummies(bureau_merged['CREDIT_CURRENCY'], prefix='CURRENCY')
        credit_active_ohe = pd.get_dummies(bureau_merged['CREDIT_ACTIVE'], prefix='CREDIT_ACTIVE')
        credit_type_ohe = pd.get_dummies(bureau_merged['CREDIT_TYPE'], prefix='CREDIT_TYPE')

        # merging the one-hot encoded columns
        bureau_merged = pd.concat([bureau_merged.drop(['CREDIT_CURRENCY', 'CREDIT_ACTIVE', 'CREDIT_TYPE'], axis=1),
                                   currency_ohe, credit_active_ohe, credit_type_ohe], axis=1)

        # aggregating the bureau_merged over all the columns
        bureau_merged_aggregated = bureau_merged.drop('SK_ID_BUREAU', axis=1).groupby('SK_ID_CURR').agg('mean')
        bureau_merged_aggregated.columns = [ele + '_MEAN_OVERALL' for ele in bureau_merged_aggregated.columns]
        # merging it with aggregates over categories
        bureau_merged_aggregated = bureau_merged_aggregated.merge(bureau_merged_aggregated_credit, on='SK_ID_CURR',
                                                                  how='outer')

        if self.verbose:
            print('Done preprocessing bureau and bureau_balance.')
            print(f"\nInitial Size of bureau: {bureau.shape}")
            print(
                f'Size of bureau and bureau_balance after Merging, Pre-Processing, Feature Engineering and Aggregation: {bureau_merged_aggregated.shape}')
            print(f'\nTotal Time Taken = {datetime.now() - self.start}')

        if self.dump_to_pickle:
            if self.verbose:
                print('\nPickling pre-processed bureau and bureau_balance to bureau_merged_aggregated.pkl')
            with open(self.file_directory + 'bureau_merged_aggregated.pkl', 'wb') as f:
                pickle.dump(bureau_merged_aggregated, f)
            if self.verbose:
                print('Done.')
        if self.verbose:
            print('-' * 100)

        return bureau_merged_aggregated

    def main(self):
        """
        Function to be called for complete preprocessing and aggregation of the bureau and bureau_balance tables.

        Inputs:
            self

        Returns:
            Final pre=processed and merged bureau and burea_balance tables
        """

        # preprocessing the bureau table next, by combining it with the aggregated bureau_balance
        bureau_merged_aggregated = self.feat_eng_bureau()

        return bureau_merged_aggregated


# --------------------------------------------------------------------
# -- FEATURE ENGINEERING FOR PREVIOUS_APPLICATION
# --------------------------------------------------------------------
class feat_eng_previous_application:
    """
    Preprocess the previous_application table.
    Contains 5 member functions:
        1. init method
        2. load_dataframe method
        4. preprocessing_feature_engineering method
        5. main method
    """

    def __init__(self, file_directory='', verbose=True, dump_to_pickle=False):
        """
        This function is used to initialize the class members

        Inputs:
            self
            file_directory: Path, str, default = ''
                The path where the file exists. Include a '/' at the end of the path in input
            verbose: bool, default = True
                Whether to enable verbosity or not
            dump_to_pickle: bool, default = False
                Whether to pickle the final preprocessed table or not

        Returns:
            None
        """

        self.verbose = verbose
        self.dump_to_pickle = dump_to_pickle
        self.file_directory = "../P7_scoring_credit/preprocessing/"
        self.path_sav_prev_app_aftproc = '../P7_scoring_credit/preprocessing/previous_application_aftproc.pkl'

    def load_dataframe(self):
        """
        Function to load the previous_application DataFrame.

        Inputs:
            self

        Returns:
            None
        """

        if self.verbose:
            self.start = datetime.now()
            print('########################################################')
            print('#        Pre-processing previous_application        #')
            print('########################################################')
            print("\nLoading the DataFrame, previous_application, into memory...")

        # loading the DataFrame into memory
        with open(self.path_sav_prev_app_aftproc, 'rb') as f:
            self.previous_application = pickle.load(f)
        self.initial_shape = self.previous_application.shape

        if self.verbose:
            print("Loaded previous_application.csv")
            print(f"Time Taken to load = {datetime.now() - self.start}")

    def preprocessing_feature_engineering(self):
        """
        Function to do preprocessing such as categorical encoding and feature engineering.

        Inputs:
            self

        Returns:
            None
        """

        if self.verbose:
            start = datetime.now()
            print("\nPerforming Preprocessing and Feature Engineering...")

        # sorting the applications from oldest to most recent previous loans for each user
        self.previous_application = self.previous_application.sort_values(by=['SK_ID_CURR', 'DAYS_FIRST_DUE'])

        # label encoding the categorical variables
        name_contract_dict = {'Approved': 0, 'Refused': 3, 'Canceled': 2, 'Unused offer': 1}
        self.previous_application['NAME_CONTRACT_STATUS'] = self.previous_application['NAME_CONTRACT_STATUS'].map(
            name_contract_dict)
        yield_group_dict = {'XNA': 0, 'low_action': 1, 'low_normal': 2, 'middle': 3, 'high': 4}
        self.previous_application['NAME_YIELD_GROUP'] = self.previous_application['NAME_YIELD_GROUP'].map(
            yield_group_dict)
        appl_per_contract_last_dict = {'Y': 1, 'N': 0}
        self.previous_application['FLAG_LAST_APPL_PER_CONTRACT'] = self.previous_application[
            'FLAG_LAST_APPL_PER_CONTRACT'].map(appl_per_contract_last_dict)
        remaining_categorical_columns = self.previous_application.dtypes[
            self.previous_application.dtypes == 'object'].index.tolist()
        for col in remaining_categorical_columns:
            encoding_dict = dict([(j, i) for i, j in enumerate(self.previous_application[col].unique(), 1)])
            self.previous_application[col] = self.previous_application[col].map(encoding_dict)

            # engineering some features on domain knowledge
        self.previous_application['MISSING_VALUES_TOTAL_PREV'] = self.previous_application.isna().sum(axis=1)
        self.previous_application['AMT_DECLINED'] = self.previous_application['AMT_APPLICATION'] - \
                                                    self.previous_application['AMT_CREDIT']
        self.previous_application['AMT_CREDIT_GOODS_RATIO'] = self.previous_application['AMT_CREDIT'] / (
                self.previous_application['AMT_GOODS_PRICE'] + 0.00001)
        self.previous_application['AMT_CREDIT_GOODS_DIFF'] = self.previous_application['AMT_CREDIT'] - \
                                                             self.previous_application['AMT_GOODS_PRICE']
        self.previous_application['AMT_CREDIT_APPLICATION_RATIO'] = self.previous_application['AMT_APPLICATION'] / (
                self.previous_application['AMT_CREDIT'] + 0.00001)
        self.previous_application['CREDIT_DOWNPAYMENT_RATIO'] = self.previous_application['AMT_DOWN_PAYMENT'] / (
                self.previous_application['AMT_CREDIT'] + 0.00001)
        self.previous_application['GOOD_DOWNPAYMET_RATIO'] = self.previous_application['AMT_DOWN_PAYMENT'] / (
                self.previous_application['AMT_GOODS_PRICE'] + 0.00001)
        self.previous_application['INTEREST_DOWNPAYMENT'] = self.previous_application['RATE_DOWN_PAYMENT'] * \
                                                            self.previous_application['AMT_DOWN_PAYMENT']
        self.previous_application['INTEREST_CREDIT'] = self.previous_application['AMT_CREDIT'] * \
                                                       self.previous_application['RATE_INTEREST_PRIMARY']
        self.previous_application['INTEREST_CREDIT_PRIVILEGED'] = self.previous_application['AMT_CREDIT'] * \
                                                                  self.previous_application['RATE_INTEREST_PRIVILEGED']
        self.previous_application['APPLICATION_AMT_TO_DECISION_RATIO'] = self.previous_application[
                                                                             'AMT_APPLICATION'] / (
                                                                                 self.previous_application[
                                                                                     'DAYS_DECISION'] + 0.00001) * -1
        self.previous_application['AMT_APPLICATION_TO_SELLERPLACE_AREA'] = self.previous_application[
                                                                               'AMT_APPLICATION'] / (
                                                                                   self.previous_application[
                                                                                       'SELLERPLACE_AREA'] + 0.00001)
        self.previous_application['ANNUITY'] = self.previous_application['AMT_CREDIT'] / (
                self.previous_application['CNT_PAYMENT'] + 0.00001)
        self.previous_application['ANNUITY_GOODS'] = self.previous_application['AMT_GOODS_PRICE'] / (
                self.previous_application['CNT_PAYMENT'] + 0.00001)
        self.previous_application['DAYS_FIRST_LAST_DUE_DIFF'] = self.previous_application['DAYS_LAST_DUE'] - \
                                                                self.previous_application['DAYS_FIRST_DUE']
        self.previous_application['AMT_CREDIT_HOUR_PROCESS_START'] = self.previous_application['AMT_CREDIT'] * \
                                                                     self.previous_application[
                                                                         'HOUR_APPR_PROCESS_START']
        self.previous_application['AMT_CREDIT_NFLAG_LAST_APPL_DAY'] = self.previous_application['AMT_CREDIT'] * \
                                                                      self.previous_application[
                                                                          'NFLAG_LAST_APPL_IN_DAY']
        self.previous_application['AMT_CREDIT_YIELD_GROUP'] = self.previous_application['AMT_CREDIT'] * \
                                                              self.previous_application['NAME_YIELD_GROUP']
        # https://www.kaggle.com/c/home-credit-default-risk/discussion/64598
        self.previous_application['AMT_INTEREST'] = self.previous_application['CNT_PAYMENT'] * \
                                                    self.previous_application[
                                                        'AMT_ANNUITY'] - self.previous_application['AMT_CREDIT']
        self.previous_application['INTEREST_SHARE'] = self.previous_application['AMT_INTEREST'] / (
                self.previous_application[
                    'AMT_CREDIT'] + 0.00001)
        self.previous_application['INTEREST_RATE'] = 2 * 12 * self.previous_application['AMT_INTEREST'] / (
                self.previous_application[
                    'AMT_CREDIT'] * (self.previous_application['CNT_PAYMENT'] + 1))

        if self.verbose:
            print("Done.")
            print(f"Time taken = {datetime.now() - start}")

    def aggregations(self):
        """
        Function to aggregate the previous applications over SK_ID_CURR

        Inputs:
            self

        Returns:
            aggregated previous_applications
        """

        if self.verbose:
            print("\nAggregating previous applications over SK_ID_CURR...")

        aggregations_for_previous_application = {
            'MISSING_VALUES_TOTAL_PREV': ['sum'],
            'NAME_CONTRACT_TYPE': ['mean', 'last'],
            'AMT_ANNUITY': ['mean', 'sum', 'max'],
            'AMT_APPLICATION': ['mean', 'max', 'sum'],
            'AMT_CREDIT': ['mean', 'max', 'sum'],
            'AMT_DOWN_PAYMENT': ['mean', 'max', 'sum'],
            'AMT_GOODS_PRICE': ['mean', 'max', 'sum'],
            'WEEKDAY_APPR_PROCESS_START': ['mean', 'max', 'min'],
            'HOUR_APPR_PROCESS_START': ['mean', 'max', 'min'],
            'FLAG_LAST_APPL_PER_CONTRACT': ['mean', 'sum'],
            'NFLAG_LAST_APPL_IN_DAY': ['mean', 'sum'],
            'RATE_DOWN_PAYMENT': ['mean', 'max'],
            'RATE_INTEREST_PRIMARY': ['mean', 'max'],
            'RATE_INTEREST_PRIVILEGED': ['mean', 'max'],
            'NAME_CASH_LOAN_PURPOSE': ['mean', 'last'],
            'NAME_CONTRACT_STATUS': ['mean', 'max', 'last'],
            'DAYS_DECISION': ['mean', 'max', 'min'],
            'NAME_PAYMENT_TYPE': ['mean', 'last'],
            'CODE_REJECT_REASON': ['mean', 'last'],
            'NAME_TYPE_SUITE': ['mean', 'last'],
            'NAME_CLIENT_TYPE': ['mean', 'last'],
            'NAME_GOODS_CATEGORY': ['mean', 'last'],
            'NAME_PORTFOLIO': ['mean', 'last'],
            'NAME_PRODUCT_TYPE': ['mean', 'last'],
            'CHANNEL_TYPE': ['mean', 'last'],
            'SELLERPLACE_AREA': ['mean', 'max', 'min'],
            'NAME_SELLER_INDUSTRY': ['mean', 'last'],
            'CNT_PAYMENT': ['sum', 'mean', 'max'],
            'NAME_YIELD_GROUP': ['mean', 'last'],
            'PRODUCT_COMBINATION': ['mean', 'last'],
            'DAYS_FIRST_DRAWING': ['mean', 'max'],
            'DAYS_FIRST_DUE': ['mean', 'max'],
            'DAYS_LAST_DUE_1ST_VERSION': ['mean'],
            'DAYS_LAST_DUE': ['mean'],
            'DAYS_TERMINATION': ['mean', 'max'],
            'NFLAG_INSURED_ON_APPROVAL': ['sum'],
            'AMT_DECLINED': ['mean', 'max', 'sum'],
            'AMT_CREDIT_GOODS_RATIO': ['mean', 'max', 'min'],
            'AMT_CREDIT_GOODS_DIFF': ['sum', 'mean', 'max', 'min'],
            'AMT_CREDIT_APPLICATION_RATIO': ['mean', 'min'],
            'CREDIT_DOWNPAYMENT_RATIO': ['mean', 'max'],
            'GOOD_DOWNPAYMET_RATIO': ['mean', 'max'],
            'INTEREST_DOWNPAYMENT': ['mean', 'sum', 'max'],
            'INTEREST_CREDIT': ['mean', 'sum', 'max'],
            'INTEREST_CREDIT_PRIVILEGED': ['mean', 'sum', 'max'],
            'APPLICATION_AMT_TO_DECISION_RATIO': ['mean', 'min'],
            'AMT_APPLICATION_TO_SELLERPLACE_AREA': ['mean', 'max'],
            'ANNUITY': ['mean', 'sum', 'max'],
            'ANNUITY_GOODS': ['mean', 'sum', 'max'],
            'DAYS_FIRST_LAST_DUE_DIFF': ['mean', 'max'],
            'AMT_CREDIT_HOUR_PROCESS_START': ['mean', 'sum'],
            'AMT_CREDIT_NFLAG_LAST_APPL_DAY': ['mean', 'max'],
            'AMT_CREDIT_YIELD_GROUP': ['mean', 'sum', 'min'],
            'AMT_INTEREST': ['mean', 'sum', 'max', 'min'],
            'INTEREST_SHARE': ['mean', 'max', 'min'],
            'INTEREST_RATE': ['mean', 'max', 'min']
        }

        # grouping the previous applications over SK_ID_CURR while only taking the latest 5 applications
        group_last_3 = self.previous_application.groupby('SK_ID_CURR').tail(5).groupby('SK_ID_CURR').agg(
            aggregations_for_previous_application)
        group_last_3.columns = ['_'.join(ele).upper() + '_LAST_5' for ele in group_last_3.columns]
        # grouping the previous applications over SK_ID_CURR while only taking the first 2 applications
        group_first_3 = self.previous_application.groupby('SK_ID_CURR').head(2).groupby('SK_ID_CURR').agg(
            aggregations_for_previous_application)
        group_first_3.columns = ['_'.join(ele).upper() + '_FIRST_2' for ele in group_first_3.columns]
        # grouping the previous applications over SK_ID_CURR while taking all the applications into consideration
        group_all = self.previous_application.groupby('SK_ID_CURR').agg(aggregations_for_previous_application)
        group_all.columns = ['_'.join(ele).upper() + '_ALL' for ele in group_all.columns]

        # merging all the applications
        previous_application_aggregated = group_last_3.merge(group_first_3, on='SK_ID_CURR', how='outer')
        previous_application_aggregated = previous_application_aggregated.merge(group_all, on='SK_ID_CURR', how='outer')

        return previous_application_aggregated

    def main(self):
        """
        Function to be called for complete preprocessing and aggregation of previous_application table.

        Inputs:
            self

        Returns:
            Final pre=processed and aggregated previous_application table.
        """

        # loading the DataFrame
        self.load_dataframe()

        # preprocessing the categorical features and creating new features
        self.preprocessing_feature_engineering()

        # aggregating data over SK_ID_CURR
        previous_application_aggregated = self.aggregations()

        if self.verbose:
            print('Done aggregations.')
            print(f"\nInitial Size of previous_application: {self.initial_shape}")
            print(
                f'Size of previous_application after Pre-Processing, Feature Engineering and Aggregation: {previous_application_aggregated.shape}')
            print(f'\nTotal Time Taken = {datetime.now() - self.start}')

        if self.dump_to_pickle:
            if self.verbose:
                print('\nPickling pre-processed previous_application to previous_application_afteng.pkl')
            with open(self.file_directory + 'previous_application_afteng.pkl', 'wb') as f:
                pickle.dump(previous_application_aggregated, f)
            if self.verbose:
                print('Done.')
        if self.verbose:
            print('-' * 100)

        return previous_application_aggregated


# --------------------------------------------------------------------
# -- FEATURE ENGINEERING FOR POS_CASH_balance
# --------------------------------------------------------------------
class feat_eng_POS_CASH_balance:
    """
    Preprocess the POS_CASH_balance table.
    Contains 6 member functions:
        1. init method
        2. load_dataframe method
        3. data_preprocessing_and_feature_engineering method
        4. aggregations_sk_id_prev method
        5. aggregations_sk_id_curr method
        6. main method
    """

    def __init__(self, file_directory='', verbose=True, dump_to_pickle=False):
        """
        This function is used to initialize the class members

        Inputs:
            self
            file_directory: Path, str, default = ''
                The path where the file exists. Include a '/' at the end of the path in input
            verbose: bool, default = True
                Whether to enable verbosity or not
            dump_to_pickle: bool, default = False
                Whether to pickle the final preprocessed table or not

        Returns:
            None
        """

        self.verbose = verbose
        self.dump_to_pickle = dump_to_pickle
        self.file_directory = "../P7_scoring_credit/preprocessing/"
        self.path_sav_POS_CASH_aftproc = '../P7_scoring_credit/preprocessing/POS_CASH_aftproc.pkl'

    def load_dataframe(self):
        """
        Function to load the POS_CASH_balance DataFrame.

        Inputs:
            self

        Returns:
            None
        """

        if self.verbose:
            self.start = datetime.now()
            print('#########################################################')
            print('#          Pre-processing POS_CASH_balance          #')
            print('#########################################################')
            print("\nLoading the DataFrame, POS_CASH_balance, into memory...")

        with open(self.path_sav_POS_CASH_aftproc, 'rb') as f:
            self.pos_cash = pickle.load(f)
        self.initial_size = self.pos_cash.shape

        if self.verbose:
            print("Loaded POS_CASH_balance.csv")
            print(f"Time Taken to load = {datetime.now() - self.start}")

    def data_preprocessing_and_feature_engineering(self):
        """
        Function to preprocess the table and create new features.

        Inputs:
            self

        Returns:
            None
        """

        if self.verbose:
            start = datetime.now()
            print("\nStarting Data Cleaning and Feature Engineering...")

        # making the MONTHS_BALANCE Positive
        self.pos_cash['MONTHS_BALANCE'] = np.abs(self.pos_cash['MONTHS_BALANCE'])
        # sorting the DataFrame according to the month of status from oldest to latest, for rolling computations
        self.pos_cash = self.pos_cash.sort_values(by=['SK_ID_PREV', 'MONTHS_BALANCE'], ascending=False)

        # computing Exponential Moving Average for some features based on MONTHS_BALANCE
        columns_for_ema = ['CNT_INSTALMENT', 'CNT_INSTALMENT_FUTURE']
        exp_columns = ['EXP_' + ele for ele in columns_for_ema]
        self.pos_cash[exp_columns] = self.pos_cash.groupby('SK_ID_PREV')[columns_for_ema].transform(
            lambda x: x.ewm(alpha=0.6).mean())

        # creating new features based on Domain Knowledge
        self.pos_cash['SK_DPD_RATIO'] = self.pos_cash['SK_DPD'] / (self.pos_cash['SK_DPD_DEF'] + 0.00001)
        self.pos_cash['TOTAL_TERM'] = self.pos_cash['CNT_INSTALMENT'] + self.pos_cash['CNT_INSTALMENT_FUTURE']
        self.pos_cash['EXP_POS_TOTAL_TERM'] = self.pos_cash['EXP_CNT_INSTALMENT'] + self.pos_cash[
            'EXP_CNT_INSTALMENT_FUTURE']

        if self.verbose:
            print("Done.")
            print(f"Time Taken = {datetime.now() - start}")

    def aggregations_sk_id_prev(self):
        """
        Function to aggregated the POS_CASH_balance rows over SK_ID_PREV

        Inputs:
            self

        Returns:
            Aggregated POS_CASH_balance table over SK_ID_PREV
        """

        if self.verbose:
            start = datetime.now()
            print("\nAggregations over SK_ID_PREV...")

        # aggregating over SK_ID_PREV
        overall_aggregations = {
            'SK_ID_CURR': ['first'],
            'MONTHS_BALANCE': ['max'],
            'CNT_INSTALMENT': ['mean', 'max', 'min'],
            'CNT_INSTALMENT_FUTURE': ['mean', 'max', 'min'],
            'SK_DPD': ['max', 'sum'],
            'SK_DPD_DEF': ['max', 'sum'],
            'EXP_CNT_INSTALMENT': ['last'],
            'EXP_CNT_INSTALMENT_FUTURE': ['last'],
            'SK_DPD_RATIO': ['mean', 'max'],
            'TOTAL_TERM': ['mean', 'max', 'last'],
            'EXP_POS_TOTAL_TERM': ['mean']
        }
        aggregations_for_year = {
            'CNT_INSTALMENT': ['mean', 'max', 'min'],
            'CNT_INSTALMENT_FUTURE': ['mean', 'max', 'min'],
            'SK_DPD': ['max', 'sum'],
            'SK_DPD_DEF': ['max', 'sum'],
            'EXP_CNT_INSTALMENT': ['last'],
            'EXP_CNT_INSTALMENT_FUTURE': ['last'],
            'SK_DPD_RATIO': ['mean', 'max'],
            'TOTAL_TERM': ['mean', 'max'],
            'EXP_POS_TOTAL_TERM': ['last']
        }
        aggregations_for_categories = {
            'CNT_INSTALMENT': ['mean', 'max', 'min'],
            'CNT_INSTALMENT_FUTURE': ['mean', 'max', 'min'],
            'SK_DPD': ['max', 'sum'],
            'SK_DPD_DEF': ['max', 'sum'],
            'EXP_CNT_INSTALMENT': ['last'],
            'EXP_CNT_INSTALMENT_FUTURE': ['last'],
            'SK_DPD_RATIO': ['mean', 'max'],
            'TOTAL_TERM': ['mean', 'max'],
            'EXP_POS_TOTAL_TERM': ['last']
        }
        # performing overall aggregations over SK_ID_PREV
        pos_cash_aggregated_overall = self.pos_cash.groupby('SK_ID_PREV').agg(overall_aggregations)
        pos_cash_aggregated_overall.columns = ['_'.join(ele).upper() for ele in pos_cash_aggregated_overall.columns]
        pos_cash_aggregated_overall.rename(columns={'SK_ID_CURR_FIRST': 'SK_ID_CURR'}, inplace=True)

        # yearwise aggregations
        self.pos_cash['YEAR_BALANCE'] = self.pos_cash['MONTHS_BALANCE'] // 12
        # aggregating over SK_ID_PREV for each last 2 years
        pos_cash_aggregated_year = pd.DataFrame()
        for year in range(2):
            group = self.pos_cash[self.pos_cash['YEAR_BALANCE'] == year].groupby('SK_ID_PREV').agg(
                aggregations_for_year)
            group.columns = ['_'.join(ele).upper() + '_YEAR_' + str(year) for ele in group.columns]
            if year == 0:
                pos_cash_aggregated_year = group
            else:
                pos_cash_aggregated_year = pos_cash_aggregated_year.merge(group, on='SK_ID_PREV', how='outer')

        # aggregating over SK_ID_PREV for rest of the years
        pos_cash_aggregated_rest_years = self.pos_cash[self.pos_cash['YEAR_BALANCE'] >= 2].groupby('SK_ID_PREV').agg(
            aggregations_for_year)
        pos_cash_aggregated_rest_years.columns = ['_'.join(ele).upper() + '_YEAR_REST' for ele in
                                                  pos_cash_aggregated_rest_years.columns]
        # merging all the years aggregations
        pos_cash_aggregated_year = pos_cash_aggregated_year.merge(pos_cash_aggregated_rest_years, on='SK_ID_PREV',
                                                                  how='outer')
        self.pos_cash = self.pos_cash.drop(['YEAR_BALANCE'], axis=1)

        # aggregating over SK_ID_PREV for each of NAME_CONTRACT_STATUS categories
        contract_type_categories = ['Active', 'Completed']
        pos_cash_aggregated_contract = pd.DataFrame()
        for i, contract_type in enumerate(contract_type_categories):
            group = self.pos_cash[self.pos_cash['NAME_CONTRACT_STATUS'] == contract_type].groupby('SK_ID_PREV').agg(
                aggregations_for_categories)
            group.columns = ['_'.join(ele).upper() + '_' + contract_type.upper() for ele in group.columns]
            if i == 0:
                pos_cash_aggregated_contract = group
            else:
                pos_cash_aggregated_contract = pos_cash_aggregated_contract.merge(group, on='SK_ID_PREV', how='outer')

        pos_cash_aggregated_rest_contract = self.pos_cash[(self.pos_cash['NAME_CONTRACT_STATUS'] != 'Active') &
                                                          (self.pos_cash[
                                                               'NAME_CONTRACT_STATUS'] != 'Completed')].groupby(
            'SK_ID_PREV').agg(aggregations_for_categories)
        pos_cash_aggregated_rest_contract.columns = ['_'.join(ele).upper() + '_REST' for ele in
                                                     pos_cash_aggregated_rest_contract.columns]
        # merging the categorical aggregations
        pos_cash_aggregated_contract = pos_cash_aggregated_contract.merge(pos_cash_aggregated_rest_contract,
                                                                          on='SK_ID_PREV', how='outer')

        # merging all the aggregations
        pos_cash_aggregated = pos_cash_aggregated_overall.merge(pos_cash_aggregated_year, on='SK_ID_PREV', how='outer')
        pos_cash_aggregated = pos_cash_aggregated.merge(pos_cash_aggregated_contract, on='SK_ID_PREV', how='outer')

        # onehot encoding the categorical feature NAME_CONTRACT_TYPE
        name_contract_dummies = pd.get_dummies(self.pos_cash['NAME_CONTRACT_STATUS'], prefix='CONTRACT')
        contract_names = name_contract_dummies.columns.tolist()
        # concatenating one-hot encoded categories with main table
        self.pos_cash = pd.concat([self.pos_cash, name_contract_dummies], axis=1)
        # aggregating these over SK_ID_PREV as well
        aggregated_cc_contract = self.pos_cash[['SK_ID_PREV'] + contract_names].groupby('SK_ID_PREV').mean()

        # merging with the final aggregations
        pos_cash_aggregated = pos_cash_aggregated.merge(aggregated_cc_contract, on='SK_ID_PREV', how='outer')

        if self.verbose:
            print("Done.")
            print(f"Time Taken = {datetime.now() - start}")

        return pos_cash_aggregated

    def aggregations_sk_id_curr(self, pos_cash_aggregated):
        """
        Function to aggregated the aggregateed POS_CASH_balance table over SK_ID_CURR

        Inputs:
            self
            pos_cash_aggregated: DataFrame
                aggregated pos_cash table over SK_ID_PREV

        Returns:
            pos_cash_balance table aggregated over SK_ID_CURR
        """

        # aggregating over SK_ID_CURR
        columns_to_aggregate = pos_cash_aggregated.columns[1:]
        # defining the aggregations to perform
        aggregations_final = {}
        for col in columns_to_aggregate:
            if 'MEAN' in col:
                aggregates = ['mean', 'sum', 'max']
            else:
                aggregates = ['mean']
            aggregations_final[col] = aggregates
        pos_cash_aggregated_final = pos_cash_aggregated.groupby('SK_ID_CURR').agg(aggregations_final)
        pos_cash_aggregated_final.columns = ['_'.join(ele).upper() for ele in pos_cash_aggregated_final.columns]

        return pos_cash_aggregated_final

    def main(self):
        """
        Function to be called for complete preprocessing and aggregation of POS_CASH_balance table.

        Inputs:
            self

        Returns:
            Final pre=processed and aggregated POS_CASH_balance table.
        """

        # loading the dataframe
        self.load_dataframe()
        # performing the data pre-processing and feature engineering
        self.data_preprocessing_and_feature_engineering()
        # performing aggregations over SK_ID_PREV
        pos_cash_aggregated = self.aggregations_sk_id_prev()

        if self.verbose:
            print("\nAggregation over SK_ID_CURR...")
        # doing aggregations over each SK_ID_CURR
        pos_cash_aggregated_final = self.aggregations_sk_id_curr(pos_cash_aggregated)

        if self.verbose:
            print('\nDone preprocessing POS_CASH_balance.')
            print(f"\nInitial Size of POS_CASH_balance: {self.initial_size}")
            print(
                f'Size of POS_CASH_balance after Pre-Processing, Feature Engineering and Aggregation: {pos_cash_aggregated_final.shape}')
            print(f'\nTotal Time Taken = {datetime.now() - self.start}')

        if self.dump_to_pickle:
            if self.verbose:
                print('\nPickling pre-processed POS_CASH_balance to POS_CASH_balance_afteng.pkl')
            with open(self.file_directory + 'POS_CASH_balance_afteng.pkl', 'wb') as f:
                pickle.dump(pos_cash_aggregated_final, f)
            if self.verbose:
                print('Done.')
        if self.verbose:
            print('-' * 100)

        return pos_cash_aggregated_final


# --------------------------------------------------------------------
# -- FEATURE ENGINEERING FOR INSTALLMENTS_PAYMENTS
# --------------------------------------------------------------------
class feat_eng_installments_payments:
    """
    Preprocess the installments_payments table.
    Contains 6 member functions:
        1. init method
        2. load_dataframe method
        3. data_preprocessing_and_feature_engineering method
        4. aggregations_sk_id_prev method
        5. aggregations_sk_id_curr method
        6. main method
    """

    def __init__(self, file_directory='', verbose=True, dump_to_pickle=False):
        """
        This function is used to initialize the class members

        Inputs:
            self
            file_directory: Path, str, default = ''
                The path where the file exists. Include a '/' at the end of the path in input
            verbose: bool, default = True
                Whether to enable verbosity or not
            dump_to_pickle: bool, default = False
                Whether to pickle the final preprocessed table or not

        Returns:
            None
        """

        self.verbose = verbose
        self.dump_to_pickle = dump_to_pickle
        self.file_directory = "../P7_scoring_credit/preprocessing/"
        self.path_sav_instpay_aftproc = '../P7_scoring_credit/preprocessing/installments_payments_aftproc.pkl'

    def load_dataframe(self):
        """
        Function to load the installments_payments.csv DataFrame.

        Inputs:
            self

        Returns:
            None
        """

        if self.verbose:
            self.start = datetime.now()
            print('#####################################################')
            print('#        Pre-processing installments_payments       #')
            print('#####################################################')
            print("\nLoading the DataFrame installments_payments into memory...")

        with open(self.path_sav_instpay_aftproc, 'rb') as f:
            self.installments_payments = pickle.load(f)
        self.initial_shape = self.installments_payments.shape

        if self.verbose:
            print("Loaded installments_payments.csv")
            print(f"Time Taken to load = {datetime.now() - self.start}")

    def data_preprocessing_and_feature_engineering(self):
        """
        Function for pre-processing and feature engineering

        Inputs:
            self

        Returns:
            None
        """

        if self.verbose:
            start = datetime.now()
            print("\nStarting Data Pre-processing and Feature Engineering...")

        # sorting by SK_ID_PREV and NUM_INSTALMENT_NUMBER
        self.installments_payments = self.installments_payments.sort_values(
            by=['SK_ID_CURR', 'SK_ID_PREV', 'NUM_INSTALMENT_NUMBER'], ascending=True)

        # getting the total NaN values in the table
        self.installments_payments['MISSING_VALS_TOTAL_INSTAL'] = self.installments_payments.isna().sum(axis=1)
        # engineering new features based on some domain based polynomial operations
        self.installments_payments['DAYS_PAYMENT_RATIO'] = self.installments_payments['DAYS_INSTALMENT'] / (
                self.installments_payments['DAYS_ENTRY_PAYMENT'] + 0.00001)
        self.installments_payments['DAYS_PAYMENT_DIFF'] = self.installments_payments['DAYS_INSTALMENT'] - \
                                                          self.installments_payments['DAYS_ENTRY_PAYMENT']
        self.installments_payments['AMT_PAYMENT_RATIO'] = self.installments_payments['AMT_PAYMENT'] / (
                self.installments_payments['AMT_INSTALMENT'] + 0.00001)
        self.installments_payments['AMT_PAYMENT_DIFF'] = self.installments_payments['AMT_INSTALMENT'] - \
                                                         self.installments_payments['AMT_PAYMENT']
        self.installments_payments['EXP_DAYS_PAYMENT_RATIO'] = self.installments_payments[
            'DAYS_PAYMENT_RATIO'].transform(lambda x: x.ewm(alpha=0.5).mean())
        self.installments_payments['EXP_DAYS_PAYMENT_DIFF'] = self.installments_payments['DAYS_PAYMENT_DIFF'].transform(
            lambda x: x.ewm(alpha=0.5).mean())
        self.installments_payments['EXP_AMT_PAYMENT_RATIO'] = self.installments_payments['AMT_PAYMENT_RATIO'].transform(
            lambda x: x.ewm(alpha=0.5).mean())
        self.installments_payments['EXP_AMT_PAYMENT_DIFF'] = self.installments_payments['AMT_PAYMENT_DIFF'].transform(
            lambda x: x.ewm(alpha=0.5).mean())

        if self.verbose:
            print("Done.")
            print(f"Time Taken = {datetime.now() - start}")

    def aggregations_sk_id_prev(self):
        """
        Function for aggregations of installments on previous loans over SK_ID_PREV

        Inputs:
            self

        Returns:
            installments_payments table aggregated over previous loans
        """

        if self.verbose:
            start = datetime.now()
            print("\nPerforming Aggregations over SK_ID_PREV...")

        # aggregating the data over SK_ID_PREV, i.e. for each previous loan
        overall_aggregations = {
            'MISSING_VALS_TOTAL_INSTAL': ['sum'],
            'NUM_INSTALMENT_VERSION': ['mean', 'sum'],
            'NUM_INSTALMENT_NUMBER': ['max'],
            'DAYS_INSTALMENT': ['max', 'min'],
            'DAYS_ENTRY_PAYMENT': ['max', 'min'],
            'AMT_INSTALMENT': ['mean', 'sum', 'max'],
            'AMT_PAYMENT': ['mean', 'sum', 'max'],
            'DAYS_PAYMENT_RATIO': ['mean', 'min', 'max'],
            'DAYS_PAYMENT_DIFF': ['mean', 'min', 'max'],
            'AMT_PAYMENT_RATIO': ['mean', 'min', 'max'],
            'AMT_PAYMENT_DIFF': ['mean', 'min', 'max'],
            'EXP_DAYS_PAYMENT_RATIO': ['last'],
            'EXP_DAYS_PAYMENT_DIFF': ['last'],
            'EXP_AMT_PAYMENT_RATIO': ['last'],
            'EXP_AMT_PAYMENT_DIFF': ['last']
        }
        limited_period_aggregations = {
            'NUM_INSTALMENT_VERSION': ['mean', 'sum'],
            'AMT_INSTALMENT': ['mean', 'sum', 'max'],
            'AMT_PAYMENT': ['mean', 'sum', 'max'],
            'DAYS_PAYMENT_RATIO': ['mean', 'min', 'max'],
            'DAYS_PAYMENT_DIFF': ['mean', 'min', 'max'],
            'AMT_PAYMENT_RATIO': ['mean', 'min', 'max'],
            'AMT_PAYMENT_DIFF': ['mean', 'min', 'max'],
            'EXP_DAYS_PAYMENT_RATIO': ['last'],
            'EXP_DAYS_PAYMENT_DIFF': ['last'],
            'EXP_AMT_PAYMENT_RATIO': ['last'],
            'EXP_AMT_PAYMENT_DIFF': ['last']
        }

        # aggregating installments_payments over SK_ID_PREV for last 1 year installments
        group_last_1_year = self.installments_payments[self.installments_payments['DAYS_INSTALMENT'] > -365].groupby(
            'SK_ID_PREV').agg(limited_period_aggregations)
        group_last_1_year.columns = ['_'.join(ele).upper() + '_LAST_1_YEAR' for ele in group_last_1_year.columns]
        # aggregating installments_payments over SK_ID_PREV for first 5 installments
        group_first_5_instalments = self.installments_payments.groupby('SK_ID_PREV', as_index=False).head(5).groupby(
            'SK_ID_PREV').agg(limited_period_aggregations)
        group_first_5_instalments.columns = ['_'.join(ele).upper() + '_FIRST_5_INSTALLMENTS' for ele in
                                             group_first_5_instalments.columns]
        # overall aggregation of installments_payments over SK_ID_PREV
        group_overall = self.installments_payments.groupby(['SK_ID_PREV', 'SK_ID_CURR'], as_index=False).agg(
            overall_aggregations)
        group_overall.columns = ['_'.join(ele).upper() for ele in group_overall.columns]
        group_overall.rename(columns={'SK_ID_PREV_': 'SK_ID_PREV', 'SK_ID_CURR_': 'SK_ID_CURR'}, inplace=True)

        # merging all of the above aggregations together
        installments_payments_agg_prev = group_overall.merge(group_last_1_year, on='SK_ID_PREV', how='outer')
        installments_payments_agg_prev = installments_payments_agg_prev.merge(group_first_5_instalments,
                                                                              on='SK_ID_PREV', how='outer')

        if self.verbose:
            print("Done.")
            print(f"Time Taken = {datetime.now() - start}")

        return installments_payments_agg_prev

    def aggregations_sk_id_curr(self, installments_payments_agg_prev):
        """
        Function to aggregate the installments payments on previous loans over SK_ID_CURR

        Inputs:
            self
            installments_payments_agg_prev: DataFrame
                installments payments aggregated over SK_ID_PREV

        Returns:
            installments payments aggregated over SK_ID_CURR
        """

        # aggregating over SK_ID_CURR
        main_features_aggregations = {
            'MISSING_VALS_TOTAL_INSTAL_SUM': ['sum'],
            'NUM_INSTALMENT_VERSION_MEAN': ['mean'],
            'NUM_INSTALMENT_VERSION_SUM': ['mean'],
            'NUM_INSTALMENT_NUMBER_MAX': ['mean', 'sum', 'max'],
            'AMT_INSTALMENT_MEAN': ['mean', 'sum', 'max'],
            'AMT_INSTALMENT_SUM': ['mean', 'sum', 'max'],
            'AMT_INSTALMENT_MAX': ['mean'],
            'AMT_PAYMENT_MEAN': ['mean', 'sum', 'max'],
            'AMT_PAYMENT_SUM': ['mean', 'sum', 'max'],
            'AMT_PAYMENT_MAX': ['mean'],
            'DAYS_PAYMENT_RATIO_MEAN': ['mean', 'min', 'max'],
            'DAYS_PAYMENT_RATIO_MIN': ['mean', 'min'],
            'DAYS_PAYMENT_RATIO_MAX': ['mean', 'max'],
            'DAYS_PAYMENT_DIFF_MEAN': ['mean', 'min', 'max'],
            'DAYS_PAYMENT_DIFF_MIN': ['mean', 'min'],
            'DAYS_PAYMENT_DIFF_MAX': ['mean', 'max'],
            'AMT_PAYMENT_RATIO_MEAN': ['mean', 'min', 'max'],
            'AMT_PAYMENT_RATIO_MIN': ['mean', 'min'],
            'AMT_PAYMENT_RATIO_MAX': ['mean', 'max'],
            'AMT_PAYMENT_DIFF_MEAN': ['mean', 'min', 'max'],
            'AMT_PAYMENT_DIFF_MIN': ['mean', 'min'],
            'AMT_PAYMENT_DIFF_MAX': ['mean', 'max'],
            'EXP_DAYS_PAYMENT_RATIO_LAST': ['mean'],
            'EXP_DAYS_PAYMENT_DIFF_LAST': ['mean'],
            'EXP_AMT_PAYMENT_RATIO_LAST': ['mean'],
            'EXP_AMT_PAYMENT_DIFF_LAST': ['mean']
        }

        grouped_main_features = installments_payments_agg_prev.groupby('SK_ID_CURR').agg(main_features_aggregations)
        grouped_main_features.columns = ['_'.join(ele).upper() for ele in grouped_main_features.columns]

        # group remaining ones
        grouped_remaining_features = installments_payments_agg_prev.iloc[:,
                                     [1] + list(range(31, len(installments_payments_agg_prev.columns)))].groupby(
            'SK_ID_CURR').mean()

        installments_payments_aggregated = grouped_main_features.merge(grouped_remaining_features, on='SK_ID_CURR',
                                                                       how='inner')

        return installments_payments_aggregated

    def main(self):
        """
        Function to be called for complete preprocessing and aggregation of installments_payments table.

        Inputs:
            self

        Returns:
            Final pre=processed and aggregated installments_payments table.
        """

        # loading the dataframe
        self.load_dataframe()
        # doing pre-processing and feature engineering
        self.data_preprocessing_and_feature_engineering()
        # First aggregating the data for each SK_ID_PREV
        installments_payments_agg_prev = self.aggregations_sk_id_prev()

        if self.verbose:
            print("\nAggregations over SK_ID_CURR...")
        # aggregating the previous loans for each SK_ID_CURR
        installments_payments_aggregated = self.aggregations_sk_id_curr(installments_payments_agg_prev)

        if self.verbose:
            print('\nDone preprocessing installments_payments.')
            print(f"\nInitial Size of installments_payments: {self.initial_shape}")
            print(
                f'Size of installments_payments after Pre-Processing, Feature Engineering and Aggregation: {installments_payments_aggregated.shape}')
            print(f'\nTotal Time Taken = {datetime.now() - self.start}')

        if self.dump_to_pickle:
            if self.verbose:
                print('\nPickling pre-processed installments_payments to install_pay_afteng.pkl')
            with open(self.file_directory + 'install_pay_afteng.pkl', 'wb') as f:
                pickle.dump(installments_payments_aggregated, f)
            if self.verbose:
                print('Done.')
        if self.verbose:
            print('-' * 100)

        return installments_payments_aggregated

    # --------------------------------------------------------------------


# -- FEATURE ENGINEERING FOR CREDIT_CARD_BALANCE
# --------------------------------------------------------------------


class feat_eng_cc_balance:
    """
    Preprocess the credit_card_balance table.
    Contains 5 member functions:
        1. init method
        2. load_dataframe method
        3. data_preprocessing_and_feature_engineering method
        4. aggregations method
        5. main method
    """

    def __init__(self, file_directory='', verbose=True, dump_to_pickle=False):
        """
        This function is used to initialize the class members

        Inputs:
            self
            file_directory: Path, str, default = ''
                The path where the file exists. Include a '/' at the end of the path in input
            verbose: bool, default = True
                Whether to enable verbosity or not
            dump_to_pickle: bool, default = False
                Whether to pickle the final preprocessed table or not

        Returns:
            None
        """

        self.verbose = verbose
        self.dump_to_pickle = dump_to_pickle
        self.file_directory = "../P7_scoring_credit/preprocessing/"
        self.path_sav_cc_balance_aftproc = '../P7_scoring_credit/preprocessing/cc_balance_aftproc.pkl'

    def load_dataframe(self):
        """
        Function to load the credit_card_balance.csv DataFrame.

        Inputs:
            self

        Returns:
            None
        """

        if self.verbose:
            self.start = datetime.now()
            print('#########################################################')
            print('#        Pre-processing credit_card_balance.csv         #')
            print('#########################################################')
            print("\nLoading the DataFrame, credit_card_balance.csv, into memory...")

        with open(self.path_sav_cc_balance_aftproc, 'rb') as f:
            self.cc_balance = pickle.load(f)
        self.initial_size = self.cc_balance.shape

        if self.verbose:
            print("Loaded credit_card_balance.csv")
            print(f"Time Taken to load = {datetime.now() - self.start}")

    def data_preprocessing_and_feature_engineering(self):
        """
        Function to preprocess the table, by removing erroneous points, and then creating new domain based features.

        Inputs:
            self

        Returns:
            None
        """

        if self.verbose:
            start = datetime.now()
            print("\nStarting Preprocessing and Feature Engineering...")

        # there is one abruptly large value for AMT_PAYMENT_CURRENT
        self.cc_balance['AMT_PAYMENT_CURRENT'][self.cc_balance['AMT_PAYMENT_CURRENT'] > 4000000] = np.nan
        # calculating the total missing values for each previous credit card
        self.cc_balance['MISSING_VALS_TOTAL_CC'] = self.cc_balance.isna().sum(axis=1)
        # making the MONTHS_BALANCE Positive
        self.cc_balance['MONTHS_BALANCE'] = np.abs(self.cc_balance['MONTHS_BALANCE'])
        # sorting the DataFrame according to the month of status from oldest to latest, for rolling computations
        self.cc_balance = self.cc_balance.sort_values(by=['SK_ID_PREV', 'MONTHS_BALANCE'], ascending=[1, 0])

        # Creating new features
        self.cc_balance['AMT_DRAWING_SUM'] = self.cc_balance['AMT_DRAWINGS_ATM_CURRENT'] + self.cc_balance[
            'AMT_DRAWINGS_CURRENT'] + self.cc_balance[
                                                 'AMT_DRAWINGS_OTHER_CURRENT'] + self.cc_balance[
                                                 'AMT_DRAWINGS_POS_CURRENT']
        self.cc_balance['BALANCE_LIMIT_RATIO'] = self.cc_balance['AMT_BALANCE'] / (
                self.cc_balance['AMT_CREDIT_LIMIT_ACTUAL'] + 0.00001)
        self.cc_balance['CNT_DRAWING_SUM'] = self.cc_balance['CNT_DRAWINGS_ATM_CURRENT'] + self.cc_balance[
            'CNT_DRAWINGS_CURRENT'] + self.cc_balance[
                                                 'CNT_DRAWINGS_OTHER_CURRENT'] + self.cc_balance[
                                                 'CNT_DRAWINGS_POS_CURRENT'] + self.cc_balance[
                                                 'CNT_INSTALMENT_MATURE_CUM']
        self.cc_balance['MIN_PAYMENT_RATIO'] = self.cc_balance['AMT_PAYMENT_CURRENT'] / (
                self.cc_balance['AMT_INST_MIN_REGULARITY'] + 0.0001)
        self.cc_balance['PAYMENT_MIN_DIFF'] = self.cc_balance['AMT_PAYMENT_CURRENT'] - self.cc_balance[
            'AMT_INST_MIN_REGULARITY']
        self.cc_balance['MIN_PAYMENT_TOTAL_RATIO'] = self.cc_balance['AMT_PAYMENT_TOTAL_CURRENT'] / (
                self.cc_balance['AMT_INST_MIN_REGULARITY'] + 0.00001)
        self.cc_balance['PAYMENT_MIN_DIFF'] = self.cc_balance['AMT_PAYMENT_TOTAL_CURRENT'] - self.cc_balance[
            'AMT_INST_MIN_REGULARITY']
        self.cc_balance['AMT_INTEREST_RECEIVABLE'] = self.cc_balance['AMT_TOTAL_RECEIVABLE'] - self.cc_balance[
            'AMT_RECEIVABLE_PRINCIPAL']
        self.cc_balance['SK_DPD_RATIO'] = self.cc_balance['SK_DPD'] / (self.cc_balance['SK_DPD_DEF'] + 0.00001)

        # calculating the rolling Exponential Weighted Moving Average over months for certain features
        rolling_columns = [
            'AMT_BALANCE',
            'AMT_CREDIT_LIMIT_ACTUAL',
            'AMT_RECEIVABLE_PRINCIPAL',
            'AMT_RECIVABLE',
            'AMT_TOTAL_RECEIVABLE',
            'AMT_DRAWING_SUM',
            'BALANCE_LIMIT_RATIO',
            'CNT_DRAWING_SUM',
            'MIN_PAYMENT_RATIO',
            'PAYMENT_MIN_DIFF',
            'MIN_PAYMENT_TOTAL_RATIO',
            'AMT_INTEREST_RECEIVABLE',
            'SK_DPD_RATIO']
        exp_weighted_columns = ['EXP_' + ele for ele in rolling_columns]
        self.cc_balance[exp_weighted_columns] = self.cc_balance.groupby(['SK_ID_CURR', 'SK_ID_PREV'])[
            rolling_columns].transform(lambda x: x.ewm(alpha=0.7).mean())

        if self.verbose:
            print("Done.")
            print(f"Time Taken = {datetime.now() - start}")

    def aggregations(self):
        """
        Function to perform aggregations of rows of credit_card_balance table, first over SK_ID_PREV,
        and then over SK_ID_CURR

        Inputs:
            self

        Returns:
            aggregated credit_card_balance table.
        """

        if self.verbose:
            print("\nAggregating the DataFrame, first over SK_ID_PREv, then over SK_ID_CURR")

        # performing aggregations over SK_ID_PREV
        overall_aggregations = {
            'SK_ID_CURR': ['first'],
            'MONTHS_BALANCE': ['max'],
            'AMT_BALANCE': ['sum', 'mean', 'max'],
            'AMT_CREDIT_LIMIT_ACTUAL': ['sum', 'mean', 'max'],
            'AMT_DRAWINGS_ATM_CURRENT': ['sum', 'max'],
            'AMT_DRAWINGS_CURRENT': ['sum', 'max'],
            'AMT_DRAWINGS_OTHER_CURRENT': ['sum', 'max'],
            'AMT_DRAWINGS_POS_CURRENT': ['sum', 'max'],
            'AMT_INST_MIN_REGULARITY': ['mean', 'min', 'max'],
            'AMT_PAYMENT_CURRENT': ['mean', 'min', 'max'],
            'AMT_PAYMENT_TOTAL_CURRENT': ['mean', 'min', 'max'],
            'AMT_RECEIVABLE_PRINCIPAL': ['sum', 'mean', 'max'],
            'AMT_RECIVABLE': ['sum', 'mean', 'max'],
            'AMT_TOTAL_RECEIVABLE': ['sum', 'mean', 'max'],
            'CNT_DRAWINGS_ATM_CURRENT': ['sum', 'max'],
            'CNT_DRAWINGS_CURRENT': ['sum', 'max'],
            'CNT_DRAWINGS_OTHER_CURRENT': ['sum', 'max'],
            'CNT_DRAWINGS_POS_CURRENT': ['sum', 'max'],
            'CNT_INSTALMENT_MATURE_CUM': ['sum', 'max', 'min'],
            'SK_DPD': ['sum', 'max'],
            'SK_DPD_DEF': ['sum', 'max'],

            'AMT_DRAWING_SUM': ['sum', 'max'],
            'BALANCE_LIMIT_RATIO': ['mean', 'max', 'min'],
            'CNT_DRAWING_SUM': ['sum', 'max'],
            'MIN_PAYMENT_RATIO': ['min', 'mean'],
            'PAYMENT_MIN_DIFF': ['min', 'mean'],
            'MIN_PAYMENT_TOTAL_RATIO': ['min', 'mean'],
            'AMT_INTEREST_RECEIVABLE': ['min', 'mean'],
            'SK_DPD_RATIO': ['max', 'mean'],

            'EXP_AMT_BALANCE': ['last'],
            'EXP_AMT_CREDIT_LIMIT_ACTUAL': ['last'],
            'EXP_AMT_RECEIVABLE_PRINCIPAL': ['last'],
            'EXP_AMT_RECIVABLE': ['last'],
            'EXP_AMT_TOTAL_RECEIVABLE': ['last'],
            'EXP_AMT_DRAWING_SUM': ['last'],
            'EXP_BALANCE_LIMIT_RATIO': ['last'],
            'EXP_CNT_DRAWING_SUM': ['last'],
            'EXP_MIN_PAYMENT_RATIO': ['last'],
            'EXP_PAYMENT_MIN_DIFF': ['last'],
            'EXP_MIN_PAYMENT_TOTAL_RATIO': ['last'],
            'EXP_AMT_INTEREST_RECEIVABLE': ['last'],
            'EXP_SK_DPD_RATIO': ['last'],
            'MISSING_VALS_TOTAL_CC': ['sum']
        }
        aggregations_for_categories = {
            'SK_DPD': ['sum', 'max'],
            'SK_DPD_DEF': ['sum', 'max'],
            'BALANCE_LIMIT_RATIO': ['mean', 'max', 'min'],
            'CNT_DRAWING_SUM': ['sum', 'max'],
            'MIN_PAYMENT_RATIO': ['min', 'mean'],
            'PAYMENT_MIN_DIFF': ['min', 'mean'],
            'MIN_PAYMENT_TOTAL_RATIO': ['min', 'mean'],
            'AMT_INTEREST_RECEIVABLE': ['min', 'mean'],
            'SK_DPD_RATIO': ['max', 'mean'],
            'EXP_AMT_DRAWING_SUM': ['last'],
            'EXP_BALANCE_LIMIT_RATIO': ['last'],
            'EXP_CNT_DRAWING_SUM': ['last'],
            'EXP_MIN_PAYMENT_RATIO': ['last'],
            'EXP_PAYMENT_MIN_DIFF': ['last'],
            'EXP_MIN_PAYMENT_TOTAL_RATIO': ['last'],
            'EXP_AMT_INTEREST_RECEIVABLE': ['last'],
            'EXP_SK_DPD_RATIO': ['last']
        }
        aggregations_for_year = {
            'SK_DPD': ['sum', 'max'],
            'SK_DPD_DEF': ['sum', 'max'],
            'BALANCE_LIMIT_RATIO': ['mean', 'max', 'min'],
            'CNT_DRAWING_SUM': ['sum', 'max'],
            'MIN_PAYMENT_RATIO': ['min', 'mean'],
            'PAYMENT_MIN_DIFF': ['min', 'mean'],
            'MIN_PAYMENT_TOTAL_RATIO': ['min', 'mean'],
            'AMT_INTEREST_RECEIVABLE': ['min', 'mean'],
            'SK_DPD_RATIO': ['max', 'mean'],
            'EXP_AMT_DRAWING_SUM': ['last'],
            'EXP_BALANCE_LIMIT_RATIO': ['last'],
            'EXP_CNT_DRAWING_SUM': ['last'],
            'EXP_MIN_PAYMENT_RATIO': ['last'],
            'EXP_PAYMENT_MIN_DIFF': ['last'],
            'EXP_MIN_PAYMENT_TOTAL_RATIO': ['last'],
            'EXP_AMT_INTEREST_RECEIVABLE': ['last'],
            'EXP_SK_DPD_RATIO': ['last']
        }
        # performing overall aggregations over SK_ID_PREV for all features
        cc_balance_aggregated_overall = self.cc_balance.groupby('SK_ID_PREV').agg(overall_aggregations)
        cc_balance_aggregated_overall.columns = ['_'.join(ele).upper() for ele in cc_balance_aggregated_overall.columns]
        cc_balance_aggregated_overall.rename(columns={'SK_ID_CURR_FIRST': 'SK_ID_CURR'}, inplace=True)

        # aggregating over SK_ID_PREV for different categories
        contract_status_categories = ['Active', 'Completed']
        cc_balance_aggregated_categories = pd.DataFrame()
        for i, contract_type in enumerate(contract_status_categories):
            group = self.cc_balance[self.cc_balance['NAME_CONTRACT_STATUS'] == contract_type].groupby('SK_ID_PREV').agg(
                aggregations_for_categories)
            group.columns = ['_'.join(ele).upper() + '_' + contract_type.upper() for ele in group.columns]
            if i == 0:
                cc_balance_aggregated_categories = group
            else:
                cc_balance_aggregated_categories = cc_balance_aggregated_categories.merge(group, on='SK_ID_PREV',
                                                                                          how='outer')
        # aggregating over SK_ID_PREV for rest of the categories
        cc_balance_aggregated_categories_rest = self.cc_balance[(self.cc_balance['NAME_CONTRACT_STATUS'] != 'Active') &
                                                                (
                                                                        self.cc_balance.NAME_CONTRACT_STATUS != 'Completed')].groupby(
            'SK_ID_PREV').agg(aggregations_for_categories)
        cc_balance_aggregated_categories_rest.columns = ['_'.join(ele).upper() + '_REST' for ele in
                                                         cc_balance_aggregated_categories_rest.columns]
        # merging all the categorical aggregations
        cc_balance_aggregated_categories = cc_balance_aggregated_categories.merge(cc_balance_aggregated_categories_rest,
                                                                                  on='SK_ID_PREV', how='outer')

        # aggregating over SK_ID_PREV for different years
        self.cc_balance['YEAR_BALANCE'] = self.cc_balance['MONTHS_BALANCE'] // 12
        cc_balance_aggregated_year = pd.DataFrame()
        for year in range(2):
            group = self.cc_balance[self.cc_balance['YEAR_BALANCE'] == year].groupby('SK_ID_PREV').agg(
                aggregations_for_year)
            group.columns = ['_'.join(ele).upper() + '_YEAR_' + str(year) for ele in group.columns]
            if year == 0:
                cc_balance_aggregated_year = group
            else:
                cc_balance_aggregated_year = cc_balance_aggregated_year.merge(group, on='SK_ID_PREV', how='outer')
        # aggregating over SK_ID_PREV for rest of years
        cc_balance_aggregated_year_rest = self.cc_balance[self.cc_balance['YEAR_BALANCE'] >= 2].groupby(
            'SK_ID_PREV').agg(aggregations_for_year)
        cc_balance_aggregated_year_rest.columns = ['_'.join(ele).upper() + '_YEAR_REST' for ele in
                                                   cc_balance_aggregated_year_rest.columns]
        # merging all the yearwise aggregations
        cc_balance_aggregated_year = cc_balance_aggregated_year.merge(cc_balance_aggregated_year_rest, on='SK_ID_PREV',
                                                                      how='outer')
        self.cc_balance = self.cc_balance.drop('YEAR_BALANCE', axis=1)

        # merging all the aggregations
        cc_aggregated = cc_balance_aggregated_overall.merge(cc_balance_aggregated_categories, on='SK_ID_PREV',
                                                            how='outer')
        cc_aggregated = cc_aggregated.merge(cc_balance_aggregated_year, on='SK_ID_PREV', how='outer')

        # one-hot encoding the categorical column NAME_CONTRACT_STATUS
        name_contract_dummies = pd.get_dummies(self.cc_balance.NAME_CONTRACT_STATUS, prefix='CONTRACT')
        contract_names = name_contract_dummies.columns.tolist()
        # merging the one-hot encoded feature with original table
        self.cc_balance = pd.concat([self.cc_balance, name_contract_dummies], axis=1)
        # aggregating over SK_ID_PREV the one-hot encoded columns
        aggregated_cc_contract = self.cc_balance[['SK_ID_PREV'] + contract_names].groupby('SK_ID_PREV').mean()

        # merging with the aggregated table
        cc_aggregated = cc_aggregated.merge(aggregated_cc_contract, on='SK_ID_PREV', how='outer')

        # now we will aggregate on SK_ID_CURR As seen from EDA, since most of the SK_ID_CURR had only 1 credit card,
        # so for aggregations, we will simply take the means
        cc_aggregated = cc_aggregated.groupby('SK_ID_CURR', as_index=False).mean()

        return cc_aggregated

    def main(self):
        """
        Function to be called for complete preprocessing and aggregation of credit_card_balance table.

        Inputs:
            self

        Returns:
            Final pre=processed and aggregated credit_card_balance table.
        """

        # loading the dataframe
        self.load_dataframe()
        # preprocessing and performing Feature Engineering
        self.data_preprocessing_and_feature_engineering()
        # aggregating over SK_ID_PREV and SK_ID_CURR
        cc_aggregated = self.aggregations()

        if self.verbose:
            print('\nDone preprocessing credit_card_balance.')
            print(f"\nInitial Size of credit_card_balance: {self.initial_size}")
            print(
                f'Size of credit_card_balance after Pre-Processing, Feature Engineering and Aggregation: {cc_aggregated.shape}')
            print(f'\nTotal Time Taken = {datetime.now() - self.start}')

        if self.dump_to_pickle:
            if self.verbose:
                print('\nPickling pre-processed credit_card_balance to cc_balance_afteng.pkl')
            with open(self.file_directory + 'cc_balance_afteng.pkl', 'wb') as f:
                pickle.dump(cc_aggregated, f)
            if self.verbose:
                print('Done.')
        if self.verbose:
            print('-' * 100)

        return cc_aggregated


# -----------------------------------------------------------------------------
# -- FUNCTION TO CREATE FEATURES USING THE INTERACTIONS BETWEEN VARIOUS TABLES
# -----------------------------------------------------------------------------


def create_new_features(data):
    """
    Function to create few more features after the merging of features, by using the
    interactions between various tables.

    Inputs:
        data: DataFrame

    Returns:
        None
    """
    
    path_prev_app_ML = "../P7_scoring_credit/preprocessing/previous_application_ML.pkl"
    with open(path_prev_app_ML, 'rb') as f:
        previous_application_ML = pickle.load(f)
    path_cc_balance_ML = "../P7_scoring_credit/preprocessing/cc_balance_ML.pkl"
    with open(path_cc_balance_ML, 'rb') as f:
        cc_balance_ML = pickle.load(f)
    path_isnt_pay_ML = "../P7_scoring_credit/preprocessing/installments_payments_ML.pkl"
    with open(path_isnt_pay_ML, 'rb') as f:
        installments_payments_ML = pickle.load(f)
    path_bureau_ML = "../P7_scoring_credit/preprocessing/bureau_ML.pkl"
    with open(path_bureau_ML, 'rb') as f:
        bureau_ML = pickle.load(f)

    # previous applications columns
    prev_annuity_columns = [ele for ele in previous_application_ML.columns if 'AMT_ANNUITY' in ele]
    for col in prev_annuity_columns:
        data['PREV_' + col + '_INCOME_RATIO'] = data[col] / (data['AMT_INCOME_TOTAL'] + 0.00001)
    prev_goods_columns = [ele for ele in previous_application_ML.columns if 'AMT_GOODS' in ele]
    for col in prev_goods_columns:
        data['PREV_' + col + '_INCOME_RATIO'] = data[col] / (data['AMT_INCOME_TOTAL'] + 0.00001)

    # credit_card_balance columns
    cc_amt_principal_cols = [ele for ele in cc_balance_ML.columns if 'AMT_RECEIVABLE_PRINCIPAL' in ele]
    for col in cc_amt_principal_cols:
        data['CC_' + col + '_INCOME_RATIO'] = data[col] / (data['AMT_INCOME_TOTAL'] + 0.00001)
    cc_amt_recivable_cols = [ele for ele in cc_balance_ML.columns if 'AMT_RECIVABLE' in ele]
    for col in cc_amt_recivable_cols:
        data['CC_' + col + '_INCOME_RATIO'] = data[col] / (data['AMT_INCOME_TOTAL'] + 0.00001)
    cc_amt_total_receivable_cols = [ele for ele in cc_balance_ML.columns if 'TOTAL_RECEIVABLE' in ele]
    for col in cc_amt_total_receivable_cols:
        data['CC_' + col + '_INCOME_RATIO'] = data[col] / (data['AMT_INCOME_TOTAL'] + 0.00001)

    # installments_payments columns
    installments_payment_cols = [ele for ele in installments_payments_ML.columns if
                                 'AMT_PAYMENT' in ele and 'RATIO' not in ele and 'DIFF' not in ele]
    for col in installments_payment_cols:
        data['INSTALLMENTS_' + col + '_INCOME_RATIO'] = data[col] / (data['AMT_INCOME_TOTAL'] + 0.00001)

    # POS_CASH_balance features have been created in its own dataframe itself

    # bureau and bureau_balance columns
    bureau_days_credit_cols = [ele for ele in bureau_ML.columns if
                               'DAYS_CREDIT' in ele and 'ENDDATE' not in ele and 'UPDATE' not in ele]
    for col in bureau_days_credit_cols:
        data['BUREAU_' + col + '_EMPLOYED_DIFF'] = data[col] - data['DAYS_EMPLOYED']
        data['BUREAU_' + col + '_REGISTRATION_DIFF'] = data[col] - data['DAYS_REGISTRATION']
    bureau_overdue_cols = [ele for ele in bureau_ML.columns if 'AMT_CREDIT' in ele and 'OVERDUE' in ele]
    for col in bureau_overdue_cols:
        data['BUREAU_' + col + '_INCOME_RATIO'] = data[col] / (data['AMT_INCOME_TOTAL'] + 0.00001)
    bureau_amt_annuity_cols = [ele for ele in bureau_ML.columns if 'AMT_ANNUITY' in ele and 'CREDIT' not in ele]
    for col in bureau_amt_annuity_cols:
        data['BUREAU_' + col + '_INCOME_RATIO'] = data[col] / (data['AMT_INCOME_TOTAL'] + 0.00001)