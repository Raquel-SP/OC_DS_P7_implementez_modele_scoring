import pandas as pd
import numpy as np
import pytest
import pickle
import os

#from tools_dataframe import managing_correlations

# Get the path where the data and model files are stored
path = os.path.dirname(os.path.realpath(__file__))
# getting our trained model from a file we created earlier
model = pickle.load(open(path + "/lgbm_best_model.pkl", "rb"))


@pytest.fixture()
def uncorrelated_columns():
    return pd.DataFrame(data=np.random.randint(1, 1500, (20, 10)),
                        columns=["col1", "col2", "col3", "col4", "col5", "col6",
                                 "col7", "col8", "col9", "col10"])


@pytest.fixture()
def correlated_columns(uncorrelated_columns):
    df_correlated_columns = uncorrelated_columns.copy()
    df_correlated_columns["col12"] = df_correlated_columns["col10"] * 3
    df_correlated_columns["col13"] = df_correlated_columns["col10"] * 4
    df_correlated_columns["col14"] = df_correlated_columns["col10"] * 5
    df_correlated_columns["col15"] = df_correlated_columns["col10"] * 6
    return df_correlated_columns


@pytest.fixture()
def defaulter():
    defaulter_dict = {'AMT_CREDIT': [-0.95870674],
                      'REGION_POPULATION_RELATIVE': [0.31059277],
                      'DAYS_BIRTH': [-1.1464291],
                      'DAYS_EMPLOYED': [0.81809485],
                      'DAYS_REGISTRATION': [-1.0982149],
                      'DAYS_ID_PUBLISH': [-0.3980247],
                      'EXT_SOURCE_1': [-1.7760689],
                      'EXT_SOURCE_2': [-0.624678],
                      'EXT_SOURCE_3': [-2.087083],
                      'DAYS_LAST_PHONE_CHANGE': [-0.8601075],
                      'CREDIT_INCOME_RATIO': [-0.9514161],
                      'CREDIT_ANNUITY_RATIO': [-0.055926606],
                      'CREDIT_GOODS_RATIO': [3.229657],
                      'INCOME_EXT_RATIO': [-0.013467618],
                      'CAR_EMPLOYED_RATIO': [-0.0025503282],
                      'EXT_SOURCE_MEAN': [-2.2556078],
                      'OBS_30_CREDIT_RATIO': [-0.2567478],
                      'REGIONS_RATING_INCOME_MUL_0': [-16.588861],
                      'DAYS_CREDIT_MEAN_OVERALL': [0.7237888],
                      'AMT_CREDIT_SUM_MEAN_OVERALL': [0.55233026],
                      'CURRENT_DEBT_TO_CREDIT_RATIO_MEAN_OVERALL': [-0.010803135],
                      'AMT_CREDIT_SUM_SUM_CREDITACTIVE_CLOSED': [1.2377608],
                      'CURRENT_CREDIT_DEBT_DIFF_MIN_CREDITACTIVE_CLOSED': [-0.22316569],
                      'DAYS_CREDIT_MEAN_CREDITACTIVE_ACTIVE': [-0.22322178],
                      'CURRENT_DEBT_TO_CREDIT_RATIO_MEAN_CREDITACTIVE_ACTIVE': [-0.016790975],
                      'CURRENT_CREDIT_DEBT_DIFF_MEAN_CREDITACTIVE_ACTIVE': [-0.24403836],
                      'HOUR_APPR_PROCESS_START_MEAN_LAST_5': [-0.41149962],
                      'DAYS_DECISION_MEAN_LAST_5': [-0.05640277],
                      'SELLERPLACE_AREA_MEAN_LAST_5': [-0.12682344],
                      'INTEREST_SHARE_MEAN_LAST_5': [-0.15072185],
                      'INTEREST_RATE_MAX_FIRST_2': [-0.052409474],
                      'DAYS_PAYMENT_DIFF_MEAN_MEAN': [-0.73244673],
                      'DAYS_PAYMENT_DIFF_MIN_MEAN': [0.07355375],
                      'DAYS_PAYMENT_DIFF_MAX_MEAN': [-1.2385377]}

    return pd.DataFrame(defaulter_dict)


@pytest.fixture()
def nonDefaulter():
    nonDefaulter_dict = {'AMT_CREDIT': [0.18875991],
                         'REGION_POPULATION_RELATIVE': [1.8329418],
                         'DAYS_BIRTH': [-0.55178857],
                         'DAYS_EMPLOYED': [0.36354733],
                         'DAYS_REGISTRATION': [-0.7757514],
                         'DAYS_ID_PUBLISH': [-0.4980613],
                         'EXT_SOURCE_1': [1.509266],
                         'EXT_SOURCE_2': [1.4020475],
                         'EXT_SOURCE_3': [0.36645404],
                         'DAYS_LAST_PHONE_CHANGE': [-0.46702847],
                         'CREDIT_INCOME_RATIO': [-1.008102],
                         'CREDIT_ANNUITY_RATIO': [2.0826716],
                         'CREDIT_GOODS_RATIO': [-0.9760997],
                         'INCOME_EXT_RATIO': [-0.017981336],
                         'CAR_EMPLOYED_RATIO': [-0.002548686],
                         'EXT_SOURCE_MEAN': [1.671374],
                         'OBS_30_CREDIT_RATIO': [0.83856386],
                         'REGIONS_RATING_INCOME_MUL_0': [3.1941118],
                         'DAYS_CREDIT_MEAN_OVERALL': [0.53855884],
                         'AMT_CREDIT_SUM_MEAN_OVERALL': [0.976577],
                         'CURRENT_DEBT_TO_CREDIT_RATIO_MEAN_OVERALL': [-0.010803135],
                         'AMT_CREDIT_SUM_SUM_CREDITACTIVE_CLOSED': [0.5647966],
                         'CURRENT_CREDIT_DEBT_DIFF_MIN_CREDITACTIVE_CLOSED': [0.09424135],
                         'DAYS_CREDIT_MEAN_CREDITACTIVE_ACTIVE': [-0.012835029],
                         'CURRENT_DEBT_TO_CREDIT_RATIO_MEAN_CREDITACTIVE_ACTIVE': [-0.016790975],
                         'CURRENT_CREDIT_DEBT_DIFF_MEAN_CREDITACTIVE_ACTIVE': [1.243053],
                         'HOUR_APPR_PROCESS_START_MEAN_LAST_5': [1.951634],
                         'DAYS_DECISION_MEAN_LAST_5': [1.1104324],
                         'SELLERPLACE_AREA_MEAN_LAST_5': [-0.16110553],
                         'INTEREST_SHARE_MEAN_LAST_5': [0.5683178],
                         'INTEREST_RATE_MAX_FIRST_2': [-0.061651107],
                         'DAYS_PAYMENT_DIFF_MEAN_MEAN': [0.15903808],
                         'DAYS_PAYMENT_DIFF_MIN_MEAN': [0.17341563],
                         'DAYS_PAYMENT_DIFF_MAX_MEAN': [-0.13489808]}
    return pd.DataFrame(nonDefaulter_dict)


def test_managing_correlations(uncorrelated_columns, correlated_columns):
    correl_threshold = 0.7
    #cols_corr_a_supp = []
    corr = correlated_columns.corr().abs()
    corr_triangle = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    cols_corr_a_supp = [var for var in corr_triangle.columns
                        if any(corr_triangle[var] > correl_threshold)]
    correlated_columns.drop(columns=cols_corr_a_supp, inplace=True)

    #correlated_columns, cols_corr_a_supp = managing_correlations(correlated_columns)

    assert correlated_columns.shape == uncorrelated_columns.shape


def test_classif_defaulters(defaulter):
    # Client credit Score
    proba = (model.predict_proba(defaulter)[:, 1]) * 100
    credit_score = round(proba[0], 2)

    # Whether the credit is denied or not
    opti_proba_threshold = 0.65  # Probability threshold optimised during modeling
    prediction = np.where(credit_score >= (opti_proba_threshold * 100),
                          'Defaulter', 'Non_defaulter')
    assert prediction == 'Defaulter'


def test_classif_nonDefaulters(nonDefaulter):
    # Client credit Score
    proba = (model.predict_proba(nonDefaulter)[:, 1]) * 100
    credit_score = round(proba[0], 2)

    # Whether the credit is denied or not
    opti_proba_threshold = 0.65  # Probability threshold optimised during modeling
    prediction = np.where(credit_score >= (opti_proba_threshold * 100),
                          'Defaulter', 'Non_defaulter')
    assert prediction == 'Non_defaulter'
