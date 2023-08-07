# -*- coding: utf-8 -*-

""" Personal library for tuning models hypermarameters and performance evaluation
"""

# ====================================================================
# Tools modeling
# Version : 0.0.0 - Created by RSP 16/06/2023
# ====================================================================

import pickle
# import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import numpy.typing as npt
import seaborn as sns
from IPython.display import display
from sklearn.metrics import confusion_matrix
# import shap
# from math import sqrt
# from collections import Counter
# from sklearn.base import BaseEstimator
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, median_absolute_error
from sklearn.metrics import recall_score, fbeta_score, precision_score, roc_auc_score, average_precision_score
from sklearn.model_selection import cross_validate

# from sklearn.dummy import DummyRegressor, DummyClassifier
# from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# from sklearn.svm import SVR
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.inspection import permutation_importance
# from sklearn.feature_selection import RFECV
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor


# --------------------------------------------------------------------
# -- VERSION
# --------------------------------------------------------------------
__version__ = '0.0.0'


# -----------------------------------------------------------------------
# -- BUSINESS METRIC
# -----------------------------------------------------------------------

def custom_score(y_real, y_pred, tn_weighting=1, fp_weighting=-1, fn_weighting=-10, tp_weighting=1):
    '''
    Business metric designed to minimise the risk to the bank of granting a loan by penalising false negatives.
    '''
    # tn (true negative): the loan is refunded: the bank earns money. => to maximize
    # fp (false positive) : the loan is denied in error: the bank loses interest,
    #    loses profit but does not actually lose money (type I error).
    # fn (false negative) : the loan is approved but the customer defaulting:
    #     the bank loses money (type II error). => to minimise
    # tp (true positive) : the loan is rightly denied: the bank neither gains nor loses money.

    '''
    Parameters
    ----------
    y_real : real class, mandatory (0 or 1).
    y_pred : predicted class, mandatory (0 or 1).
    
    tn_weighting : Weighting of True Negative, optional (1 by default), 
    fp_weighting : Weighting of False Positive rate, optional (-1 by default), 
    fn_weighting : Weighting of False Negative rate, optional (-10 by default), 
    tp_weighting : Weighting of True Positive rate, optional (1 by default), 
    
    Returns
    -------
    score : normalised gain (between 0 and 1) a high score indicates better performance
    
    '''
    # Confusion matrix
    (tn, fp, fn, tp) = confusion_matrix(y_true=y_real, y_pred=y_pred).ravel()
    # Total Gain
    real_gain = tn * tn_weighting + fp * fp_weighting + fn * fn_weighting + tp * tp_weighting
    # Maximum Gain : all predictions are correct
    gain_max = (fp + tn) * tn_weighting + (fn + tp) * tp_weighting
    # Minimum Gain : all predictions are false
    gain_min = (fp + tn) * fp_weighting + (fn + tp) * fn_weighting

    custom_score = (real_gain - gain_min) / (gain_max - gain_min)

    # Normalised gain (between 0 and 1) a high score indicates better performance
    return custom_score


# --------------------------------------------------------------------
# -- mlFlow
# --------------------------------------------------------------------


# ------------------------------------------------------------------------
# -- DATA BALANCE METHODS EVALUATION
# ------------------------------------------------------------------------

class balance_method_analysis:
    """
    Analyses the data balance method that improves scores the most.
    
    Contains XX functions:
    1. init method
    2.
    
    """

    def __init__(self, model,
                 X_train, X_val, y_train, y_val,
                 df_results, title,
                 show_table=True, show_confusion_matrix=True,
                 file_df_result_score='', verbose=True, dump_to_pickle=False):
        '''
        This function is used to initialize the class members 
        
         ------------
        @Inputs :
        ------------
            self
            model : initialized classification model, mandatory.
            X_train : train set matrix X, mandatory.
            X_val : validation set matrix X, mandatory.
            y_train : train set vecteur y, mandatory.
            y_val : test set, vecteur y, mandatory.
            df_results : dataframe for scores saving, mandatory
            title : experience name to record in the dataframe, mandatory.
            show_table : shows the results table (optional, default = True).
            file_df_result_score: Path, str, default = ''
                The path where the file exists. Include a '/' at the end of the path in input
            verbose: bool, default = True
                Whether to enable verbosity or not
            dump_to_pickle: bool, default = False
                Whether to pickle the final table with model scores or not
                
        ------------
        @Returns
        ------------
            None
        '''

        self.model = model
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.df_results = df_results
        self.title = title
        self.show_table = show_table
        self.show_confusion_matrix = show_confusion_matrix
        self.file_df_result_score = '/home/raquelsp/Documents/Openclassrooms/P7_implementez_modele_scoring/P7_travail/P7_scoring_credit/model_tests/df_results_scores.pkl'
        self.verbose = verbose
        self.dump_to_pickle = dump_to_pickle

    def train_model(self):
        '''
        Funtion to train and fit the model
        
        Inputs: self
        Returns: None
        '''
        if self.verbose:
            self.time_start = datetime.now()
            print('###################################################')
            print('#        Model fitting and training          #')
            print('###################################################')
            print("\n")

        # Start of execution
        time_start = datetime.now()
        # Training the model with the training set
        model.fit(self.X_train, self.y_train)
        # End of train exectuion
        time_end_train = datetime.now()
        # Predictions with the validation set
        y_pred = model.predict(self.X_val)
        self.y_pred = y_pred
        # End of prediction exectuion
        time_end = datetime.now()

        if self.verbose:
            print("Model trained")
            print(f"Time Taken to train the model = {datetime.now() - self.start}")

    def custom_score(self, tn_weighting=1, fp_weighting=-1, fn_weighting=-10, tp_weighting=1):
        '''
        Business metric designed to minimise the risk to the bank of granting a loan by penalising false negatives.
        '''
        # tn (true negative): the loan is refunded: the bank earns money. => to maximize
        # fp (false positive) : the loan is denied in error: the bank loses interest, loses profit but does not actually lose money (type I error).
        # fn (false negative) : the loan is approved but the customer defaulting: the bank loses money (type II error). => to minimise 
        # tp (true positif) : the loan is rightly denied: the bank neither gains nor loses money.

        '''
        Parameters
        ----------  
        tn_weighting : Weighting of True Negative, optional (1 by default), 
        fp_weighting : Weighting of False Positive rate, optional (-1 by default), 
        fn_weighting : Weighting of False Negative rate, optional (-10 by default), 
        tp_weighting : Weighting of True Positive rate, optional (1 by default), 
    
        Returns
        -------
        score : normalised gain (between 0 and 1) a high score indicates better performance    
        '''

        self.tn_weighting = tn_weighting
        self.fp_weighting = fp_weighting
        self.fn_weighting = fn_weighting
        self.tp_weighting = tp_weighting

        # Confusion matrix
        (tn, fp, fn, tp) = confusion_matrix(self.y_val, self.y_pred).ravel()
        # Total Gain
        real_gain = tn * self.tn_weighting + fp * self.fp_weighting + fn * self.fn_weighting + tp * self.tp_weighting
        # Maximum Gain : all predictions are correct
        gain_max = (fp + tn) * self.tn_weighting + (fn + tp) * self.tp_weighting
        # Minimum Gain : all predictions are false
        gain_min = (fp + tn) * self.fp_weighting + (fn + tp) * self.fn_weighting

        custom_score = (real_gain - gain_min) / (gain_max - gain_min)

        # Normalised gain (between 0 and 1) a high score indicates better performance
        return custom_score

    def compute_metrics(self):
        '''
        Funtion to compute model scores
        
        Inputs: self      
        Returns: None
        '''

        # Probabilities
        y_proba = model.predict_proba(self.X_val)[:, 1]

        # Recall
        recall = recall_score(self.y_val, self.y_pred)
        # Precision
        precision = precision_score(self.y_val, self.y_pred)
        # F-score ou Fbeta
        f1_score = fbeta_score(self.y_val, self.y_pred, beta=1)
        f5_score = fbeta_score(self.y_val, self.y_pred, beta=5)
        f10_score = fbeta_score(self.y_val, self.y_pred, beta=10)
        # Score ROC AUC
        roc_auc = roc_auc_score(self.y_val, y_proba)
        # Score AP
        ap_score = average_precision_score(self.y_val, y_proba)
        # Business metric
        business_metric = custom_score(self.y_val, self.y_pred)

        # Training runtime
        time_exec_train = time_end_train - time_start
        # Training+validation runtime
        time_execution = time_end - time_start

        # cross validation
        scoring = ['roc_auc', 'recall', 'precision']
        scores = cross_validate(model, self.X_train, self.y_train, cv=10,
                                scoring=scoring, return_train_score=True)

        # Saving Performance dataframe
        df_results = pd.concat([df_results, (pd.DataFrame({
            'Experience': [title],
            'Business_score': [business_metric]
        }))], axis=0)

        with open('model_tests/df_results_scores.pkl', 'wb') as f:
            pickle.dump(df_results, f)
        if self.verbose:
            print('Done.')
        if self.verbose:
            print('-' * 100)

        return df_results

    def construct_confusion_matrix(self):

        plt.figure(figsize=(6, 4))

        cm = confusion_matrix(self.y_val, self.y_pred)

        labels = ['Non-Defaulters', 'Defaulters']

        sns.heatmap(cm,
                    xticklabels=labels,
                    yticklabels=labels,
                    annot=True,
                    fmt='d',
                    cmap=plt.cm.Blues)
        plt.title(f'Confusion matrix for : {self.title}')
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.show()

    def main(self):
        '''
        Function to be called for analyze the data balance method that improves scores the most.
        
        Inputs:
            self
            
        Returns:
            Table with the model results scores.
        '''

        # Fit and predict
        self.train_model()

        # preprocessing the categorical features and creating new features
        self.custom_score()

        # computing model scores
        df_results = self.compute_metrics()

        with open('/model_tests/df_results_scores.pkl', 'wb') as f:
            pickle.dump(df_results, f)

        if self.show_table:
            if self.verbose:
                mask = df_results['Experience'] == self.title
                display(df_results[mask].style.hide(axis="index"))

        if self.show_confusion_matrix:
            self.construct_confusion_matrix()

        if self.verbose:
            print('Computing done')
            print(f'\nTotal Time Taken = {datetime.now() - self.time_start}')
            print('Done.')
            print('-' * 100)

        return df_results


# -----------------------------------------------------------------------
# -- SETTING THE PROBABILITY THRESHOLD
# -----------------------------------------------------------------------

def prob_threshold(model, X_val, y_val, title, df, score_function=custom_score, n=1, label=None, color='b', ax=None):
    """
    Determine the optimal probability threshold for the business metric.

    Parameters
    ----------
    model : model to train, mandatory.
    y_val : target true value
    X_val : data to test
    title : graphic title
    df : dataframe for data saving
    score_function : score for which we want to evaluate the probability threshold
    n : gain for class 1 (default) or 0.

    Returns
    -------
    None.
    """
    thresholds = np.arange(0, 1, 0.01)
    sav_gains = []

    for threshold in thresholds:
        # Model Score : n = 0 or 1
        y_proba = model.predict_proba(X_val)[:, n]

        # Score > solvency threshold : returns 1 otherwise 0
        y_pred = (y_proba > threshold)
        y_pred = np.multiply(y_pred, 1)

        # Saving the business metric score
        sav_gains.append(score_function(y_val, y_pred))

    df = pd.DataFrame({'Thresholds': thresholds,
                       'Business Score': sav_gains})

    # Maximum business metric score
    gain_max = df['Business Score'].max()
    print(f'Maximum business metric score : {gain_max}')
    # Optimal threshold for the business metric
    threshold_max = df.loc[df['Business Score'].argmax(), 'Thresholds']
    print(f'Optimal threshold : {threshold_max}')

    def plot_vline(df, x_col, y_col, ax, color='k', line_at='max'):
        """draw a vertical line at max or min value of 'y' on a plot"""
        row_idx = 0
        if line_at == 'max':
            row_idx = df[y_col].argmax()
        elif line_at == 'min':
            row_idx = df[y_col].argmin()
        line_x = df.loc[row_idx, x_col]
        line_label = f'{line_at} at {x_col}={line_x:.3f}'
        ax.axvline(line_x, c=color, linestyle="--", label=line_label)
        ax.legend(frameon=True)
        plt.suptitle(title)

    if ax is None:
        _, ax = plt.subplots()

    if label is None:
        label = f'{score_function.__name__}'
    sns.lineplot(x=thresholds, y=sav_gains, label=label, color=color, ax=ax)
    plot_vline(pd.DataFrame({'threshold': thresholds, 'score': sav_gains}),
               'threshold', 'score', color=color, ax=ax)
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("score")

    return df


# ------------------------------------------------------------------------
# -- MODEL SCORES TAKING ACCOUNT OF HTE THRESHOLD
# ------------------------------------------------------------------------

def process_classif_thresh(model, threshold_, X_train, X_val, y_train,
                           y_val, df_results_thresh, title,
                           show_results=True,
                           show_conf_matrix=True):
    """
    Runs a binary classification model, performs cross-validation and saves scores.
    ----------
    * Parameters
    model : initialised classification model, mandatory
    threshold_ : optimum probability threshold
    X_train : train set matrix, mandatory
    X_val : test set matrix, mandatory
    y_train : train target, mandatory
    y_val : test target, mandatory
    df_results_thresh : dataframe for saving scores, mandatory
    title : experience name to record in the dataframe, mandatory
    show_results : shows results dataframe (optional, default True)
    ----------
    * Returns
    df_results_thresh : dataframe with saved performances
    y_pred : model predictions
    """
    # Start computing
    time_start = datetime.now()

    # Model score : n = 0 ou 1
    # Probabilities
    y_proba = model.predict_proba(X_val)[:, 1]

    # Predictions from validation dataset
    # Score > probability threshold : returns 1 else 0
    y_pred = (y_proba > threshold_)
    y_pred = np.multiply(y_pred, 1)

    # End computing
    time_end = datetime.now()

    # Metrics computing
    # Recall
    recall = recall_score(y_val, y_pred)
    # Precision
    precision = precision_score(y_val, y_pred)
    # F-metric ou F-beta
    f1_score = fbeta_score(y_val, y_pred, beta=1)
    f2_score = fbeta_score(y_val, y_pred, beta=2)
    f5_score = fbeta_score(y_val, y_pred, beta=5)
    f10_score = fbeta_score(y_val, y_pred, beta=10)
    # Score ROC AUC
    roc_auc = roc_auc_score(y_val, y_proba)
    # Score average precision
    ap_score = average_precision_score(y_val, y_proba)
    # Business metric
    business_metric = custom_score(y_val, y_pred)

    # Total_duration
    time_execution = time_end - time_start

    # cross validation
    scoring = ['roc_auc', 'recall', 'precision']
    scores = cross_validate(model, X_train, y_train, cv=10,
                            scoring=scoring, return_train_score=True)

    (tn, fp, fn, tp) = confusion_matrix(y_val, y_pred).ravel()

    if show_results:
        mask = df_results_thresh['Experience'] == title
        display(df_results_thresh[mask].style.hide(axis="index"))

    if show_conf_matrix:
        plt.figure(figsize=(6, 4))

        cm = confusion_matrix(y_val, y_pred)

        labels = ['Non-Defaulters', 'Defaulters']

        sns.heatmap(cm,
                    xticklabels=labels,
                    yticklabels=labels,
                    annot=True,
                    fmt='d',
                    cmap=plt.cm.Blues)
        plt.title(f'Confusion matrix for : {title}')
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.show()

    # Performances saving
    df_results_thresh = df_results_thresh.append(pd.DataFrame({
        'Experience': [title],
        'Recall': [recall],
        'Precision': [precision],
        'f1_score': [f1_score],
        'f2_score': [f2_score],
        'f5_score': [f5_score],
        'f10_score': [f10_score],
        'roc_auc': [roc_auc],
        'ap_score': [ap_score],
        'Business_score': [business_metric],
        'TN': [tn],
        'FP': [fp],
        'FN': [fn],
        'TP': [tp],
        'total_duration': [time_execution],
        # Cross-validation
        'Train_roc_auc_CV': [scores['train_roc_auc'].mean()],
        'Train_roc_auc_CV +/-': [scores['train_roc_auc'].std()],
        'Test_roc_auc_CV': [scores['test_roc_auc'].mean()],
        'Test_roc_auc_CV +/-': [scores['test_roc_auc'].std()],
        'Train_recall_CV': [scores['train_recall'].mean()],
        'Train_recall_CV +/-': [scores['train_recall'].std()],
        'Test_recall_CV': [scores['test_recall'].mean()],
        'Test_recall_CV +/-': [scores['test_recall'].std()],
        'Train_precision_CV': [scores['train_precision'].mean()],
        'Train_precision_CV +/-': [scores['train_precision'].std()],
        'Test_precision_CV': [scores['test_precision'].mean()],
        'Test_precision_CV +/-': [scores['test_precision'].std()],
        }), ignore_index=True)


    # Saving results dataframe
    path_df_results = '/home/raquelsp/Documents/Openclassrooms/P7_implementez_modele_scoring/P7_travail' \
                          '/P7_scoring_credit/model_tests/df_results_thresh.pkl '
    with open(path_df_results, 'wb') as f:
        pickle.dump(df_results_thresh, f, pickle.HIGHEST_PROTOCOL)

    return df_results_thresh
