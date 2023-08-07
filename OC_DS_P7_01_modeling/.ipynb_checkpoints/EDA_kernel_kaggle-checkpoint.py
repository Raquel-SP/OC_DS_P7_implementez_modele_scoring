""" Library containing the functions used in the exploratory analysis from Rishabh Rao's github :
     https://github.com/rishabhrao1997/Home-Credit-Default-Risk
    For explanations:
     https://medium.com/thecyphy/home-credit-default-risk-part-1-3bfe3c7ddd7a
"""

# ! /usr/bin/env python3
# -*- coding: utf-8 -*-

# ====================================================================
# Functions EDA Kernel Kaggle -  project 7 Openclassrooms
# Version : 0.0.0 - Created by RSP 05/2023
# ====================================================================
from IPython.core.display import display
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split

import phik

# Plotly
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

plotly.offline.init_notebook_mode(connected=True)

# --------------------------------------------------------------------
# -- VERSION
# --------------------------------------------------------------------
__version__ = '0.0.0'


# ===========================================================================
# == EDA
# ===========================================================================


# ----------------------------------------------------------------------------
# -- ANALYSIS OF UNIQUE AND COMMON VALUES OF A VARIABLE ON SEVERAL DATAFRAMES
# ----------------------------------------------------------------------------

def common_values_dataframes(dataframe, dataframe2, dataframe3, id_cle):
    print('-' * 79)
    print(f'Number of unique values for {id_cle} in {dataframe.name}.csv : {len(dataframe[id_cle].unique())}')
    print(f'Number of unique values for SK_ID_CURR in {dataframe.name}.csv : {len(dataframe.SK_ID_CURR.unique())}')
    print(
        f'Number of unique values for SK_ID_CURR in {dataframe2.name}.csv and {dataframe.name}.csv : {len(set(dataframe2.SK_ID_CURR.unique()).intersection(set(dataframe.SK_ID_CURR.unique())))}')
    print(
        f'Number of common values for SK_ID_CURR in {dataframe2.name}.csv and {dataframe.name}.csv : {len(set(dataframe3.SK_ID_CURR.unique()).intersection(set(dataframe.SK_ID_CURR.unique())))}')
    print('-' * 79)
    duplicate = \
        dataframe.shape[0] - dataframe.duplicated().shape[0]
    print(f'Number of duplicate values in {dataframe.name}.csv : {duplicate}')
    print('-' * 79)


# --------------------------------------------------------------------
# -- PRINT UNIQUE VALUES FOR CATEGORICAL VARIABLES
# --------------------------------------------------------------------

def print_unique_categories(data, column_name, show_counts=False):
    """
    Function to print the unique values and their counts for categorical variables

        Inputs:
        data: DataFrame
            The DataFrame from which to print statistics
        column_name: str
            Column's name whose stats are to be printed
        show_counts: bool, default = False
            Whether to show counts of each category or not

    """

    print('-' * 100)
    print(f"The unique categories of '{column_name}' are:\n{data[column_name].unique()}")
    print('-' * 100)

    if show_counts:
        print(f"Counts of each category are:\n{data[column_name].value_counts()}")
        print('-' * 100)


# --------------------------------------------------------------------
# -- PLOT CATEGORICAL VARIABLES PIE PLOTS
# --------------------------------------------------------------------

def plot_categorical_variables_pie(data, column_name, plot_defaulter=True, hole=0):
    """
    Function to plot categorical variables Pie Plots

    Inputs:
        data: DataFrame
            The DataFrame from which to plot
        column_name: str
            Column's name whose distribution is to be plotted
        plot_defaulter: bool
            Whether to plot the Pie Plot for Defaulters or not
        hole: int, default = 0
            Radius of hole to be cut out from Pie Chart
    """

    if plot_defaulter:
        cols = 2
        specs = [[{'type': 'domain'}, {'type': 'domain'}]]
        titles = [f'Distribution of {column_name} for all Targets',
                  f'Percentage of Defaulters for each category of {column_name}']
    else:
        cols = 1
        specs = [[{'type': 'domain'}]]
        titles = [f'Distribution of {column_name} for all Targets']

    values_categorical = data[column_name].value_counts()
    labels_categorical = values_categorical.index

    fig = make_subplots(rows=1, cols=cols,
                        specs=specs,
                        subplot_titles=titles)

    fig.add_trace(go.Pie(values=values_categorical, labels=labels_categorical, hole=hole,
                         textinfo='label+percent', textposition='inside'), row=1, col=1)

    if plot_defaulter:
        percentage_defaulter_per_category = data[column_name][data.TARGET == 1].value_counts() * 100 / data[
            column_name].value_counts()
        percentage_defaulter_per_category.dropna(inplace=True)
        percentage_defaulter_per_category = percentage_defaulter_per_category.round(2)

        fig.add_trace(go.Pie(values=percentage_defaulter_per_category, labels=percentage_defaulter_per_category.index,
                             hole=hole, textinfo='label+value', hoverinfo='label+value'), row=1, col=2)

    fig.update_layout(title=f'Distribution of {column_name}')
    fig.show()


# --------------------------------------------------------------------
# -- PLOT CATEGORICAL VARIABLES BAR PLOTS
# --------------------------------------------------------------------

def plot_categorical_variables_bar(data, column_name, figsize=(18, 6), percentage_display=True, plot_defaulter=True,
                                   rotation=0, horizontal_adjust=0, fontsize_percent='xx-small'):
    """
    Function to plot Categorical Variables Bar Plots

    Inputs:
        data: DataFrame
            The DataFrame from which to plot
        column_name: str
            Column's name whose distribution is to be plotted
        figsize: tuple, default = (18,6)
            Size of the figure to be plotted
        percentage_display: bool, default = True
            Whether to display the percentages on top of Bars in Bar-Plot
        plot_defaulter: bool
            Whether to plot the Bar Plots for Defaulters or not
        rotation: int, default = 0
            Degree of rotation for x-tick labels
        horizontal_adjust: int, default = 0
            Horizontal adjustment parameter for percentages displayed on the top of Bars of Bar-Plot
        fontsize_percent: str, default = 'xx-small'
            Fontsize for percentage Display

    """

    print(f"Total Number of unique categories of {column_name} = {len(data[column_name].unique())}")

    plt.figure(figsize=figsize, tight_layout=False)
    sns.set(style='whitegrid', font_scale=1.2)

    # plotting overall distribution of category
    plt.subplot(1, 2, 1)
    data_to_plot = data[column_name].value_counts().sort_values(ascending=False)
    ax = sns.barplot(x=data_to_plot.index, y=data_to_plot, palette='Set1')

    if percentage_display:
        total_datapoints = len(data[column_name].dropna())
        for p in ax.patches:
            ax.text(p.get_x() + horizontal_adjust, p.get_height() + 0.005 * total_datapoints,
                    '{:1.02f}%'.format(p.get_height() * 100 / total_datapoints), fontsize=fontsize_percent)

    plt.xlabel(column_name, labelpad=10)
    plt.title(f'Distribution of {column_name}', pad=20)
    plt.xticks(rotation=rotation)
    plt.ylabel('Counts')

    # plotting distribution of category for Defaulters
    if plot_defaulter:
        percentage_defaulter_per_category = (data[column_name][data.TARGET == 1].value_counts() * 100 / data[
            column_name].value_counts()).dropna().sort_values(ascending=False)

        plt.subplot(1, 2, 2)
        sns.barplot(x=percentage_defaulter_per_category.index, y=percentage_defaulter_per_category, palette='Set2')
        plt.ylabel('Percentage of Defaulter per category')
        plt.xlabel(column_name, labelpad=10)
        plt.xticks(rotation=rotation)
        plt.title(f'Percentage of Defaulters for each category of {column_name}', pad=20)
    plt.show()


# --------------------------------------------------------------------
# -- PLOT CONTINUOUS VARIABLES DISTRIBUTION
# --------------------------------------------------------------------

def plot_continuous_variables(data, column_name, plots=['displot', 'CDF', 'box', 'violin'],
                              scale_limits=None, figsize=(20, 8), histogram=True, log_scale=False):
    """
    Function to plot continuous variables distribution

    Inputs:
        data: DataFrame
            The DataFrame from which to plot.
        column_name: str
            Column's name whose distribution is to be plotted.
        plots: list, default = ['displot', 'CDF', box', 'violin']
            List of plots to plot for Continuous Variable.
        scale_limits: tuple (left, right), default = None
            To control the limits of values to be plotted in case of outliers.
        figsize: tuple, default = (20,8)
            Size of the figure to be plotted.
        histogram: bool, default = True
            Whether to plot histogram along with displot or not.
        log_scale: bool, default = False
            Whether to use log-scale for variables with outlying points.
    """

    data_to_plot = data.copy()
    if scale_limits:
        # taking only the data within the specified limits
        data_to_plot[column_name] = data[column_name][
            (data[column_name] > scale_limits[0]) & (data[column_name] < scale_limits[1])]

    number_of_subplots = len(plots)
    plt.figure(figsize=figsize)
    sns.set_style('whitegrid')

    for i, ele in enumerate(plots):
        plt.subplot(1, number_of_subplots, i + 1)
        plt.subplots_adjust(wspace=0.25)

        if ele == 'CDF':
            # making the percentile DataFrame for both positive and negative Class Labels
            percentile_values_0 = data_to_plot[data_to_plot.TARGET == 0][[column_name]].dropna().sort_values(
                by=column_name)
            percentile_values_0['Percentile'] = [ele / (len(percentile_values_0) - 1) for ele in
                                                 range(len(percentile_values_0))]

            percentile_values_1 = data_to_plot[data_to_plot.TARGET == 1][[column_name]].dropna().sort_values(
                by=column_name)
            percentile_values_1['Percentile'] = [ele / (len(percentile_values_1) - 1) for ele in
                                                 range(len(percentile_values_1))]

            plt.plot(percentile_values_0[column_name], percentile_values_0['Percentile'], color='red',
                     label='Non-Defaulters')
            plt.plot(percentile_values_1[column_name], percentile_values_1['Percentile'], color='black',
                     label='Defaulters')
            plt.xlabel(column_name)
            plt.ylabel('Probability')
            plt.title('CDF of {}'.format(column_name))
            plt.legend(fontsize='medium')
            if log_scale:
                plt.xscale('log')
                plt.xlabel(column_name + ' - (log-scale)')

        if ele == 'distplot':
            sns.distplot(data_to_plot[column_name][data['TARGET'] == 0].dropna(),
                         label='Non-Defaulters', hist=False, color='red')
            sns.distplot(data_to_plot[column_name][data['TARGET'] == 1].dropna(),
                         label='Defaulters', hist=False, color='black')
            plt.xlabel(column_name)
            plt.ylabel('Probability Density')
            plt.legend(fontsize='medium')
            plt.title("Dist-Plot of {}".format(column_name))
            if log_scale:
                plt.xscale('log')
                plt.xlabel(f'{column_name} (log scale)')

        if ele == 'violin':
            sns.violinplot(x='TARGET', y=column_name, data=data_to_plot)
            plt.title("Violin-Plot of {}".format(column_name))
            if log_scale:
                plt.yscale('log')
                plt.ylabel(f'{column_name} (log Scale)')

        if ele == 'box':
            sns.boxplot(x='TARGET', y=column_name, data=data_to_plot)
            plt.title("Box-Plot of {}".format(column_name))
            if log_scale:
                plt.yscale('log')
                plt.ylabel(f'{column_name} (log Scale)')

    plt.show()


# --------------------------------------------------------------------
# -- PRINT PERCENTILE VALUES
# --------------------------------------------------------------------

def print_percentiles(data, column_name, percentiles=None):
    """
    Function to print percentile values for given column

    Inputs:
        data: DataFrame
            The DataFrame from which to print percentiles
        column_name: str
            Column's name whose percentiles are to be printed
        percentiles: list, default = None
            The list of percentiles to print, if not given, default are printed
    """

    print('-' * 100)
    if not percentiles:
        percentiles = list(range(0, 80, 25)) + list(range(90, 101, 2))
    for i in percentiles:
        print(f'The {i}th percentile value of {column_name} is {np.percentile(data[column_name].dropna(), i)}')
    print("-" * 100)


# --------------------------------------------------------------------
# -- HEATMAP OF THE VALUES OF Phi-K CORRELATION COEFFICIENT
# --------------------------------------------------------------------

def plot_phik_matrix(data, categorical_columns, figsize=(20, 20),
                     mask_upper=True, tight_layout=True, linewidth=0.1,
                     fontsize=10, cmap='Blues', show_target_top_corr=True,
                     target_top_columns=10):
    """
    Function to Phi_k matrix for categorical features
    We will draw a heat map of the values of the Phi-K correlation coefficient
    between the 2 variables.

    The Phi-K coefficient is similar to the correlation coefficient except that
    it can be used with a pair of categorical variables to check whether one variable
    shows some sort of association with the other categorical variable. Its maximum
    value can be 1, indicating a maximum association between two categorical variables.
    Inputs:
        data: DataFrame
            The DataFrame from which to build correlation matrix
        categorical_columns: list
            List of categorical columns whose PhiK values are to be plotted
        figsize: tuple, default = (25,23)
            Size of the figure to be plotted
        mask_upper: bool, default = True
            Whether to plot only the lower triangle of heatmap or plot full.
        tight_layout: bool, default = True
            Whether to keep tight layout or not
        linewidth: float/int, default = 0.1
            The linewidth to use for heatmap
        fontsize: int, default = 10
            The font size for the X and Y tick labels
        cmap: str, default = 'Blues'
            The colormap to be used for heatmap
        show_target_top_corr: bool, default = True
            Whether to show top/highly correlated features with Target.
        target_top_columns: int, default = 10
            The number of top correlated features with target to display
    """
    # first fetching only the categorical features
    data_for_phik = data[categorical_columns].astype('object')
    phik_matrix = data_for_phik.phik_matrix()

    print('-' * 79)

    if mask_upper:
        mask_array = np.ones(phik_matrix.shape)
        mask_array = np.triu(mask_array)
    else:
        mask_array = np.zeros(phik_matrix.shape)

    plt.figure(figsize=figsize, tight_layout=tight_layout)
    sns.heatmap(
        phik_matrix,
        annot=False,
        mask=mask_array,
        linewidth=linewidth,
        cmap=cmap)
    plt.xticks(rotation=90, fontsize=fontsize)
    plt.yticks(rotation=0, fontsize=fontsize)
    plt.title("Phi-K Correlation Heatmap of categorical variables",
              fontsize=fontsize + 4)
    plt.show()

    print("-" * 79)

    if show_target_top_corr:
        # Seeing the top columns with highest correlation with the target
        # variable in application_train
        print("The categories with the highest values of Phi-K correlation with the target variable are as follows :")
        phik_df = pd.DataFrame(
            {'Variable': phik_matrix.TARGET.index[1:], 'Phik-Correlation': phik_matrix.TARGET.values[1:]})
        phik_df = phik_df.sort_values(by='Phik-Correlation', ascending=False)
        display(phik_df.head(target_top_columns).style.hide(axis="index"))
        print("-" * 79)


# -----------------------------------------------
# -- CORRELATION MATRIX FOR CONTINUOUS VARIABLES
# -----------------------------------------------

class correlation_matrix:
    """
    Class to plot heatmap of Correlation Matrix and print Top Correlated Features with Target.
    Contains three methods:
        1. init method
        2. plot_correlation_matrix method
        3. target_top_corr method
    """

    def __init__(
            self,
            data,
            columns_to_drop,
            figsize=(25, 23),
            mask_upper=True,
            tight_layout=True,
            linewidth=0.1,
            fontsize=10,
            cmap='Blues'):
        """
        Function to initialize the class members.

        Inputs:
            data: DataFrame
                The DataFrame from which to build correlation matrix
            columns_to_drop: list
                Columns which have to be dropped while building the correlation matrix (for example the Loan ID)
            figsize: tuple, default = (25,23)
                Size of the figure to be plotted
            mask_upper: bool, default = True
                Whether to plot only the lower triangle of heatmap or plot full.
            tight_layout: bool, default = True
                Whether to keep tight layout or not
            linewidth: float/int, default = 0.1
                The linewidth to use for heatmap
            fontsize: int, default = 10
                The font size for the X and Y tick labels
            cmap: str, default = 'Blues'
                The colormap to be used for heatmap

        Returns:
            None
        """

        self.data = data
        # self.columns_to_drop = columns_to_drop
        to_drop = columns_to_drop + ['TARGET']
        self.to_drop = to_drop
        self.corr_data = self.data.drop(self.to_drop, axis=1).corr()
        self.figsize = figsize
        self.mask_upper = mask_upper
        self.tight_layout = tight_layout
        self.linewidth = linewidth
        self.fontsize = fontsize
        self.cmap = cmap

    def plot_correlation_matrix(self):
        """
        Function to plot the Correlation Matrix Heatmap

        Inputs:
            self

        Returns:
            None
        """

        # print('-' * 79)
        # building the correlation dataframe

        if self.mask_upper:
            # masking the heatmap to show only lower triangle. This is to save
            # the RAM.
            mask_array = np.ones(self.corr_data.shape)
            mask_array = np.triu(mask_array)
        else:
            mask_array = np.zeros(self.corr_data.shape)

        plt.figure(figsize=self.figsize, tight_layout=self.tight_layout)
        sns.heatmap(
            self.corr_data,
            annot=False,
            mask=mask_array,
            linewidth=self.linewidth,
            cmap=self.cmap)
        plt.xticks(rotation=90, fontsize=self.fontsize)
        plt.yticks(fontsize=self.fontsize)
        plt.title("Heatmap de corrélation des variables numériques", fontsize=20)
        plt.show()
        # print("-" * 100)

    def target_top_corr(self, target_top_columns=10):
        """
        Function to return the Top Correlated features with the Target

        Inputs:
            self
            target_top_columns: int, default = 10
                The number of top correlated features with target to display

        Returns:
            Top correlated features DataFrame.
        """

        phik_target_arr = np.zeros(self.corr_data.shape[1])
        # calculating the Phik-Correlation with Target
        for index, column in enumerate(self.corr_data.columns):
            phik_target_arr[index] = self.data[[
                'TARGET', column]].phik_matrix().iloc[0, 1]
        # getting the top correlated columns and their values
        top_corr_target_df = pd.DataFrame(
            {'Column Name': self.corr_data.columns, 'Phik-Correlation': phik_target_arr})
        top_corr_target_df = top_corr_target_df.sort_values(
            by='Phik-Correlation', ascending=False)

        return top_corr_target_df.iloc[:target_top_columns]


# ===========================================================================
# == PARTIE FEATURES SELECTION
# ===========================================================================


def plot_feature_importances(df, threshold=0.9):
    """
    Plots 15 most important features and the cumulative importance of features.
    Prints the number of features needed to reach threshold cumulative importance.
    Source : 
    https://www.kaggle.com/willkoehrsen/introduction-to-feature-selection
    Parameters
    --------
    df : dataframe
        Dataframe of feature importances. Columns must be feature and importance
    threshold : float, default = 0.9
        Threshold for prining information about cumulative importances
    Return
    --------
    df : dataframe
        Dataframe ordered by feature importances with a normalized column (sums to 1)
        and a cumulative importance column    
    """

    plt.rcParams['font.size'] = 18

    # Sort features according to importance
    df = df.sort_values('importance', ascending=False).reset_index()

    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize=(10, 12))
    ax = plt.subplot()

    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:30]))),
            df['importance_normalized'].head(30),
            align='center', edgecolor='k')

    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:30]))))
    ax.set_yticklabels(df['feature'].head(30))

    # Plot labeling
    plt.xlabel('Importance normalisée');
    plt.title('Features Importances')
    plt.show()

    # Cumulative importance plot
    plt.figure(figsize=(8, 6))
    plt.plot(list(range(len(df))), df['cumulative_importance'], 'r-')
    plt.xlabel('Nombre de variables');
    plt.ylabel('Cumulative Importance');
    plt.title('Cumulative Feature Importance');
    plt.show();

    importance_index = np.min(np.where(df['cumulative_importance'] > threshold))
    print('%d variables nécessaires pour %0.2f de cumulative importance' % (importance_index + 1, threshold))

    return df


def identify_zero_importance_features(train, train_labels, iterations=2):
    """
    Identify zero importance features in a training dataset based on the 
    feature importances from a gradient boosting model. 
    
    Parameters
    --------
    train : dataframe
        Training features
        
    train_labels : np.array
        Labels for training data
        
    iterations : integer, default = 2
        Number of cross validation splits to use for determining feature importances
    """

    # Initialize an empty array to hold feature importances
    feature_importances = np.zeros(train.shape[1])

    # Create the model with several hyperparameters
    model = lgb.LGBMClassifier(objective='binary', boosting_type='goss', n_estimators=10000, class_weight='balanced')

    # Fit the model multiple times to avoid overfitting
    for i in range(iterations):
        # Split into training and validation set
        train_features, valid_features, train_y, valid_y = train_test_split(train, train_labels, test_size=0.25,
                                                                            random_state=i)

        # Train using early stopping
        model.fit(train_features, train_y, early_stopping_rounds=100, eval_set=[(valid_features, valid_y)],
                  eval_metric='auc', verbose=200)

        # Record the feature importances
        feature_importances += model.feature_importances_ / iterations

    feature_importances = pd.DataFrame({'feature': list(train.columns), 'importance': feature_importances}).sort_values(
        'importance', ascending=False)

    # Find the features with zero importance
    zero_features = list(feature_importances[feature_importances['importance'] == 0.0]['feature'])
    print('\nThere are %d features with 0.0 importance' % len(zero_features))

    return zero_features, feature_importances
