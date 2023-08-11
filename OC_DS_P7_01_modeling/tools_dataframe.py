""" Personal library for manipulating the dataframe,
    variable description, column renaming, etc.
"""

# coding: utf-8

# ====================================================================
# Version : 0.0.1 - Created by RSP 05/2023
# ====================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# --------------------------------------------------------------------
# -- VERSION
# --------------------------------------------------------------------
__version__ = '0.0.0'


# --------------------------------------------------------------------
# -- DATASET DESCRIPTION
# --------------------------------------------------------------------
def complet_description(df):
    nb_row = df.index.size
    nb_col = df.columns.size
    Types = pd.DataFrame(df.dtypes).T.rename(index={0: 'Type'})
    Null = pd.DataFrame(df.isna().sum()).T.rename(index={0: 'null'})
    Duplicated = pd.DataFrame(df.shape[0] - df.isna().sum() - df.nunique()).T.rename(index={0: 'Duplicated'})
    PercCount = pd.DataFrame(100 - 100 * (df.isna().sum()) / nb_row).T.rename(index={0: 'Filling percentage'})
    Describe = df.describe(include='all')
    infor = pd.concat([Types, Null, Duplicated, PercCount, Describe], axis=0).T.sort_values(
        "Filling percentage").reset_index()
    infor = infor.rename(columns={"index": "Variable"})
    return infor


# --------------------------------------------------------------------
# -- DATA TYPES VISUALISATION
# --------------------------------------------------------------------
def visu_dataTypes(df):
    n_types = df['Type'].value_counts()
    values_types = n_types.values.tolist()
    blues_n = sns.color_palette(palette="Blues", n_colors=n_types.shape[0])
    labes_types = df['Type'].unique()

    plt.pie(values_types, labels=labes_types,
            colors=blues_n, autopct="%.0f%%", startangle=90)
    plt.title("Data types")
    plt.show()


# ---------------------------------------------------------------------------
# -- GET MISSING VALUES
# ---------------------------------------------------------------------------

def get_missing_values(df_work, percentage, show_heatmap, retour=False):
    """Information about missing values
       @param : df_work dataframe, obligatory
                   percentage : boolean si True shows the number heatmap
                   show_heatmap : boolean si  shows the number heatmap
    """

    # 1. Total missing values
    nb_nan_tot = df_work.isna().sum().sum()
    nb_data_tot = np.product(df_work.shape)
    pourc_nan_tot = round((nb_nan_tot / nb_data_tot) * 100, 2)
    print(
        f'Missing values : {nb_nan_tot} NaN for {nb_data_tot} data ({pourc_nan_tot} %)')

    if percentage:
        print("-------------------------------------------------------------")
        print("Number and % of missing values by variable\n")
        # 2. Visualization of number and percentage of missing values per variable
        values = df_work.isnull().sum()
        percentage = 100 * values / len(df_work)
        table = pd.concat([values, percentage.round(2)], axis=1)
        table.columns = [
            'Number of missing values',
            '% of missing values']
        display(table[table['Number of missing values'] != 0]
                .sort_values('% of missing values', ascending=False)
                .style.background_gradient('seismic'))

    if show_heatmap:
        print("-------------------------------------------------------------")
        print("Heatmap of missing values")
        # 3. Heatmap of missing values
        plt.figure(figsize=(20, 10))
        sns.heatmap(df_work.isna(), cbar=False)
        plt.show()

    if retour:
        return table


# --------------------------------------------------------------------
# -- COLUMN FILLING VISUALIZATION
# --------------------------------------------------------------------
def column_filling_visu(df):
    fig, ax = plt.subplots(figsize=(20, 4))

    ax.bar(df["Variable"],
           df["Filling percentage"], color="#2994ff")
    plt.axhline(y=100, color='r', linestyle='-')
    ax.set_ylabel("%")
    ax.set_title("Columns filling percentage")
    plt.xticks(rotation=90)
    # plt.tight_layout
    plt.show()


# --------------------------------------------------------------------
# -- DESCRIPTION OF CATEGORICAL VARIABLES
# --------------------------------------------------------------------
def univ_cate_vari(dataframe, feature, tableau=True, graphDistsrib=True):
    """
     Displays the frequency table and a bar graph with the frequency
     of the words
    @Params IN : 
        dataframe : DataFrame, required
        feature : categorical variables to be analyzed, required
        tableau : booléen, True = displays the frequency table
        graphDistsrib : booléen,  True = displays the percentage distribution
        graph"""

    if tableau:
        df = dataframe[feature].value_counts().to_frame().reset_index()
        df = df.rename(columns={'index': feature, feature: 'count'})
        df['Frequency_%'] = 100 * df['count'] / (dataframe.shape[0])
        display(df.head(10).style.hide(axis="index"))

    if graphDistsrib:
        plt.figure(figsize=(4, 6))
        df_graph = df.sort_values('Frequency_%', ascending=False).head(20)
        sns.barplot(data=df_graph, x=feature, y='Frequency_%', palette="Blues")
        plt.title("Distribution of " + feature)
        plt.show()


# --------------------------------------------------------------------
# -- CALCULATE BIN DISTRIBUTION
# --------------------------------------------------------------------


def distribution_variables_plages(
        dataframe, variable, liste_bins):
    """
    Retourne les plages des pourcentages des valeurs pour le découpage transmis
    Parameters
    ----------
    @param : dataframe : DataFrame, obligatoire
                variable : variable à découper obligatoire
                liste_bins: liste des découpages facultatif int ou pintervallindex
    @returns : dataframe des plages de nan
    """
    nb_lignes = len(dataframe[variable])
    s_gpe_cut = pd.cut(
        dataframe[variable],
        bins=liste_bins).value_counts().sort_index()
    df_cut = pd.DataFrame({'Plage': s_gpe_cut.index,
                           'nb_données': s_gpe_cut.values})
    df_cut['%_données'] = [
        (row * 100) / nb_lignes for row in df_cut['nb_données']]

    return df_cut.style.hide_index()


# ---------------------------------------------------------------------------------
# -- MANAGE STRONG CORRELATIONS
# ---------------------------------------------------------------------------------

def managing_correlations(df, correl_threshold=0.7):
    """
    Function removing features with a correlation over 0.7
    ------------
    @ Parameters :
    * df : dataframe of train dataset, mandatory
    * correl_threshold : correlation value over which features will be removed, default = 0.7
    ------------
    @ Return :
    * df : dataframe of train dataset after removing strong correlations
    * cols_corr_a_supp : list of columns to remove
    """

    # Absolute value correlation matrix to avoid having to manage
    # positive and negative correlations separately
    corr = df.corr().abs()

    # Only the part above the diagonal is retained so that
    # the correlations are taken into account only once (axial symmetry).
    corr_triangle = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    # Variables with a Pearson coef > correl_threshold?
    cols_corr_a_supp = [var for var in corr_triangle.columns
                        if any(corr_triangle[var] > correl_threshold)]
    print(f'There are {len(cols_corr_a_supp)} variables with strong correlation to be removed.\n')

    # Drop variables with strong correlation
    print(f'Original shape : {df.shape}')
    df.drop(columns=cols_corr_a_supp, inplace=True)
    print(f'Post correlation managing shape : {df.shape}')

    return df, cols_corr_a_supp
