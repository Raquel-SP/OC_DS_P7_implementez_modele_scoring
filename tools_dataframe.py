""" Personal library for manipulating the dataframe,
    variable description, column renaming, etc.
"""

#! /usr/bin/env python3
# coding: utf-8

# ====================================================================
# Version : 0.0.1 - Created by RSP 05/2023
# ====================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import re
import os

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
    Types = pd.DataFrame(df.dtypes).T.rename(index={0:'Type'}) 
    Null = pd.DataFrame(df.isna().sum()).T.rename(index={0:'null'})
    Duplicated = pd.DataFrame(df.shape[0]-df.isna().sum()-df.nunique()).T.rename(index={0:'Duplicated'}) 
    PercCount = pd.DataFrame(100-100*(df.isna().sum())/nb_row).T.rename(index={0:'Filling percentage'})
    Describe = df.describe(datetime_is_numeric=True, include='all')
    infor = pd.concat([Types,Null,Duplicated,PercCount, Describe], axis =0).T.sort_values("Filling percentage").reset_index() 
    infor = infor.rename(columns={"index":"Variable"})
    return infor


# --------------------------------------------------------------------
# -- DATA TYPES VISUALISATION
# --------------------------------------------------------------------
def visu_dataTypes (df):
    n_types = df['Type'].value_counts()
    values_types = n_types.values.tolist()
    blues_n = sns.color_palette(palette="Blues", n_colors=n_types.shape[0])
    labes_types = df['Type'].unique()

    plt.pie(values_types, labels=labes_types,
            colors=blues_n, autopct="%.0f%%", startangle=90)
    plt.title("Data types")
    plt.show()


# --------------------------------------------------------------------
# -- COLUMN FILLING VISUALIZATION
# --------------------------------------------------------------------
def column_filling_visu(df):
    fig, ax = plt.subplots(figsize=(20, 4))

    ax.bar(df["Variable"],
           df["Filling percentage"], color="#2994ff")
    plt.axhline(y =100, color = 'r', linestyle = '-')
    ax.set_ylabel("%")
    ax.set_title("Columns filling percentage")
    plt.xticks(rotation=90)
    plt.tight_layout
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
        df = df.rename(columns={'index':feature, feature:'num_entries'})
        df['Frequency_%'] = 100*df['num_entries']/((dataframe.shape[0]))
        display(df.head(10).style.hide(axis="index"))

    if graphDistsrib:
        plt.figure(figsize=(4,6))
        df_graph = df.sort_values('Frequency_%', ascending=False).head(20)
        sns.barplot(data= df_graph, x=feature, y='Frequency_%' , palette = "Blues")
        plt.title("Distribution of " + feature)
        plt.show()

