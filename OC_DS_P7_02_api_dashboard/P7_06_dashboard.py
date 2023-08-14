# Display a dashboard in streamlit in order to diplay and explain client's scoring for credit



# Import required librairies
import streamlit as st
import streamlit.components.v1 as components
from urllib.request import urlopen
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from PIL import Image
import shap
import plotly.graph_objects as go

#################################################
#               Streamlit settings              #
#################################################
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title = "Prêt à dépenser - Scoring Crédit", layout="wide")


#################################################
#               API configuration               #
#################################################
# local :
# API_url = "http://127.0.0.1:8000/"
# online :
API_url = "http://35.180.169.135:8000/"
# Initialize javascript for shap plots
shap.initjs()


#################################################
#                    Get data                   #
#################################################
# Get all the clients data through an API

# Data used for computing probability
json_url_API = urlopen(API_url + "dataAPI")
API_data_all = json.loads(json_url_API.read())
API_data_all = pd.DataFrame(API_data_all)

# General and non standardized Data
json_url_Gene = urlopen(API_url + "dataGeneral")
general_data_all = json.loads(json_url_Gene.read())
general_data_all = pd.DataFrame(general_data_all)



#################################################
#               General appearance              #
#################################################
# Create a search buton
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    

def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

#local_css("style.css")
#remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')


# Deleting default margins
padding = 1
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)



#################################################
#                    Settings                   #
#################################################
# Create the list of features used for prediction
columns_model = list(API_data_all.drop(columns=["SK_ID_CURR"]).columns)


# General client information
client_pers_info = ['SK_ID_CURR',
                    'AGE', 'GENDER',
                    'NAME_FAMILY_STATUS',
                    'CNT_CHILDREN', 'NAME_EDUCATION_TYPE',
                    'NAME_INCOME_TYPE', 'YEARS_WORKING',
                   ]
df_pers_client_info = general_data_all[client_pers_info]

client_econom_info = ['SK_ID_CURR',
                      'AMT_INCOME_TOTAL', 
                      'NAME_CONTRACT_TYPE',
                      'AMT_GOODS_PRICE',
                      'NAME_HOUSING_TYPE', ]
df_econo_client_info = general_data_all[client_econom_info]


# Create the list of clients
client_list = general_data_all["SK_ID_CURR"].tolist()

# Create the list of columns non Scaled
columns_nonSca = list(general_data_all.drop(columns=["SK_ID_CURR", "TARGET"]).columns)


# Prepare data for the comparison plots, data non scaled
data_plot = general_data_all[columns_nonSca]

# Create the list of booleans columns
categories = data_plot.select_dtypes(include=['object']).columns.tolist()

# Create lists for categorical and other columns
col_one = []
col_std = []
for col in data_plot.columns:
    if "_cat_" in col or col in categories:
        col_one.append(col)
    else:
        col_std.append(col)

# Re-order the columns
columns = col_std + col_one

# Create the reference data (mean, median, mode)
Z = general_data_all[columns]
data_ref = pd.DataFrame(index=Z.columns)
data_ref["mean"] = Z.mean()
data_ref["median"] = Z.median()
data_ref["mode"] = Z.mode().iloc[0, :]
data_ref = data_ref.transpose()
# Remove values when not relevant
for col in data_ref.columns:
    if col in col_one:
        data_ref.loc["median", col] = np.NaN
    else:
        data_ref.loc["mode", col] = np.NaN


##################################################
#                     IMAGES                     #
##################################################
# Logo
logo =  Image.open("/home/raquelsp/Documents/Openclassrooms/P7_implementez_modele_scoring/P7_travail/P7_scoring_credit/OC_DS_P7_01_modeling/img/logo_projet.png") 


#################################################
#                   Functions                   #
#################################################

# Display shap force plot
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)







#################################################
#              Dashboard structure              #
#################################################

header = st.container()
select_client = st.container()
client_info = st.container()
scoring_info = st.container()
main_feat = st.container()
plus_client_info = st.container()
distrib_def_nondef = st.container()


#===============================================#
#                     Header                    #
#===============================================#

with header:
    st.markdown("<style>body{background-color: #F5F5F5; }</style>", unsafe_allow_html=True)
    
    main_header = '<b><p style="font-family:sans-serif; color:#2994ff; text-align: center; font-size: 42px;">Prêt à dépenser</p></b>'
    sub_header = '<b><p style="font-family:sans-serif; text-align: center; font-size: 20px;">Présentation du scoring client et explication du score attribué.</p></b>'
    st.markdown(main_header, unsafe_allow_html=True)
    st.markdown(sub_header, unsafe_allow_html=True)
    st.text("""
    Ce tableau de bord permet de visualiser les informations concernant un client et les comparer avec celles de l'ensemble de la base de données.
    Vous devez fournir l’identifiant du client (_SK_ID_CURR_) dans le champ de recherches ci-dessous et vos accèderez aux informations rélatives au score.
    La barre à gauche permet l'accées à :
        * les comparaisons avec l'ensemble des clients
        * la distribution des clients défaillants vs. non défaillants para variable
    """)
    st.divider()


#===============================================#
#                    Sidebar                    #
#===============================================#

st.sidebar.image(logo, width=240, caption=" Dashboard - Aide à la décision",
                 use_column_width='always')


#===============================================#
#                 Select client                 #
#===============================================#
with select_client:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("")
        col1.header("Selectionner l'identifiant du client")
    with col2:
        id_client = st.selectbox("Selectionner l'identifiant du client", options=client_list)
    st.divider()


#===============================================#
#                    Sidebar                    #
#===============================================#
# manually define the default columns names
default = ['AMT_CREDIT',
           'DAYS_BIRTH',
           'EXT_SOURCE_1',
           'CREDIT_ANNUITY_RATIO',
           'CREDIT_GOODS_RATIO',
           'EXT_SOURCE_MEAN',
           'REGIONS_RATING_INCOME_MUL_0',
           'AMT_CREDIT_SUM_MEAN_OVERALL',
           'CURRENT_DEBT_TO_CREDIT_RATIO_MEAN_CREDITACTIVE_ACTIVE',
           'DAYS_PAYMENT_DIFF_MIN_MEAN']

# In the sidebar allow to select several columns in the list
st.sidebar.subheader("Comparer avec l'ensemble des clients")
columns_selected = st.sidebar.multiselect("Sélectionner les informations à afficher",
                                 columns_nonSca, default=None)

# Create the sub-lists of columns for the plots in the selected columns
columns_categ = []
columns_quanti = []
for col in columns:
    if col in columns_selected:
        if col in categories:
            columns_categ.append(col)
        else:
            columns_quanti.append(col)

# In the sidebar allow to select the vislalization of defaulters / non defaulters per column
st.sidebar.subheader("Distribution des défaillants par variable")
defaulters_distrib = st.sidebar.checkbox("Voir les distributions")

#===============================================#
#           General client information          #
#===============================================#
with client_info:

    st.subheader('Information sur le client')
    client_pers_info = df_pers_client_info[df_pers_client_info['SK_ID_CURR'] == id_client].iloc[:, :]
    client_pers_info.set_index('SK_ID_CURR', inplace=True)
    st.table(client_pers_info)
    client_econ_info = df_econo_client_info[df_econo_client_info['SK_ID_CURR'] == id_client].iloc[:, :]
    client_econ_info.set_index('SK_ID_CURR', inplace=True)
    st.table(client_econ_info)
    

#===============================================#
#             Client score information          #
#===============================================#

with scoring_info:
    st.divider()
    with st.spinner("Traitement en cours..."):
        # Get the data for the selected client and the prediction from an API
        json_url_client_api = urlopen(API_url + "dataAPI/client/" + str(id_client))
        API_data_client = json.loads(json_url_client_api.read())
        df_client_api = pd.DataFrame(API_data_client)
        # Get the data non scaled for the selected client
        df_client_nonSca = general_data_all[general_data_all['SK_ID_CURR'] == id_client].iloc[:, :]

    col1, col2 = st.columns([1, 1])
    with col1:
        st.header("Score du client")
    
        fig = go.Figure(go.Indicator(
            mode = 'gauge+number+delta',
            # Score du client en % df_dashboard['SCORE_CLIENT_%']
            value = df_client_api["credit_score"][0],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': 'Crédit score du client', 'font': {'size': 24}},
    
            gauge = {'axis': {'range': [None, 100],
                              'tickwidth': 3,
                              'tickcolor': 'darkblue'},
                     'bar': {'color': 'white', 'thickness' : 0.25},
                     'bgcolor': 'white',
                     'borderwidth': 2,
                     'bordercolor': 'gray',
                     'steps': [{'range': [0, 25], 'color': 'Green'},
                       {'range': [25, 49.49], 'color': 'LimeGreen'},
                       {'range': [49.5, 50.5], 'color': 'red'},
                       {'range': [50.51, 75], 'color': 'Orange'},
                       {'range': [75, 100], 'color': 'Crimson'}],
                     'threshold': {'line': {'color': 'white', 'width': 10},
                                   'thickness': 0.8,
                                   'value': df_client_api["credit_score"][0]}}))
    
        fig.update_layout(paper_bgcolor='white',
                          height=400, width=600,
                          font={'color': 'darkblue', 'family': 'Arial'})
        st.plotly_chart(fig)
        with col2:
            if df_client_api["prediction"][0]=="Credit denied":
                st.error("Risque élevé")
            else:
                st.success("Risque faible")


#===============================================#
#     Main features, local interpretability     #
#===============================================#

with main_feat:
    st.divider()
    st.header("Variables ayant le plus d'impact sur le score du client")
    
    # List the columns we don't need for the explanation
    columns_info = ["SK_ID_CURR", "proba", "credit_score", "prediction", "expected"]
    # Store the columns names to use them in the shap plots
    client_data = df_client_api.drop(columns = columns_info).iloc[0:1,:]
    features_analysis = client_data.columns
    
    # store the data we want to explain in the shap plots
    data_explain = np.asarray(client_data)
    shap_values = df_client_api.drop(columns = columns_info).iloc[1,:].values
    expected_value = df_client_api["expected"].tolist()[0]
    
    
    # display a shap force plot
    fig_force = shap.force_plot(base_value = expected_value,
                                shap_values = shap_values,
                                features = data_explain,
                                feature_names = features_analysis) 
    st_shap(fig_force)
    
    
    # in an expander, display the client's data and comparison with average
    with st.expander("Ouvrir pour afficher l'analyse détaillée des variables avec le plus d'influence"):
        col1, col2 = st.columns([1, 1])
        with col1:
            # display a shap waterfall plot
            fig_water = shap.plots._waterfall.waterfall_legacy(expected_value,
                                        shap_values = shap_values,
                                        feature_names = features_analysis) 
            st.pyplot(fig_water, use_container_width=True)
        with col2:
            # display a shap decision plot
            fig_decision = shap.decision_plot(
                expected_value, 
                shap_values,
                features_analysis)
            st.pyplot(fig_decision, use_container_width=True)

#===============================================#
#          Detailed information client          #
#===============================================#

with plus_client_info:
    if len(columns_selected) >=1 :
        st.divider()
        html_info_detail="""
            <div class="card">
                <div class="card-body" style="border-radius: 10px 10px 0px 0px;
                      background: #DEC7CB; padding-top: 5px; width: auto;
                      height: 40px;">
                      <h3 class="card-title" style="background-color:#DEC7CB; color:Crimson;
                          font-family:Georgia; text-align: center; padding: 0px 0;">
                          Comparer le client
                      </h3>
                </div>
            </div>
            """
        st.markdown(html_info_detail, unsafe_allow_html=True)
        with st.spinner("Construction des graphiques en cours..."):
            # Display plots that compare the current client within all the clients
            # For quantitative features first
            
            
            # Set the style for average values markers
            meanpointprops = dict(markeredgecolor="black", markersize=5,
                                  markerfacecolor="yellow", markeredgewidth=0.66)
            # Build the boxplots for each feature
            for col in columns_selected:
                if col not in categories:
                    # Initialize the figure
                    f, ax = plt.subplots(figsize=(4, 2))
                    sns.boxplot(
                            data=data_plot[col],
                            orient="h",
                            whis=3,
                            palette="crest",
                            linewidth=0.7,
                            width=0.6,
                            showfliers=False,
                            showmeans=True,
                            meanprops=meanpointprops)
                    # Add in a point to show current client
                    sns.stripplot(
                            data=df_client_nonSca[col],
                            orient="h",
                            size=5,
                            palette="blend:firebrick,firebrick",
                            marker="D",
                            edgecolor="black",
                            linewidth=0.66)
                    
                    # Manage y labels style
                    ax.set(xlabel="", ylabel=col)
                    plt.rcParams['font.size']=6
                    # Remove axes lines
                    sns.despine(trim=True, left=True, bottom=True, top=True)
                    # Removes ticks for x and y
                    plt.tick_params(left=False, bottom=False)
                    # Add separation lines for y values
                    #lines = [ax.axhline(y, color="grey", linestyle="solid", linewidth=0.7)
                    #                        for y in np.arange(0.5, len(col)-1, 1)]
                    # Proxy artists to add a legend
                    average = mlines.Line2D([], [], color="yellow", marker="^",
                                            linestyle="None", markeredgecolor="black",
                                            markeredgewidth=0.66, markersize=5, label="moyenne")
                    current = mlines.Line2D([], [], color="firebrick", marker="D",
                                            linestyle="None", markeredgecolor="black",
                                            markeredgewidth=0.66, markersize=5, label="client courant")
                    plt.legend(handles=[average, current], bbox_to_anchor=(1, 1), fontsize="small")
                    # Display the plot
                    st.pyplot(f, use_container_width=False)
            
            # Then for categories
            # First ceate a summary dataframe
            df_plot_cat = pd.DataFrame()
            for col in columns_categ:
                df_plot_cat = pd.concat([
                        df_plot_cat,
                        pd.DataFrame(data_plot[col].value_counts()).transpose()])
                with plt.style.context("_mpl-gallery-nogrid"):
                    st.text("Le client en étude est dans la catégorie: " + str(df_client_nonSca[col].tolist()[0]))
                    plt.figure(figsize=(18, 6), tight_layout=False)
                    sns.set(style='whitegrid', font_scale=1.2)
                    # plotting overall distribution of category
                    plt.subplot(1, 2, 1)
                    data_to_plot = data_plot[col].value_counts().sort_values(ascending=False)
                    ax = sns.barplot(x=data_to_plot.index, y=data_to_plot, palette='Set1')
    
                    total_datapoints = len(data_plot[col].dropna())
                    for p in ax.patches:
                        ax.text(p.get_x(), p.get_height() + 0.005 * total_datapoints,
                            '{:1.02f}%'.format(p.get_height() * 100 / total_datapoints), fontsize='xx-small')
        
                    plt.xlabel(col, labelpad=10)
                    plt.title(f'Distribution de {col}', pad=20)
                    plt.xticks(rotation=0)
                    plt.ylabel('Counts')
        
                    # plotting distribution of category for Defaulters
                    percentage_defaulter_per_category = (general_data_all[col][general_data_all.TARGET == 1].value_counts() * 100 / general_data_all[
                        col].value_counts()).dropna().sort_values(ascending=False)
            
                    plt.subplot(1, 2, 2)
                    sns.barplot(x=percentage_defaulter_per_category.index, y=percentage_defaulter_per_category, palette='Set2')
                    plt.ylabel('Pourcentage de DEFAILLANTS par catégorie')
                    plt.xlabel(col, labelpad=10)
                    plt.xticks(rotation=0)
                    plt.title(f'Pourcentage de DEFAILLANTS par catégorie dans {col}', pad=20)
                    st.pyplot()
                             
            # in an expander, display the client's data and comparison with average
            st.divider()
            with st.expander("Ouvrir pour afficher les données détaillées"):
                temp_df = pd.concat([client_data, data_ref])
                new_df = temp_df.transpose()
                new_df.columns = ["Client (" + str(id_client) + ")", "Moyenne",
                                  "Médiane", "Mode"]
                st.table(new_df.loc[columns_selected,:])
            
 
#===============================================#
#     Defaulters/nonDefaulters distribution     #
#===============================================#

with distrib_def_nondef :
    st.divider()
    if defaulters_distrib :
        html_facteurs_influence="""
            <div class="card">
                <div class="card-body" style="border-radius: 10px 10px 0px 0px;
                      background: #DEC7CB; padding-top: 5px; width: auto;
                      height: 40px;">
                      <h3 class="card-title" style="background-color:#DEC7CB; color:Crimson;
                          font-family:Georgia; text-align: center; padding: 0px 0;">
                          Distribution des variables générale/pour les défaillants
                      </h3>
                </div>
            </div>
            """
        st.markdown(html_facteurs_influence, unsafe_allow_html=True)
        with st.expander('Distribution des variables',
                              expanded=True):
            choix = st.selectbox("Choisir une variable : ", columns_nonSca)
            
            if choix not in categories :
                plt.figure(figsize=(5, 3))
                sns.set_style('whitegrid')
                sns.distplot(general_data_all[choix][general_data_all['TARGET'] == 0].dropna(),
                         label='Non-Défaillants', hist=False, color='red')
                sns.distplot(general_data_all[choix][general_data_all['TARGET'] == 1].dropna(),
                             label='Défaillants', hist=False, color='black')
                plt.xlabel(choix)
                plt.ylabel('Probability Density')
                plt.legend(fontsize='xx-small')
                plt.title("Dist-Plot of {}".format(choix))
                st.pyplot(use_container_width=False)
                
            if choix in categories :
                distrib = ["TARGET", choix]
                df_plot_cat = pd.DataFrame(general_data_all[distrib].value_counts()).reset_index()
                df_plot_cat_pivot = df_plot_cat.pivot(index=choix, columns='TARGET', values=0)
                df_plot_cat_pivot.reset_index(inplace=True)
                df_plot_cat_pivot.columns = [choix, "Non défaillants", "Défaillants"]
                ax=df_plot_cat_pivot.plot(x=choix, kind='bar', stacked=True,
                        figsize=(4, 3), fontsize=6, rot=45)
                ax.set_xlabel(None)
                ax.set_title('Distribution de défaillants/non-défaillants par catégorie',pad=20, fontdict={'fontsize':8})
                
                st.pyplot(use_container_width=False)
                            
