# Display a dashboard in streamlit in order to diplay and explain client's scoring for credit



# Import required librairies
import streamlit as st
import streamlit.components.v1 as components
from urllib.request import urlopen
import json
import datetime
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
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
API_url = "http://127.0.0.1:5000/"
# online :
#API_url = "http://bl0ws.pythonanywhere.com/"
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

local_css("style.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')


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

# Create the list of columns
columns = list(general_data_all.drop(columns="SK_ID_CURR").columns)


##################################################
#                     IMAGES                     #
##################################################
# Logo
logo =  Image.open("/home/raquelsp/Documents/Openclassrooms/P7_implementez_modele_scoring/P7_travail/P7_scoring_credit/img/logo_projet.png") 


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
    Vous devez fournir l’identifiant du client (_SK_ID_CURR_) dans le champ de recherches et vos accèderez à toutes les informations.
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
#           Denied / Approuved credit           #
#===============================================#


    
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
        json_url_client = urlopen(API_url + "dataAPI/client/" + str(id_client))
        API_data_client = json.loads(json_url_client.read())
        df = pd.DataFrame(API_data_client)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.header("Score du client")
    
        fig = go.Figure(go.Indicator(
            mode = 'gauge+number+delta',
            # Score du client en % df_dashboard['SCORE_CLIENT_%']
            value = df["credit_score"][0],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': 'Crédit score du client', 'font': {'size': 24}},
    
            gauge = {'axis': {'range': [None, 100],
                              'tickwidth': 3,
                              'tickcolor': 'darkblue'},
                     'bar': {'color': 'white', 'thickness' : 0.25},
                     'bgcolor': 'white',
                     'borderwidth': 2,
                     'bordercolor': 'gray',
                     'steps': [{'range': [0, 33.49], 'color': 'Green'},
                               {'range': [33.5, 64.49], 'color': 'LimeGreen'},
                               {'range': [64.5, 65.49], 'color': 'red'},
                               {'range': [65.5, 100], 'color': 'crimson'}],
                     'threshold': {'line': {'color': 'white', 'width': 10},
                                   'thickness': 0.8,
                                   'value': df["credit_score"][0]}}))
    
        fig.update_layout(paper_bgcolor='white',
                          height=400, width=600,
                          font={'color': 'darkblue', 'family': 'Arial'})
        st.plotly_chart(fig)
        with col2:
            if df["prediction"][0]=="Credit denied":
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
    columns_info = ["SK_ID_CURR", "proba", "credit_score", "prediction"]
    # Store the columns names to use them in the shap plots
    client_data = df.drop(columns = columns_info).iloc[0:1,:]
    features_analysis = client_data.columns
    
    # store the data we want to explain in the shap plots
    data_explain = np.asarray(client_data)
    shap_values = df.drop(columns = columns_info).iloc[1,:].values
    expected_value = df["expected"][0]
    
    # display a shap force plot
    fig_force = shap.force_plot(
        expected_value,
        shap_values,
        data_explain,
        feature_names=features_analysis,
    ) 
    st_shap(fig_force)
    
    
    # in an expander, display the client's data and comparison with average
    with st.expander("Ouvrir pour afficher l'analyse détaillée"):
        st.text("""
        Ce tableau de bord permet de visualiser les informations concernant un client et les comparer avec celles de l'ensemble de la base de données.
        Vous devez fournir l’identifiant du client (_SK_ID_CURR_) dans le champ de recherches et vos accèderez à toutes les informations.
        """)
        # display a shap waterfall plot
        fig_water = shap.plots._waterfall.waterfall_legacy(
            expected_value,
            shap_values,
            feature_names=features_analysis,
            max_display=10)
        st.pyplot(fig_water)
        
        # display a shap decision plot
        fig_decision = shap.decision_plot(
            expected_value, 
            shap_values,
            features_analysis)
        st_shap(fig_decision)
    


