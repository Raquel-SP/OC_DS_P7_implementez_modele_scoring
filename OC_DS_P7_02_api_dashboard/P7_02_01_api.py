# API to load clients' data, predict the scoring, product shap values

# Import required libraries
import flask
from flask import jsonify
import pickle
import pandas as pd
import numpy as np
import shap
import os

# Initializing the API
app = flask.Flask(__name__)
app.config["DEBUG"] = True

# Get the path where the data and model files are stored
path = os.path.dirname(os.path.realpath(__file__))


# Home page
@app.route('/', methods=['GET'])
def home():
    return """
<h1>Home Credit Scoring API</h1>
<p>API for :
 - load customer data
 - load the prediction model
 - predict the scoring of a customer
 - produce shap values to explain the prediction
 </p>
"""


# getting our trained model from a file we created earlier
model = pickle.load(open(path + "/resources/lgbm_best_model.pkl", "rb"))

#------------------#
# getting the data #
#------------------#
# Data for credit score computing
X_val = pd.read_csv(path + "/resources/X_val.csv")

# Create a subsample
X_val_0 = X_val[X_val["TARGET"] == 0].sample(100, random_state=84)
X_val_1 = X_val[X_val["TARGET"] == 1].sample(100, random_state=84)
X_val_sub = pd.concat([X_val_0, X_val_1]).reset_index(drop=True)
X_val_sub = X_val_sub.drop(columns=["TARGET"])

columns_val = X_val_sub.columns.tolist()
clients_val = X_val_sub['SK_ID_CURR'].tolist() # Get the list of clients in X_val

#Data post-feature engineering before standardisation
final_train_data = pickle.load(open(path + "/resources/final_train_data.pkl", "rb"))
# final_train_data = final_train_data[final_train_data["SK_ID_CURR"].isin(clients_val)]
    
# Reduce information to features used for modeling
train_data_modeling_val = final_train_data[columns_val]

# Get original application_train dataset for extract general information
application_train = pd.read_csv(path + "/resources/application_train.csv")

# General client information
client_info_columns = ['SK_ID_CURR',
                       'TARGET',
                       'DAYS_BIRTH', 'CODE_GENDER',
                       'NAME_FAMILY_STATUS',
                       'CNT_CHILDREN', 'NAME_EDUCATION_TYPE',
                       'NAME_INCOME_TYPE', 'DAYS_EMPLOYED',
                       'AMT_INCOME_TOTAL',
                       'NAME_CONTRACT_TYPE',
                       'AMT_GOODS_PRICE',
                       'NAME_HOUSING_TYPE',]
general_info = application_train[client_info_columns]
# Change age features to years (instead of days)
# Transform DAYS_BIRTH to years
general_info['AGE'] = np.trunc(np.abs(general_info['DAYS_BIRTH']  / 365)).astype('int8')
# Transform DAYS_EMPLOYED to years
general_info['YEARS_WORKING'] = np.trunc(np.abs(general_info['DAYS_EMPLOYED'] / 365)).astype('int8')
# Transform gender : 0 = Féminin et 1 = Masculin
general_info['GENDER'] = ['Woman' if row == 0 else 'Man' for row in general_info['CODE_GENDER']]

general_info = general_info.drop(columns=['DAYS_BIRTH', 'DAYS_EMPLOYED', 'CODE_GENDER'])

# Combine general information comming from app_train dataset
# with information used for modeling after feature engineering
dashboard_dataset = train_data_modeling_val.merge(general_info, on='SK_ID_CURR', how='left')


# defining a route to get clients data used for prediction
@app.route("/dataAPI", methods=["GET"])
def get_apidata():
    df_all = X_val_sub.to_dict("list")
    return jsonify(df_all)

# defining a route to get non scaled and general data
@app.route("/dataGeneral", methods=["GET"])
def get_Dashboarddata():
    df_all = dashboard_dataset.to_dict("list")
    return jsonify(df_all)


# defining a route to get clients data and prediction
@app.route("/dataAPI/client/<client_id>", methods=["GET"])
def client_data(client_id):
    # filter the data thanks to the id from the request
    client_modeliced = X_val_sub[X_val_sub["SK_ID_CURR"] == int(client_id)]
    client_to_modelice = client_modeliced.drop(columns="SK_ID_CURR")
        
    # Client credit Score
    client_modeliced['proba']=(model.predict_proba(client_to_modelice)[:,1])*100
    client_modeliced['credit_score']=round(client_modeliced['proba'], 2)
    
    # Whether the credit is denied or not
    proba_threshold = 0.5
    client_modeliced['prediction'] = np.where(client_modeliced['credit_score']>=(proba_threshold*100), 'Credit denied', 'Credit approuved')

    # calculate features importance in this prediction
    X_val_sub_shap = X_val_sub.drop(columns="SK_ID_CURR")
    explainer = shap.KernelExplainer(model.predict_proba, X_val_sub_shap)

    shap_values = explainer.shap_values(client_to_modelice)

    # add the shap values in the dataframe
    client_modeliced["expected"] = explainer.expected_value[1]
    new_line = [99999] + list(shap_values[1])[0].tolist()\
        + [0, 0, 0, explainer.expected_value[1]]
    client_modeliced.loc[1] = new_line

    # create the dictionary to be sent
    sample = client_modeliced.to_dict("list")
    # returning sample and prediction objects as json
    return jsonify(sample)


# To remove when online :
if __name__ == '__main__':
    app.run()
