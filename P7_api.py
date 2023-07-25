# API to load clients' data, predict the scoring, product shap values

# Import required librairies
import flask
from flask import jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
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
model = pickle.load(open(path + "/model_tests/lgbm_best_model.pkl", "rb"))
# getting the data
X = pd.read_csv(path + "/data_dashboard.csv")
# List of selected features for modelling
selected_features = ['AMT_CREDIT',
                     'REGION_POPULATION_RELATIVE',
                     'DAYS_BIRTH',
                     'DAYS_EMPLOYED',
                     'DAYS_REGISTRATION',
                     'DAYS_ID_PUBLISH',
                     'EXT_SOURCE_1',
                     'EXT_SOURCE_2',
                     'EXT_SOURCE_3',
                     'DAYS_LAST_PHONE_CHANGE',
                     'CREDIT_INCOME_RATIO',
                     'CREDIT_ANNUITY_RATIO',
                     'CREDIT_GOODS_RATIO',
                     'INCOME_EXT_RATIO',
                     'CAR_EMPLOYED_RATIO',
                     'EXT_SOURCE_MEAN',
                     'OBS_30_CREDIT_RATIO',
                     'ENQ_CREDIT_RATIO',
                     'REGIONS_RATING_INCOME_MUL_0',
                     'DAYS_CREDIT_MEAN_OVERALL',
                     'AMT_CREDIT_SUM_MEAN_OVERALL',
                     'CURRENT_DEBT_TO_CREDIT_RATIO_MEAN_OVERALL',
                     'DAYS_CREDIT_MEAN_CREDITACTIVE_ACTIVE',
                     'CURRENT_DEBT_TO_CREDIT_RATIO_MEAN_CREDITACTIVE_ACTIVE',
                     'CURRENT_CREDIT_DEBT_DIFF_MEAN_CREDITACTIVE_ACTIVE',
                     'HOUR_APPR_PROCESS_START_MEAN_LAST_5',
                     'DAYS_DECISION_MEAN_LAST_5',
                     'SELLERPLACE_AREA_MEAN_LAST_5',
                     'INTEREST_SHARE_MEAN_LAST_5',
                     'INTEREST_SHARE_MEAN_FIRST_2',
                     'DAYS_PAYMENT_DIFF_MEAN_MEAN',
                     'DAYS_PAYMENT_DIFF_MIN_MEAN',
                     'DAYS_PAYMENT_DIFF_MAX_MEAN']


# defining a route to get all clients data
@app.route("/data", methods=["GET"])
def get_data():
    df_all = X.to_dict("list")
    return jsonify(df_all)


# defining a route to get clients data and prediction
@app.route("/data/client/<client_id>", methods=["GET"])
def client_data(client_id):
    # filter the data thanks to the id from the request
    df_sample = X[X["SK_ID_CURR"] == int(client_id)]

    df_sample_fs = df_sample[selected_features]
    column_to_add = df_sample.pop("SK_ID_CURR")
    df_sample_fs.insert(0, "SK_ID_CURR", column_to_add)

    # standardizing the data
    scaler = StandardScaler()
    sample_scaled = scaler.fit_transform(df_sample_fs[selected_features])
    # replacing nan values with 0
    sample_scaled[np.isnan(sample_scaled)] = 0

    # calculate prediction and probability for this client
    df_sample_fs["prediction"] = model.predict(sample_scaled).tolist()[0]
    df_sample_fs['proba_1'] = model.predict_proba(sample_scaled)[:, 1].tolist()[0]

    # calculate features importance in this prediction
    explainer = shap.KernelExplainer(model.predict_proba, scaler.fit_transform(X[selected_features]))

    shap_values = explainer.shap_values(sample_scaled, check_additivity=False)

    # add the shap values in the dataframe
    df_sample_fs["expected"] = explainer.expected_value[1]
    new_line = [99999] + list(shap_values[1])[0].tolist() + [0, 0, explainer.expected_value[1]]
    df_sample_fs.loc[1] = new_line

    # create the dictionary to be sent
    sample = df_sample_fs.to_dict("list")
    # returning sample and prediction objects as json
    return jsonify(sample)


# To remove when online :
if __name__ == '__main__':
    app.run(host='0.0.0.0')
