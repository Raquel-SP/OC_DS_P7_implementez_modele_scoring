# API to load clients' data, predict the scoring, product shap values

# Import required libraries
import flask
from flask import jsonify
import pickle
import pandas as pd
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
model = pickle.load(open(path + "/lgbm_best_model.pkl", "rb"))
# getting the data
X_dashboard = pd.read_csv(path + "/data_dashboard.csv")
X_model = pd.read_csv(path + "/data_api_scaled.csv")


# defining a route to get all clients data
@app.route("/data", methods=["GET"])
def get_data():
    df_all = X_dashboard.to_dict("list")
    return jsonify(df_all)


# defining a route to get clients data and prediction
@app.route("/data/client/<client_id>", methods=["GET"])
def client_data(client_id):
    # filter the data thanks to the id from the request
    client_modeliced = X_model[X_model["SK_ID_CURR"] == int(client_id)]
    client_to_modelice = client_modeliced.drop(columns="SK_ID_CURR")

    # calculate prediction and probability for this client
    client_modeliced["prediction"] = model.predict(client_to_modelice).tolist()[0]
    client_modeliced['proba_1'] = model.predict_proba(client_to_modelice)[:, 1]\
        .tolist()[0]

    # calculate features importance in this prediction
    explainer = shap.KernelExplainer(model.predict_proba, client_to_modelice)

    shap_values = explainer.shap_values(client_to_modelice, check_additivity=False)

    # add the shap values in the dataframe
    client_modeliced["expected"] = explainer.expected_value[1]
    new_line = [99999] + list(shap_values[1])[0].tolist()\
        + [0, 0, explainer.expected_value[1]]
    client_modeliced.loc[1] = new_line

    # create the dictionary to be sent
    sample = client_modeliced.to_dict("list")
    # returning sample and prediction objects as json
    return jsonify(sample)


# To remove when online :
if __name__ == '__main__':
    app.run(host='0.0.0.0')
