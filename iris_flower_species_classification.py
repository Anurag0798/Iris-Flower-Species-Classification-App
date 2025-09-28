import pandas as pd
import numpy as np

import joblib

import streamlit as st

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://anurag:07121998@cluster0.ugo9l.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['Iris_Flower_Species']     # Create a new database
collection = db['Iris_Flower_Species_Classification']     # Create a new collection/table in the database

def load_models():
    logistic_reg_binary_scalar_data = joblib.load('logistics_binary_scalar.joblib')
    log_reg_binary_model = logistic_reg_binary_scalar_data['model']
    scalar_binary = logistic_reg_binary_scalar_data['scaler']
    
    logistic_reg_ovr_multiclass_scalar_data = joblib.load('logistics_ovr_multi_scalar.joblib')
    log_reg_ovr_multiclass_model = logistic_reg_ovr_multiclass_scalar_data['model']
    scalar_multiclass = logistic_reg_ovr_multiclass_scalar_data['scaler']
    
    logistic_reg_multinomial_multiclass_scalar_data = joblib.load('logistics_multinomial_multi_scalar.joblib')
    log_reg_multinomial_multiclass_model = logistic_reg_multinomial_multiclass_scalar_data['model']
    
    svc_binary_scalar_data = joblib.load('svm(c)_binary_scalar.joblib')
    svc_binary_model = svc_binary_scalar_data['model']
    svc_binary_scalar = svc_binary_scalar_data['scaler']

    svc_multiclass_scalar_data = joblib.load('svm(c)_multi_scalar.joblib')
    svc_multiclass_model = svc_multiclass_scalar_data['model']
    svc_multiclass_scalar = svc_multiclass_scalar_data['scaler']

    decision_tree_classifier_multiclass_scalar_data = joblib.load('decision_tree(c)_multiclass_scalar.joblib')
    decision_tree_classifier_multiclass_model = decision_tree_classifier_multiclass_scalar_data['model']
    decision_tree_classifier_multiclass_scalar = decision_tree_classifier_multiclass_scalar_data['scaler']

    random_forest_classifier_multiclass_scalar_data = joblib.load('random_forest(c)_multiclass_scalar.joblib')
    random_forest_classifier_multiclass_model = random_forest_classifier_multiclass_scalar_data['model']
    random_forest_classifier_multiclass_scalar = random_forest_classifier_multiclass_scalar_data['scaler']

    return [log_reg_binary_model, log_reg_ovr_multiclass_model, log_reg_multinomial_multiclass_model, 
            svc_binary_model, svc_multiclass_model, 
            decision_tree_classifier_multiclass_model,
            random_forest_classifier_multiclass_model,
            scalar_binary, scalar_multiclass, svc_binary_scalar, svc_multiclass_scalar, 
            decision_tree_classifier_multiclass_scalar, random_forest_classifier_multiclass_scalar]



def preprocessing_input_data(data, scalar):
    df = pd.DataFrame([data])

    df_scaled = scalar.transform(df)

    return df_scaled



def predict_data(data, model, scalar):
    df_scaled = preprocessing_input_data(data, scalar)

    return model.predict(df_scaled)



def main():
    st.title("Iris Flower Species Classification App")
    st.write("Enter the data below to get a prediction for the Iris Flower Species")

    [log_reg_binary_model, log_reg_ovr_multiclass_model, log_reg_multinomial_multiclass_model, 
     svc_binary_model, svc_multiclass_model,
     decision_tree_classifier_multiclass_model, 
     random_forest_classifier_multiclass_model,
     scalar_binary, scalar_multiclass, svc_binary_scalar, svc_multiclass_scalar,
     decision_tree_classifier_multiclass_scalar, random_forest_classifier_multiclass_scalar] = load_models()

    sepal_length = st.slider("Sepal Length (cm)", 0.0, 10.0, 5.0)
    sepal_width = st.slider("Sepal Width (cm)", 0.0, 10.0, 5.0)
    petal_length = st.slider("Petal Length (cm)", 0.0, 10.0, 5.0)
    petal_width = st.slider("Petal Width (cm)", 0.0, 10.0, 5.0)

    model_options = [
        "Logistic Regression Binary Classification",
        "Logistic Regression OVR Multiclass Classification",
        "Logistic Regression Multinomial Multiclass Classification",
        "SVM Binary Classification",
        "SVM Multiclass Classification",
        "Decision Tree Classification",
        "Random Forest Classification"
    ]

    selected_model = st.selectbox("Select a model to predict the specie", model_options)

    if st.button("Classify the species"):
        user_data = {
            "sepal length (cm)": sepal_length,
            "sepal width (cm)": sepal_width,
            "petal length (cm)": petal_length,
            "petal width (cm)": petal_width
        }

        if selected_model == "Logistic Regression Binary Classification":
            prediction = predict_data(user_data, log_reg_binary_model, scalar_binary)
        elif selected_model == "Logistic Regression OVR Multiclass Classification":
            prediction = predict_data(user_data, log_reg_ovr_multiclass_model, scalar_multiclass)
        elif selected_model == "Logistic Regression Multinomial Multiclass Classification":
            prediction = predict_data(user_data, log_reg_multinomial_multiclass_model, scalar_multiclass)
        elif selected_model == "SVM Binary Classification":
            prediction = predict_data(user_data, svc_binary_model, svc_binary_scalar)
        elif selected_model == "SVM Multiclass Classification":
            prediction = predict_data(user_data, svc_multiclass_model, svc_multiclass_scalar)
        elif selected_model == "Decision Tree Classification":
            prediction = predict_data(user_data, decision_tree_classifier_multiclass_model, decision_tree_classifier_multiclass_scalar)
        elif selected_model == "Random Forest Classification":
            prediction = predict_data(user_data, random_forest_classifier_multiclass_model, random_forest_classifier_multiclass_scalar)

        if prediction == 0:
            prediction = "Setosa"
        elif prediction == 1:
            prediction = "Versicolor"
        elif prediction == 2:
            prediction = "Virginica"

        st.success(f"The specie of the flower is {prediction} using the {selected_model} classifier")

        user_data["prediction"] = prediction     # Add the ridge prediction to the user_data dictionary
        collection.insert_one(user_data)     # Insert the user_data dictionary as a record to the MongoDB collection



if __name__ == "__main__":
    main()