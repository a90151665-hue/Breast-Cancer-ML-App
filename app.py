import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer


model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

data = load_breast_cancer()
feature_names = data.feature_names

st.set_page_config(page_title="Breast cancer prediction", layout="wide")

st.title("Breast Cancer Prediction App")
st.write("Entre feature values to predict wheather the tumor is **Benign or Maligant**.")


input_data = []

for feature in feature_names:
    value = st.sidebar.number_input(feature, min_value=0.0, value=1.0)
    input_data.append(value)

input_array = np.array(input_data).reshape(1, -1)


if st.button("Predict"):
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)
    probability = model.predict_proba(scaled_input)

    if prediction[0] == 1:
        st.success(" Prediction: Benign")
    else:
        st.error(" Prediction:Malignant")

    st.write("Prediction Probability:")
    st.write(probability)    