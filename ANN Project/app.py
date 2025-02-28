import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import pickle
import streamlit as st

# Load the model
# tf.keras.backend.clear_session()
model=tf.keras.models.load_model('model.h5',compile=False)

# load encoder and scalar
import tensorflow as tf
st.write(f"TensorFlow Version: {tf.__version__}")



# try:
#     model = tf.keras.models.load_model('model.h5', compile=False)
#     st.success("Model Loaded Successfully!")
# except Exception as e:
#     st.error(f"Error loading model: {e}")


with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

    # for scalar
with open('scalar.pkl','rb') as file:
    scaler=pickle.load(file)
    # for geo
with open('one_ht_enc_geo.pkl','rb') as file:
    label_encoder_geo=pickle.load(file)


# checking errors
# if model is None:
#     st.error("Model not loaded properly!")

# if label_encoder_gender is None:
#     st.error("Gender encoder not loaded!")

# if label_encoder_geo is None:
#     st.error("Geography encoder not loaded!")

# if scaler is None:
#     st.error("Scaler not loaded!")

# if "model" not in st.session_state:
#     tf.keras.backend.clear_session()
#     model = tf.keras.models.load_model('model.h5', compile=False)

# # Load encoders and scaler once
# if "label_encoder_gender" not in st.session_state:
#     with open('label_encoder_gender.pkl', 'rb') as file:
#         label_encoder_gender = pickle.load(file)

# if "label_encoder_geo" not in st.session_state:
#     with open('one_ht_enc_geo.pkl', 'rb') as file:
#         label_encoder_geo = pickle.load(file)

# if "scaler" not in st.session_state:
#     with open('scalar.pkl', 'rb') as file:
#         scaler = pickle.load(file)

# Streamlit app

st.title("Customer CHURN Prediction..")

# input
geography = st.selectbox("Geography",label_encoder_geo.categories_[0])
credit_score = st.number_input("Credit Score")

gender = st.selectbox("Gender",label_encoder_gender.classes_)
age = st.slider("Age",18,92)
tenure = st.slider("Tenure (Years with Bank)",0,10)
balance = st.number_input("Balance")
num_of_products = st.slider("Number of Products",1,4)
has_credit_card = st.selectbox("Has Credit Card?", [0, 1])
is_active_member = st.selectbox("Is Active Member?", [0, 1])
estimated_salary = st.number_input("Estimated Salary")

input_data=pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_credit_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})

geo_encoded=label_encoder_geo.transform([[geography]]).toarray()
# geo_encoded=label_encoder_geo.transform([[geography]]).reshape(1,-1)
geo_encoder_df=pd.DataFrame(geo_encoded,columns=label_encoder_geo.get_feature_names_out(['Geography']))

# combine data
input_df=pd.concat([input_data.reset_index(drop=True),geo_encoder_df.reset_index(drop=True)],axis=1) #Drop true if old columns present remove and reset with new indexes
input_df_scaled = scaler.transform(input_df)  # Scale input data


# prediction churn

prediction=model.predict(input_df_scaled)
prediction_prob=prediction[0][0]
st.write(f'Churn Probablity: {prediction_prob:.2f}')
if prediction_prob>0.5:
    st.write("Customer likely to CHURN")
else:
    st.write("Customer Not likely to CHURN")