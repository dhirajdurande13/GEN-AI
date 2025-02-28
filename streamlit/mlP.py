import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# load_iris
# his function loads the Iris dataset, a popular dataset in machine learning. It contains 150 samples of iris flowers with 4 features each (sepal length, sepal width, petal length, petal width) 
# It’s often used for testing and learning classification algorithms.
# 

# Random-Forest:-
# This is a machine learning model used for classification tasks. It builds multiple decision trees during training and combines their results to make accurate predictions.
import sklearn
print(sklearn.__version__)  # Check if it's installed correctly
# to load data in cache no need to load every time from library
st.cache_data
def load_data():
    iris=load_iris()
    df=pd.DataFrame(iris.data,columns=iris.feature_names)
    df['species']=iris.target
    return df, iris.target_names



df,target_name=load_data()
model=RandomForestClassifier()

model.fit(df.iloc[:,:-1],df['species'])
# exclude the last variable because it is dependent other are independent

sepal_length=st.slider('Sepal length',float(df['sepal length (cm)'].min()),float(df['sepal length (cm)'].max()))
sepal_width=st.slider('Sepal width',float(df['sepal width (cm)'].min()),float(df['sepal width (cm)'].max()))
petal_length=st.slider('petal length',float(df['petal length (cm)'].min()),float(df['petal length (cm)'].max()))
petal_width=st.slider('petal width',float(df['petal width (cm)'].min()),float(df['petal width (cm)'].max()))

input_data=[[sepal_length,sepal_width,petal_length,petal_width]]
# prediction
prediction=model.predict(input_data)

predicted_species=target_name[prediction[0]]

st.write(predicted_species)