import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

header= st.container()

dataset= st.container()

features= st.container()

modeltraining = st.container()

@st.cache_data #Caching the data
def get_data(filename):
    taxi_data= pd.read_csv('data/taxi_data.csv')
    return taxi_data


with header:
    st.title("Welcome to my First Data Science Project")


with dataset:
    st.header("NYC Taxi Dataset ")
    st.text("I found this dataset on xyx.com")
    taxi_data= get_data('data/taxi_data.csv')
    st.write(taxi_data.head())

    st.subheader("Pickup location distribution from the NYC Dataset")
    pulocation_dist=pd.DataFrame(taxi_data["PULocationID"].value_counts()).head(50)
    st.bar_chart(pulocation_dist)



with features:
    st.header("The features I created  ")
    st.markdown("* **First Feature: ** I created this feature becuase of this.... I calculated it using this logic....")
    st.markdown("* **Second Feature: ** I created this feature becuase of this.... I calculated it using this logic....")



with modeltraining:
    st.header("Time to train the model")
    st.text("Here you choose the hyperparameters of the model and see how the performance changes")

    sel_col,disp_col=st.columns(2)

    max_depth=sel_col.slider("What should be the next step of the model?",min_value=10,max_value=100,value=20,step=10)

    n_estimators=sel_col.selectbox("How many trees should there be?",options=[100,200,300],index=0)
    sel_col.text("Here's a list of my features: ")
    sel_col.write(taxi_data.columns)

    if n_estimators== "No limit":
        regr = RandomForestRegressor(max_depth=max_depth)
    else:
        regr=RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators)


    input_feature=sel_col.text_input("Which feature should be used as the input feature?",'trip_miles')
    
    X=taxi_data[[input_feature]]
    y=taxi_data[['trip_miles']]

    regr.fit(X,y)

    prediction=regr.predict(y)
    disp_col.subheader("Mean absolure error of the model is: ")
    disp_col.write(mean_absolute_error(y,prediction))
    disp_col.subheader("Mean squared error of the model is: ")
    disp_col.write(mean_squared_error(y,prediction))
    disp_col.subheader("R squared score of the model is: ")
    disp_col.write(r2_score(y,prediction))


   
