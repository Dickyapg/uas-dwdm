import streamlit as st
import pandas as pd
import numpy as np
import pickle  # to load a saved model


app_mode = st.sidebar.selectbox(
    'Select Page', ['Home', 'Prediction'])  # two pages

if app_mode == 'Home':
    st.title('Iris')
    st.markdown('Dataset :')
    data = pd.read_csv('iris.csv')
    st.write(data.head())

elif app_mode == 'Prediction':
    st.sidebar.header("Informations about the client :")
    sepalLengthCM = st.sidebar.slider('Sepal Length', 0.0,10.0, 1.0)
    sepalWidthCM = st.sidebar.slider('Sepal Width', 0.0,10.0, 1.0)
    petalLengthCM = st.sidebar.slider('Petal Length', 0.0,10.0, 1.0)
    petalWidthCM = st.sidebar.slider('Petal Width', 0.0,10.0, 1.0)
    data = {
        'SepalLengthCm': sepalLengthCM,
        'SepalWidthCm': sepalWidthCM,
        'PetalLengthCm': petalLengthCM,
        'PetalWidthCm': petalWidthCM,
    }
    

    if st.button("Click to Predict"):
        feature_list = [sepalLengthCM,sepalWidthCM,petalLengthCM,petalWidthCM,]            
        single_sample = np.array(feature_list).reshape(1, -1)
        loaded_model = pickle.load(open('model.pkl', "rb"))
        prediction = loaded_model.predict(single_sample)
        st.markdown("Hasil prediksinya adalah ")
        st.subheader(prediction[0])
