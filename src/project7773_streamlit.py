import streamlit as st
import requests

st.title('TSLA Daily Close Price Prediction App')

# user input
x1_input = st.text_input('Enter current trade date (e.g. 2022-08-22):', value='2022-08-22')
x2_input = st.number_input('Enter previous day sentiment mean value:', value=0.3)
x3_input = st.number_input('Enter previous day sentiment std value:', value=0.5)

# get model prediction from Flask server
def get_prediction(x1, x2, x3):
    url = f'http://127.0.0.1:5000/predict?x1={x1}&x2={x2}&x3={x3}'
    response = requests.get(url)
    if response.status_code == 200:
        prediction = response.json()['data'][0]
        return prediction
    else:
        return None

# Predict
if st.button('Predict'):
    prediction = get_prediction(x1_input, x2_input, x3_input)
    if prediction is not None:
        st.success(f'Predicted value: {prediction}')
    else:
        st.error('Error in prediction')

## Conclusion:
'''
We only need the date input to retrieve the trading data of TSLA, SP500, VIX, US30, etc, as well as the technical indicators including MACD, KDJ through some steps of calculation. 
However, sentiment data cannot be easily derived automatically as the trading data, so we need two inputs here for sentiment metrics separately.
'''