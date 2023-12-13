from flask import Flask, request, jsonify
import uuid
import time
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model


model_filename = './price_model.h5'
latest_model = load_model(model_filename)

# We initialise the Flask object to run the flask app
app = Flask(__name__)


@app.route('/predict', methods=['GET'])
def main():
    """
    This function runs when the endpoint /predict is hit with a GET request.
      
    It looks for an input x value, runs the model and returns the prediction.
    """
    start = time.time()
    cur_trade_date = request.args.get('x1')
    cur_trade_date = datetime.strptime(cur_trade_date , '%Y-%m-%d').date()
    sentiment_mean_preday = float(request.args.get('x2'))
    sentiment_std_preday = float(request.args.get('x3'))
    start_date = datetime.strptime("2021-09-29", '%Y-%m-%d').date()
    end_date = cur_trade_date
    end_date += timedelta(days=1)
    # Define the data for the DataFrame
    temp_data = {
        'Date': [cur_trade_date],
        'sentiment_mean_preday': [sentiment_mean_preday],
        'sentiment_std_preday': [sentiment_std_preday]
    }
    
    # Create the DataFrame
    grouped_tsla_tweets_df = pd.DataFrame(temp_data)
    
    # Historical data of Tesla from Yahoo Finance
    ticker_symbol='TSLA'
    tsla_data = yf.Ticker(ticker_symbol)
    tsla_df = tsla_data.history(start=start_date, end=end_date)
    tsla_df.reset_index(inplace=True)
    tsla_df['Date'] = pd.to_datetime(tsla_df['Date']).dt.date

    # calculate technical index
    tsla_df['daily_return'] = tsla_df['Close'].pct_change()
    tsla_df['return_volatility'] = tsla_df['daily_return'].rolling(window=20).std() * (252 ** 0.5)  # 年化波动率
    tsla_df['MA20'] = tsla_df['Close'].rolling(window=20).mean()
    tsla_df['20SD'] = tsla_df['Close'].rolling(window=20).std()
    tsla_df['upper_band'] = tsla_df['MA20'] + (tsla_df['20SD'] * 2)
    tsla_df['lower_band'] = tsla_df['MA20'] - (tsla_df['20SD'] * 2)
    # MACD
    exp1 = tsla_df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = tsla_df['Close'].ewm(span=26, adjust=False).mean()
    tsla_df['MACD'] = exp1 - exp2
    # EMA
    tsla_df['EMA'] = tsla_df['Close'].ewm(span=20, adjust=False).mean()
    # Logmomentum
    tsla_df['logmomentum'] = np.log(tsla_df['Close'] / tsla_df['Close'].shift(1))
    # KDJ
    low_list = tsla_df['Low'].rolling(9, min_periods=9).min()
    low_list.fillna(value = tsla_df['Low'].expanding().min(), inplace = True)
    high_list = tsla_df['High'].rolling(9, min_periods=9).max()
    high_list.fillna(value = tsla_df['High'].expanding().max(), inplace = True)
    rsv = (tsla_df['Close'] - low_list) / (high_list - low_list) * 100
    tsla_df['K'] = pd.DataFrame(rsv).ewm(com=2).mean()
    tsla_df['D'] = tsla_df['K'].ewm(com=2).mean()
    tsla_df['J'] = 3 * tsla_df['K'] - 2 * tsla_df['D']
    # Assign weights to K, D, and J. These weights can be adjusted based on your analysis needs.
    weight_K = 0.4
    weight_D = 0.3
    weight_J = 0.3
    tsla_df['KDJ_score'] = (tsla_df['K'] * weight_K) + (tsla_df['D'] * weight_D) + (tsla_df['J'] * weight_J)

    # VIX Inex
    vix = yf.Ticker("^VIX")
    vix_df = vix.history(start=start_date, end=end_date)
    vix_df.reset_index(inplace=True)
    vix_df['Date'] = pd.to_datetime(vix_df['Date']).dt.date

    ## SP500 data
    sp500 = yf.Ticker("^GSPC")
    sp500_df = sp500.history(start=start_date, end=end_date)
    sp500_df.reset_index(inplace=True)
    sp500_df['Date'] = pd.to_datetime(vix_df['Date']).dt.date

    #U.S. 30 Year Treasury rate data
    us30 = yf.Ticker("^TYX")
    us30_df = us30.history(start=start_date, end=end_date)
    us30_df.reset_index(inplace=True)
    us30_df['Date'] = pd.to_datetime(vix_df['Date']).dt.date

    # rename tsla_df columns
    tsla_df.columns = ['tsla_' + col if col != 'Date' else col for col in tsla_df.columns]
    vix_df.columns = ['vix_' + col if col != 'Date' else col for col in vix_df.columns]
    sp500_df.columns = ['sp500_' + col if col != 'Date' else col for col in sp500_df.columns]
    us30_df.columns = ['us30_' + col if col != 'Date' else col for col in us30_df.columns]

    # merge data
    merged_df = pd.merge(tsla_df, vix_df, on='Date', how='inner')
    merged_df = pd.merge(merged_df, sp500_df, on='Date', how='inner')
    merged_df = pd.merge(merged_df, us30_df, on='Date', how='inner')

    ## shift Dates such that we can use previous day's metrics to predict the next day's price.
    merged_df['Date'] = merged_df['Date'].shift(-1)
    merged_df.columns = [col + '_preday' if col != 'Date' else col for col in merged_df.columns]
    merged_df = pd.merge(merged_df, grouped_tsla_tweets_df, on='Date', how='inner')
    merged_df = merged_df.dropna()
    merged_df.reset_index(inplace=True)
    
    selected_columns = [
        'Date',
        'tsla_daily_return_preday','tsla_return_volatility_preday','tsla_Open_preday', 'tsla_High_preday', 'tsla_Low_preday', 'tsla_Volume_preday', 
        'tsla_MA20_preday', 'tsla_MACD_preday', 'tsla_KDJ_score_preday', 'tsla_upper_band_preday', 'tsla_lower_band_preday', 'tsla_EMA_preday', 'tsla_logmomentum_preday', 
        'sentiment_mean_preday', 'sentiment_std_preday', 
        'tsla_Close_preday',
        'vix_Close_preday', 
        'sp500_Close_preday', 
        'us30_Close_preday'
        ]
    merged_df['Date'] = pd.to_datetime(merged_df['Date']).dt.date
    merged_df = merged_df[selected_columns]
    merged_df.sort_values('Date', inplace=True)
    last_row_list = merged_df.iloc[-1].tolist()
    #inputs = np.array(last_row_list[1:])
    inputs = np.array(last_row_list[1:]).reshape(1, -1)
    inputs = np.expand_dims(inputs, axis=1)
        
    ## predict:
    val = latest_model.predict(inputs)
    for i in range(len(val)):
        ymax = 409.9700012207031
        ymin = 209.3866729736328
        ystd = (val[i][0] - 0) / (1 - 0) 
        val[i][0] = ymin + ystd * (ymax - ymin)
        
    # returning the response to the client	
    response = {
      'metadata': {
         'eventId': str(uuid.uuid4()),
         'serverTimestamp':round(time.time() * 1000), # epoch time in ms
         'serverProcessingTime': round((time.time() - start) * 1000) # in ms
      },
      'data': [val[0].tolist()]
    }
    
    return jsonify(response)
      

if __name__=='__main__':
  # Run the Flask app to run the server
  app.run(debug=True)

