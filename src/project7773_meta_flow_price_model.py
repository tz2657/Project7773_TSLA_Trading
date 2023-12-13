import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime
from comet_ml import Experiment
from metaflow import FlowSpec, step, IncludeFile, Parameter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from datetime import timedelta
from sklearn.metrics import r2_score

class Price_LSTMModelFlow(FlowSpec):
    comet_api_key = 'bWPiZX9BSQwryTCmMQzGYDg6d'
    comet_project_name = 'tz2657_group_project_price_model'
    comet_workspace = 'nyu-fre-7773-2021'
    
    @step
    def start(self):
        self.next(self.get_data)

    @step
    def get_data(self):
        ## read tsla_tweets_df
        tlsa_tweets_df_file_path='./tsla_tweets_df.csv'
        tsla_tweets_df = pd.read_csv(tlsa_tweets_df_file_path)
        tsla_tweets_df['Date'] = pd.to_datetime(tsla_tweets_df['Date']).dt.date
        
        ticker_symbol = "TSLA"
        start_date1 = datetime.strptime("2021-08-01", '%Y-%m-%d').date()
        start_date = datetime.strptime("2021-09-29", '%Y-%m-%d').date()
        end_date = datetime.strptime("2022-10-01", '%Y-%m-%d').date()
        
        tsla_data = yf.Ticker(ticker_symbol)
        tsla_df = tsla_data.history(start=start_date1, end=end_date)
        tsla_df.reset_index(inplace=True)
        tsla_df['Date'] = pd.to_datetime(tsla_df['Date']).dt.date
        
        # get the technical indicators metrics
        tsla_df['daily_return'] = tsla_df['Close'].pct_change()
        tsla_df['return_volatility'] = tsla_df['daily_return'].rolling(window=20).std() * (252 ** 0.5)  # 年化波动率
        tsla_df['MA20'] = tsla_df['Close'].rolling(window=20).mean()
        tsla_df['20SD'] = tsla_df['Close'].rolling(window=20).std()
        tsla_df['upper_band'] = tsla_df['MA20'] + (tsla_df['20SD'] * 2)
        tsla_df['lower_band'] = tsla_df['MA20'] - (tsla_df['20SD'] * 2)
        exp1 = tsla_df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = tsla_df['Close'].ewm(span=26, adjust=False).mean()
        tsla_df['MACD'] = exp1 - exp2
        tsla_df['EMA'] = tsla_df['Close'].ewm(span=20, adjust=False).mean()
        tsla_df['logmomentum'] = np.log(tsla_df['Close'] / tsla_df['Close'].shift(1))
        
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
        # Calculate the weighted KDJ feature
        tsla_df['KDJ_score'] = (tsla_df['K'] * weight_K) + (tsla_df['D'] * weight_D) + (tsla_df['J'] * weight_J)
        tsla_df = tsla_df[(tsla_df['Date'] >= start_date) & (tsla_df['Date'] <= end_date)]
        tsla_df = tsla_df.reset_index(drop=True)
        
        ## For the tweets date that are not in trade days, we artficially change the tweets date to its respective next trade days.
        unique_tweet_dates = set(tsla_tweets_df['Date']) - set(tsla_df['Date'])
        next_trading_day = {}
        sorted_trading_days = sorted(list(set(tsla_df['Date'])))
        for non_trading_day in unique_tweet_dates:
            for trading_day in sorted_trading_days:
                if trading_day > non_trading_day:
                    next_trading_day[non_trading_day] = trading_day
                    break
        # update the dates in grouped_tsla_tweets_df
        tsla_tweets_df['Date'] = tsla_tweets_df['Date'].apply(lambda x: next_trading_day.get(x, x))
        
        
        # 'POSITIVE' -> 1, 'NEGATIVE' -> -1, 'NEUTRAL' -> 0
        grouped_tsla_tweets_df = tsla_tweets_df.groupby('Date')['sentiment'].agg(['mean', 'std'])
        
        grouped_tsla_tweets_df.rename(columns={'mean': 'sentiment_mean', 'std': 'sentiment_std'}, inplace=True)
        grouped_tsla_tweets_df.reset_index(inplace=True)
        grouped_tsla_tweets_df['Date'] = pd.to_datetime(grouped_tsla_tweets_df['Date']).dt.date
        
        # get vix, sp500, us30 data
        vix = yf.Ticker("^VIX")
        vix_df = vix.history(start=start_date, end=end_date)
        vix_df.reset_index(inplace=True)
        vix_df['Date'] = pd.to_datetime(vix_df['Date']).dt.date
        sp500 = yf.Ticker("^GSPC")
        sp500_df = sp500.history(start=start_date, end=end_date)
        sp500_df.reset_index(inplace=True)
        sp500_df['Date'] = pd.to_datetime(vix_df['Date']).dt.date
        us30 = yf.Ticker("^TYX")
        us30_df = us30.history(start=start_date, end=end_date)
        us30_df.reset_index(inplace=True)
        us30_df['Date'] = pd.to_datetime(vix_df['Date']).dt.date
        
        # rename columns by adding preffix
        tsla_df.columns = ['tsla_' + col if col != 'Date' else col for col in tsla_df.columns]
        vix_df.columns = ['vix_' + col if col != 'Date' else col for col in vix_df.columns]
        sp500_df.columns = ['sp500_' + col if col != 'Date' else col for col in sp500_df.columns]
        us30_df.columns = ['us30_' + col if col != 'Date' else col for col in us30_df.columns]

        merged_df = pd.merge(tsla_df, grouped_tsla_tweets_df, on='Date', how='left')
        merged_df = pd.merge(merged_df, vix_df, on='Date', how='inner')
        merged_df = pd.merge(merged_df, sp500_df, on='Date', how='inner')
        merged_df = pd.merge(merged_df, us30_df, on='Date', how='inner')
        

        ## shift Dates such that we can use previous day's metrics to predict the next day's price.
        merged_df['Date'] = merged_df['Date'].shift(-1)
        merged_df.columns = [col + '_preday' if col != 'Date' else col for col in merged_df.columns]
        merged_df = merged_df.dropna()
        ## to add today close price for each date
        merged_df = pd.merge(merged_df, tsla_df, on='Date', how='inner')





        ## derive correct decisions: buy, sell or wait
        ## if next day's close price > open price, then buy at next day's open price; if next day's close price < open price, then short at next day's open price;
        ## if next day's close price and open price are similar (0.995), then wait
        def calculate_correct_decision(tsla_df):
            decisions = []
            for i in range(len(tsla_df)):
                #today_close = tsla_df.iloc[i]['tsla_Close']
                today_day_open = tsla_df.iloc[i]['tsla_Open']
                today_day_close = tsla_df.iloc[i]['tsla_Close']

                price_change = (today_day_close - today_day_open) / today_day_open
                threshold = 0.005  # 0.5%

                if price_change > threshold:
                    decisions.append('Long')
                elif price_change < -threshold:
                    decisions.append('Short')
                else:
                    decisions.append('Wait')
            return decisions

        merged_df['correct_decision'] = calculate_correct_decision(merged_df)
        
        selected_columns = [
            'Date',
            'tsla_daily_return_preday','tsla_return_volatility_preday','tsla_Open_preday', 'tsla_High_preday', 'tsla_Low_preday', 'tsla_Volume_preday', 
            'tsla_MA20_preday', 'tsla_MACD_preday', 'tsla_KDJ_score_preday', 'tsla_upper_band_preday', 'tsla_lower_band_preday', 'tsla_EMA_preday', 'tsla_logmomentum_preday', 
            'sentiment_mean_preday', 'sentiment_std_preday', 
            'tsla_Close_preday',
            'vix_Close_preday', 
            'sp500_Close_preday', 
            'us30_Close_preday',
            'tsla_Open',
            'tsla_Close', ##########
            'correct_decision']
        merged_df['Date'] = pd.to_datetime(merged_df['Date']).dt.date
        self.merged_df = merged_df[selected_columns]
        #merged_df = merged_df.dropna()
        output_merged_df_path = './merged_df.csv'
        self.merged_df.to_csv(output_merged_df_path, index=False)
        self.tsla_df = tsla_df
        self.next(self.prepare_data)


    @step
    def prepare_data(self):
        df = pd.read_csv('./merged_df.csv')
        df.sort_values('Date', inplace=True)
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        
        # select features
        self.features = ['tsla_daily_return_preday','tsla_return_volatility_preday','tsla_Open_preday', 'tsla_High_preday', 'tsla_Low_preday', 'tsla_Volume_preday', 
                    'tsla_MA20_preday', 'tsla_MACD_preday', 'tsla_KDJ_score_preday', 'tsla_upper_band_preday', 'tsla_lower_band_preday', 'tsla_EMA_preday', 'tsla_logmomentum_preday', 
                    'sentiment_mean_preday', 'sentiment_std_preday', 
                    'tsla_Close_preday',
                    'vix_Close_preday', 
                    'sp500_Close_preday', 
                    'us30_Close_preday']
        target = 'tsla_Close'
        
        # scalize
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_df = scaler.fit_transform(df[self.features + [target]])
        self.scaler = scaler
        
        # defin train, validation, test set starting dates
        train_size = int(len(df) * 0.60)
        val_start = train_size + 20
        val_size = train_size + 20 + 25
        test_start = val_size + 20
        self.test_start = test_start
        
        # split train, validation, test sets
        self.train_data = scaled_df[:train_size]
        self.val_data = scaled_df[val_start:val_size]
        self.test_data = scaled_df[test_start:]
        self.df = df
        self.next(self.train_model)

    @step
    def train_model(self):
        X_train, y_train = self.train_data[:, :-1], self.train_data[:, -1]
        X_val, y_val = self.val_data[:, :-1], self.val_data[:, -1]
        X_test, y_test = self.test_data[:, :-1], self.test_data[:, -1]
        
        # reshape into [samples, timesteps, features] for LSTM input requirements
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
        self.X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        self.y_test = y_test
        
        # build LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), verbose=0)
        self.model = model
        self.next(self.evaluate_model)

    @step
    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        y_pred_inv = self.scaler.inverse_transform(np.concatenate((self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[2]), y_pred), axis=1))[:, -1]
        y_test_inv = self.scaler.inverse_transform(np.concatenate((self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[2]), self.y_test.reshape(-1, 1)), axis=1))[:, -1]
        
        mse = mean_squared_error(y_test_inv, y_pred_inv)
        print('Mean Squared Error:', mse)
        # Calculate R² score
        r_squared = r2_score(y_test_inv, y_pred_inv)
        print('R² Score:', r_squared)

        # plot actual price and predicted price
        plt.figure(figsize=(10, 6))
        plt.plot(self.df['Date'][self.test_start:], y_test_inv, label='Actual Price', color='blue')
        plt.plot(self.df['Date'][self.test_start:], y_pred_inv, label='Predicted Price', color='red')
        plt.title('Actual vs Predicted Stock Prices')
        plt.xlabel('Date Index')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
        
        ## Commet Connection:
        exp = Experiment(project_name = self.comet_project_name,
                         api_key = self.comet_api_key,
                         workspace = self.comet_workspace,
                         auto_param_logging=False)
        results = {'R^2 Score': r_squared,
                   'Mean Squared Error': mse}
        # Log the results
        exp.log_metrics(results)
        exp.end()
        
        self.next(self.end)

    @step
    def end(self):
        print("Price prediction model flow completed!")

if __name__ == '__main__':
    Price_LSTMModelFlow()







