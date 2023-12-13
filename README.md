# TSLA Stock Price and Trade Decision Prediction
* The project contains two parts. The first part is to predict daily close price of TSLA. The second part is to predict and classify the daily trade decisions (long, short or wait).


## Introduction and Background
* The project includes two metaflows training models to predict TSLA daily close price and trade decisions respectively.
* Common features include daily price metrics of TSLA and its technical indicators as well as values of SP500, VIX index and US30 Bond yield. In addition, daily tweets of TSLA are utilized to provide daily sentiment metrics (convert each tweet to sentiment labels) grouped by dates where the original dataset is from https://www.kaggle.com/datasets/equinxx/stock-tweets-for-sentiment-analysis-and-prediction
* The step of converting all the tweets collected to sentimengt labels is extremely tim consuming and has been skipped in the metaflow which will be introduced later. The output of the convertion is saved into `tsla_tweets_df.csv` in `src` folder which is the start point for data preprocessing.    
* For decision models, we introduce a simple strategy here: if the decision is "short", we short one share of TSLA at open price when the market opens and long it back at close price at the end of trade day; if the decision is "long", we long one share of TSLA at open price and short it reversely at close price. When putting the decision labels, A threshold of 0.5% is set here so that if current date's ratio of close price on open price is greater than 1.005 then label "long"; if the ratio smaller than 0.995 then label "short"; else then label "wait". LSTM models are used to fit the finally merged dataset.

## Requirements
This project requires Python 3 and other dependencies listed in `requirements.lock`. Setup the environment with `rye sync` and go into src; then quickly run the usual small_flow.py with Metaflow, to produce the model we will be serving (the script and the datasets are repeated for convenience, but nothing has changed from pre

## Structuring ML projects with Metaflow
* `src` is the folder containing all the scripts we need to run for this project.
* `pyproject.toml` holds the Python dependencies required for running the scripts (use `rye sync` as usual etc.).
* To run `project7773_meta_flow_price_model.py`, you need to use the Metaflow syntax `python project7773_meta_flow_price_model.py run` and `python project7773_meta_flow_price_model.py run`.
* Make sure that `tsla_tweets_df.csv` is in `src` folder.
* The running results are also pushed to Commet https://www.comet.com/nyu-fre-7773-2021/tz2657-group-project-decision-model/view/new/panels and https://www.comet.com/nyu-fre-7773-2021/tz2657-group-project-price-model/view/new/panels

## Deploy
* In `src` folder, `project7773_flask_app.py` is the "back-end" tier composed by a Flask app serving model predictions for TSLA close price. The app needs to call http://127.0.0.1:5000/predict?x1={x1}&x2={x2}&x3={x3} with x1 (date), x2 (previous day sentiment mean), x3 (previous day sentiment std) as inputs. You need to use  `python project7773_flask_app.py` to run this file in terminal.
* `project7773_streamlit.py` is the "front-end" tier, composed by a Streamlit app that accepts a numerical input from the user, performs a GET request to your Flask app with that input, and displays the prediction. You need to use `streamlit run project7773_streamlit.py` to run this file.

