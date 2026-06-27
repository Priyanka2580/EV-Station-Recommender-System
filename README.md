# EV Charging Station Recommender

A  machine-learning pipeline that recommends the best EV charging stations for an user based on location, hardware compatibility, predicted wait time, utilization, reliability, session duration, and cost. The pipeline is exposed through an interactive Streamlit app.

## How it works

The recommender runs as a sequential pipeline that narrows down and scores candidate stations step by step. It starts by filtering stations to the user's chosen city and state, then filters by connector type and minimum power output before scoring hardware fit. From there, machine learning models predict the likelihood of a wait and the expected utilization at the requested day and time, while a rule-based score captures station reliability, and further models estimate session duration and dynamic pricing to arrive at a total cost. All these scores are normalized and combined into a single weighted score, with the top 5 stations returned as recommendations. The underlying dataset has around 1.3 million time-stamped station-status records covering location, hardware specs, live status, pricing, and contextual signals like traffic and peak hours.

## Running the app

Launch the interactive Streamlit app and pick a city, charger type, minimum power, preferred network, and day/time in the sidebar, then click *Find Best Stations* to see the top 5 ranked recommendations with cost, duration, and match-score breakdowns. A one-shot script is also available for quick debugging or scripted runs, printing the ranked results straight to the console based on a small set of query parameters.

## Features

- Sequential, multi-stage pipeline from regional filtering to hybrid ranking.
- Trained ML models covering queueing, utilization, session duration, and dynamic pricing.
- Algorithm selection chosen by comparison across XGBoost, LightGBM, and Random Forest.
- Leakage-free train/validation/test methodology with explicit overfitting checks.
- Self-contained, reusable scikit-learn pipelines for preprocessing and inference.
- Interactive Streamlit front-end with live progress feedback and ranked result cards.
