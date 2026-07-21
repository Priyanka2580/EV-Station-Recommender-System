import pandas as pd

# Decision threshold for flagging a station as high-utilization.
# Tuned down from the classifier's default of 0.5: at 0.5 the model almost
# never predicts the minority "high utilization" class (recall ~0.09); 0.3
# trades some precision for much better recall (~0.82) on that class.
HIGH_UTILIZATION_THRESHOLD = 0.3

def predict_high_utilization(df, peak_model, day_of_week, hour_of_day):

    # Uses the pre-trained model (LGBMClassifier) to predict the probability of high utilization (high_utilization).
    if df.empty:
        return df

    df_copy = df.copy()

    # Broadcast user-supplied spatial-temporal context
    df_copy['day_of_week'] = day_of_week
    df_copy['hour_of_day'] = hour_of_day

    # Features required by Model B
    features = [
        'hour_of_day', 'day_of_week', 'traffic_congestion_index', 'is_peak_hour'
    ]

    # Predict probability of high utilization (class 1)
    df_copy['high_utilization_probability'] = peak_model.predict_proba(df_copy[features])[:, 1]
    df_copy['high_utilization_flag'] = (df_copy['high_utilization_probability'] >= HIGH_UTILIZATION_THRESHOLD).astype(int)

    return df_copy

