import pandas as pd

def predict_wait_time(df, wait_model):
    """
    Uses the pre-trained wait_model (XGBClassifier) 
    to predict the probability of a queue (has_wait).
    Model A is refined to use: ports_out_of_service, utilization_rate, traffic_congestion_index.
    """
    df_copy = df.copy()
    
    if df_copy.empty:
        return df_copy
        
    # Select features required by refined Model A
    features = ['ports_out_of_service', 'utilization_rate', 'traffic_congestion_index']
    
    # Predict queue probability (class 1)
    df_copy['queue_probability'] = wait_model.predict_proba(df_copy[features])[:, 1]
    
    return df_copy

