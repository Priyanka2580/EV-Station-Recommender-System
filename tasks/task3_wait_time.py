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
        
    # Features required by Model A (Structural Risk Focus)
    features = [
        'power_output_kw', 'ports_total', 'traffic_congestion_index', 
        'is_peak_hour','charger_type', 'pricing_type'
    ]
    
    # Predict queue probability (class 1)
    df_copy['queue_probability'] = wait_model.predict_proba(df_copy[features])[:, 1]
    
    return df_copy

