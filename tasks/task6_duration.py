import pandas as pd

def predict_duration(df, duration_model):
    """
    Uses the pre-trained duration_model to predict average session duration.
    Model C is now refined to use only hardware features: charger_type and power_output_kw.
    """
    df_copy = df.copy() 
    
    if df_copy.empty:
        return df_copy
        
    # Select features required by refined Model C (Hardware only)
    features = ['power_output_kw', 'charger_type']
    
    # Predict
    df_copy['predicted_duration_mins'] = duration_model.predict(df_copy[features])
    
    return df_copy

