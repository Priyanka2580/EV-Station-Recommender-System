import pandas as pd
import numpy as np

def estimate_cost(df, price_model):
    """
    1. Predicts dynamic current_price using Model D.
    2. Estimates total charging cost based on predicted_price, duration, and pricing type.
    """
    df_copy = df.copy()
    
    if df_copy.empty:
        return df_copy

    # --- Part A: Price Prediction (Model D) ---
    # Features required by Model D (Structural/Economic Focus)
    price_features = [
        'hour_of_day', 'is_peak_hour', 'power_output_kw', 'ports_total',
        'charger_type', 'pricing_type', 'network'
    ]
    
    # Predict base price
    df_copy['predicted_price'] = price_model.predict(df_copy[price_features])
    
    # DETERMINISTIC RULE: Free stations always cost exactly 0.000
    df_copy.loc[df_copy['pricing_type'] == 'free', 'predicted_price'] = 0.0
    
    # Ensure no negative prices
    df_copy['predicted_price'] = df_copy['predicted_price'].clip(lower=0)
    
    # --- Part B: Total Cost Calculation ---
    def calculate_row_cost(row):
        pt = str(row['pricing_type']).lower()
        duration = row['predicted_duration_mins']
        price = row['predicted_price']
        power = row['power_output_kw']
        
        if pt == 'per_kwh':
            return (power * duration / 60) * price
        elif pt == 'per_minute':
            return duration * price
        elif pt == 'flat_rate':
            return price
        else:
            return 0.0
            
    df_copy['estimated_total_cost'] = df_copy.apply(calculate_row_cost, axis=1)
    
    return df_copy

