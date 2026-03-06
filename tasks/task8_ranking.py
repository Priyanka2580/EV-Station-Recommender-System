import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def hybrid_ranking(df):
    """
    Normalizes features and computes a weighted final score for ranking.
    Includes station identity columns and handles single-station edge cases.
    """
    df_copy = df.copy()
    
    if df_copy.empty:
        return df_copy
    
    # Identity columns to keep in final output (synchronized with main.ipynb)
    identity_cols = ['station_id', 'station_name', 'city', 'state', 'network', 'charger_type', 'power_output_kw']
    
    # Metric columns to normalize and score
    norm_cols = [
        'compatibility_score', 'queue_probability', 'high_utilization_probability',
        'reliability_score', 'predicted_duration_mins', 'estimated_total_cost'
    ]
    
    # Lower-is-better columns (to be inverted after normalization)
    invert_cols = [
        'queue_probability', 
        'high_utilization_probability', 
        'predicted_duration_mins', 
        'estimated_total_cost'
    ]
    
    # --- EDGE CASE: Single Station ---
    if len(df_copy) == 1:
        # MinMaxScaler fails with 1 row; assign top score manually
        df_copy['final_score'] = 1.0
        output_cols = identity_cols + ['final_score'] + norm_cols
        return df_copy[output_cols]

    # --- NORMALIZATION ---
    scaler = MinMaxScaler()
    df_copy[norm_cols] = scaler.fit_transform(df_copy[norm_cols])
    
    # Invert designated columns (1 - normalized_value)
    # Now 1.0 always means 'best' for all components
    for col in invert_cols:
        df_copy[col] = 1.0 - df_copy[col]
    
    # --- WEIGHTED SCORING ---
    # Total weight = 1.0
    df_copy['final_score'] = (
        0.20 * df_copy['compatibility_score'] +
        0.20 * df_copy['queue_probability'] +
        0.10 * df_copy['high_utilization_probability'] +
        0.20 * df_copy['reliability_score'] +
        0.10 * df_copy['predicted_duration_mins'] +
        0.20 * df_copy['estimated_total_cost']
    )
    
    # --- FINAL SORT ---
    # Return top 5 with identity columns first for visibility
    output_cols = identity_cols + ['final_score'] + norm_cols
    ranked_df = df_copy.sort_values(by='final_score', ascending=False).head(5)
    
    return ranked_df[output_cols]

