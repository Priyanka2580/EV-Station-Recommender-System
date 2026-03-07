import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def hybrid_ranking(df):
    """
    Normalizes features and computes a weighted final score for ranking.
    Original display columns (duration_mins, total_cost, reliability_score)
    are preserved; separate _score columns are created for the ranking math.
    """
    df_copy = df.copy()
    
    if df_copy.empty:
        return df_copy
    
    # Identity columns to keep in final output (synchronized with main.ipynb)
    identity_cols = ['station_id', 'station_name', 'city', 'state', 'network', 'charger_type', 'power_output_kw']
    
    # --- STEP 1: Create score copies for the 3 columns whose units must be preserved ---
    # These originals will NOT be touched by the scaler.
    df_copy['duration_score']      = df_copy['predicted_duration_mins']
    df_copy['cost_score']          = df_copy['estimated_total_cost']
    df_copy['reliability_score_n'] = df_copy['reliability_score']
    
    # --- STEP 2: Normalize only the score/probability columns ---
    # compatibility_score and probabilities are already ~0-1 so they are fine to scale directly.
    # For the 3 new _score cols we scale in isolation to keep original cols clean.
    norm_cols = [
        'compatibility_score',
        'queue_probability',
        'high_utilization_probability',
        'reliability_score_n',
        'duration_score',
        'cost_score'
    ]
    
    # Lower-is-better score columns (inverted after normalization)
    invert_cols = [
        'queue_probability',
        'high_utilization_probability',
        'duration_score',
        'cost_score'
    ]
    
    # --- EDGE CASE: Single Station ---
    if len(df_copy) == 1:
        # MinMaxScaler fails with 1 row; assign top score manually
        df_copy['final_score'] = 1.0
        output_cols = identity_cols + ['final_score', 'predicted_duration_mins', 'estimated_total_cost', 'reliability_score'] + ['queue_probability', 'high_utilization_probability']
        return df_copy[output_cols]

    # --- NORMALIZATION (only on score columns) ---
    scaler = MinMaxScaler()
    df_copy[norm_cols] = scaler.fit_transform(df_copy[norm_cols])
    
    # Invert lower-is-better score columns (1 = best for all components)
    for col in invert_cols:
        df_copy[col] = 1.0 - df_copy[col]
    
    # --- WEIGHTED SCORING ---
    # Total weight = 1.0
    df_copy['final_score'] = (
        0.20 * df_copy['compatibility_score'] +
        0.20 * df_copy['queue_probability'] +
        0.10 * df_copy['high_utilization_probability'] +
        0.20 * df_copy['reliability_score_n'] +
        0.10 * df_copy['duration_score'] +
        0.20 * df_copy['cost_score']
    )
    
    # --- FINAL SORT ---
    # Return top 5 with identity + ORIGINAL display columns + final_score
    output_cols = (
        identity_cols +
        ['final_score',
         'predicted_duration_mins',   # Original minutes (e.g. 120 mins)
         'estimated_total_cost',      # Original dollars (e.g. $15.00)
         'reliability_score',         # Original 0-100 score
         'queue_probability',
         'high_utilization_probability']
    )
    ranked_df = df_copy.sort_values(by='final_score', ascending=False).head(5)
    
    return ranked_df[output_cols]
