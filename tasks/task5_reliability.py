import pandas as pd
import numpy as np

def compute_reliability(df):
    
    # Computes a reliability score (0-100) based on availability and status.
    
    if df.empty:
        return df
        
    df_copy = df.copy()
    
    # Fill ports_out_of_service nulls with 0
    df_copy['ports_out_of_service'] = df_copy['ports_out_of_service'].fillna(0)
    
    # Encode station_status: 1 if 'Operational' or 'Active', else 0
    # Note: User request says 1 if 'Active' else 0
    df_copy['station_status_encoded'] = df_copy['station_status'].apply(lambda x: 1 if str(x).lower() in ['active', 'operational'] else 0)
    
    # Availability ratio logic
    df_copy['ports_available_calc'] = (df_copy['ports_total'] - df_copy['ports_occupied'] - df_copy['ports_out_of_service']).clip(lower=0)
    df_copy['availability_ratio'] = df_copy['ports_available_calc'] / df_copy['ports_total']
    
    # Penalty for out-of-service ports
    df_copy['out_of_service_penalty'] = df_copy['ports_out_of_service'] / df_copy['ports_total']
    
    # Reliability Score Formula
    df_copy['reliability_score'] = (
        0.4 * df_copy['availability_ratio'] +
        0.3 * (1 - df_copy['utilization_rate']) +
        0.2 * (1 - df_copy['out_of_service_penalty']) +
        0.1 * df_copy['station_status_encoded']
    ) * 100
    
    return df_copy

