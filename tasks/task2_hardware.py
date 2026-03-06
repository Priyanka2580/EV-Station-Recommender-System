import pandas as pd

def hardware_compatibility(df, charger_type, min_power_kw, network):
    """
    Filters by connector type and minimum power requirements.
    Adds a compatibility score weighted towards higher power output and brand affinity.
    """
    df_copy = df.copy()
    
    # --- ROBUST STRING CHECK ---
    # Handles 'DC Fast Charger' (user) vs 'DC Fast Charge' (dataset)
    clean_type = charger_type.lower().strip()
    if clean_type.endswith('charger'): clean_type = clean_type[:-1]
    
    mask = (df_copy['charger_type'].str.lower().str.contains(clean_type, regex=False)) & (df_copy['power_output_kw'] >= min_power_kw)
    df_filtered = df_copy[mask].copy()
    
    if df_filtered.empty:
        return df_filtered
    
    # Normalized Power Score (compared to filtered set's max)
    max_p = df_filtered['power_output_kw'].max()
    df_filtered['power_score'] = df_filtered['power_output_kw'] / max_p
    
    # Network Bonus (1 for match, 0 for miss)
    df_filtered['network_bonus'] = df_filtered['network'].apply(lambda x: 1 if str(x).lower().strip() == network.lower().strip() else 0)
    
    # Final Hardware Score (70% Power, 30% Brand Preference)
    df_filtered['compatibility_score'] = 0.7 * df_filtered['power_score'] + 0.3 * df_filtered['network_bonus']
    
    return df_filtered

