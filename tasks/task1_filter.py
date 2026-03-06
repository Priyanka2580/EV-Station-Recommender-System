import pandas as pd

def filter_by_region(df, city, state):
    """
    Filters the dataframe by city and state (case-insensitive and whitespace stripped).
    Returns a copy of the filtered dataframe.
    """
    df_copy = df.copy()
    
    # Standardize inputs
    city = str(city).strip().lower()
    state = str(state).strip().lower()
    
    # Filter logic
    mask = (df_copy['city'].str.strip().str.lower() == city) & (df_copy['state'].str.strip().str.lower() == state)
    filtered_df = df_copy[mask]
    
    return filtered_df

