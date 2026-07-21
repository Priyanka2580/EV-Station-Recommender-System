import pandas as pd

def predict_wait_time(df):

    # Rule-based (no model): has_wait is deterministic in the data --
    # a station has a queue exactly when every port is out of service
    # (ports_out_of_service == ports_total, equivalently ports_available == 0).
    df_copy = df.copy()

    if df_copy.empty:
        return df_copy

    df_copy['queue_probability'] = (
        df_copy['ports_out_of_service'] == df_copy['ports_total']
    ).astype(float)

    return df_copy

