import pandas as pd
import joblib
import os

# --- 1. CONFIGURATION & INPUT ---
USER_INPUT = {
    'city': "Atlanta",
    'state': "GA",
    'charger_type': "DC Fast Charge",
    'min_power_kw': 50,
    'network': "ChargePoint",
    'day_of_week': 0,    # Monday
    'hour_of_day': 17     # 5:00 PM
}

def load_pipeline_data(filepath):
    if not os.path.exists(filepath):
        # Fallback to local data folder if running from a different directory
        filepath = os.path.join('data', os.path.basename(filepath))
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    # Using Parquet for optimal performance with 1.3M rows
    df = pd.read_parquet(filepath)
    
    # Deduplicate: Keep only the most recent entry per station
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by='timestamp', ascending=False)
        df = df.drop_duplicates(subset=['station_id'], keep='first')
    
    cols_to_drop = [
        'timestamp', 'latitude', 'longitude', 'location_type', 'amenities_nearby',
        'ports_available', 'precipitation_mm', 'temperature_f', 'weather_condition',
        'gas_price_per_gallon', 'local_event', 'is_weekend', 'month'
    ]
    # Drop columns only if they exist
    existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=existing_cols_to_drop)

def ensure_models():
    # Synchronized with Original Filenames as requested
    model_paths = [
        'models/high_utilization_model.pkl',
        'models/duration_model.pkl',
        'models/price_model.pkl'
    ]
    return all(os.path.exists(p) for p in model_paths)

if not ensure_models():
    print("CRITICAL: Models not found at the expected paths. Please run models/train_models.ipynb first.")
else:
    # --- LOAD DATA & MODELS ---
    df = load_pipeline_data('data/ev_stations.parquet')
    peak_model = joblib.load('models/high_utilization_model.pkl')
    duration_model = joblib.load('models/duration_model.pkl')
    price_model = joblib.load('models/price_model.pkl')
    
    # --- INJECT USER CONTEXT ---
    df['hour_of_day'] = USER_INPUT['hour_of_day']
    df['day_of_week'] = USER_INPUT['day_of_week']
    # Derive is_peak_hour context for the high-utilization and price models
    df['is_peak_hour'] = df['hour_of_day'].apply(lambda x: 1 if x in [7, 8, 9, 17, 18, 19, 20] else 0)

    # Fix stale traffic_congestion_index — replace with the correct value for the user's selected hour
    hourly_congestion = {
        0: 2.0, 1: 2.0, 2: 2.0, 3: 2.0, 4: 2.0, 5: 2.0,
        6: 5.0, 7: 8.0, 8: 8.0, 9: 8.0, 10: 5.0,
        11: 5.0, 12: 5.0, 13: 5.0, 14: 5.0, 15: 5.0, 16: 5.0,
        17: 8.0, 18: 8.0, 19: 8.0, 20: 5.0,
        21: 5.0, 22: 5.0, 23: 5.0
    }
    df['traffic_congestion_index'] = hourly_congestion[USER_INPUT['hour_of_day']]
    
    print("Starting Sequential Pipeline (Production Python Script)...")
    
    # --- TASK EXECUTION SEQUENCE ---
    # We use exec(open().read()) to maintain the sequential global scope of the tasks
    
    # Task 1: Filter by Region
    exec(open('tasks/task1_filter.py').read())
    df = filter_by_region(df, USER_INPUT['city'], USER_INPUT['state'])
    print(f"Task 1 Complete: {len(df)} stations remain.")
    
    if not df.empty:
        # Task 2: Hardware Compatibility
        exec(open('tasks/task2_hardware.py').read())
        df = hardware_compatibility(df, USER_INPUT['charger_type'], USER_INPUT['min_power_kw'], USER_INPUT['network'])
        print(f"Task 2 Complete: {len(df)} stations remain.")
        
        if not df.empty:
            # Task 3: Wait Time (rule-based)
            exec(open('tasks/task3_wait_time.py').read())
            df = predict_wait_time(df)
            print("Task 3 Complete: Queue probabilities predicted.")
            
            # Task 4: High Utilization (Model B)
            exec(open('tasks/task4_high_utilization.py').read())
            df = predict_high_utilization(df, peak_model, USER_INPUT['day_of_week'], USER_INPUT['hour_of_day'])
            print("Task 4 Complete: High utilization probabilities added.")
            
            # Task 5: Reliability
            exec(open('tasks/task5_reliability.py').read())
            df = compute_reliability(df)
            print("Task 5 Complete: Reliability scores computed.")
            
            # Task 6: Duration (Model C)
            exec(open('tasks/task6_duration.py').read())
            df = predict_duration(df, duration_model)
            print("Task 6 Complete: Predicted durations added.")
            
            # Task 7: Cost (Model D)
            exec(open('tasks/task7_cost.py').read())
            df = estimate_cost(df, price_model)
            print("Task 7 Complete: Dynamic prices and total costs estimated.")
            
            # Task 8: Ranking
            exec(open('tasks/task8_ranking.py').read())
            top_5 = hybrid_ranking(df)
            print("Task 8 Complete: Top 5 stations ranked.")
            
            # --- FINAL OUTPUT ---
            print("\n--- TOP 5 RECOMMENDED STATIONS (Honest Logic) ---")
            print(top_5[['station_id', 'station_name', 'city', 'state', 'charger_type', 'power_output_kw', 'final_score']].to_string(index=False))
        else:
            print("No compatible stations found in the region.")
    else:
        print("No stations found in the specified region.")
