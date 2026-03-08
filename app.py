# Run with: streamlit run app.py
# Requires: pip install streamlit joblib pandas scikit-learn

import sys
import traceback

print("=== App Starting ===", flush=True)

try:
    import streamlit as st
    print("=== Streamlit imported ===", flush=True)
    
    import pandas as pd
    print("=== Pandas imported ===", flush=True)
    
    import joblib
    print("=== Joblib imported ===", flush=True)
    
    from tasks.task1_filter import filter_by_region
    print("=== Tasks imported ===", flush=True)

except Exception as e:
    print(f"=== IMPORT ERROR: {e} ===", flush=True)
    traceback.print_exc()
    sys.exit(1)

import os
from datetime import datetime

# Import task functions
from tasks.task2_hardware import hardware_compatibility
from tasks.task3_wait_time import predict_wait_time
from tasks.task4_high_utilization import predict_high_utilization
from tasks.task5_reliability import compute_reliability
from tasks.task6_duration import predict_duration
from tasks.task7_cost import estimate_cost
from tasks.task8_ranking import hybrid_ranking

# --- Page Configuration ---
st.set_page_config(
    page_title="EV Charging Recommender",
    page_icon="⚡",
    layout="wide"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #00C853;
        --secondary-color: #1565C0;
    }
    
    .main-header {
        color: var(--secondary-color);
        font-family: 'Inter', sans-serif;
    }
    
    .station-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 5px solid var(--primary-color);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .rank-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 50%;
        color: white;
        font-weight: bold;
        margin-right: 10px;
    }
    
    .rank-1 { background-color: #FFD700; color: #000; } /* Gold */
    .rank-2 { background-color: #C0C0C0; color: #000; } /* Silver */
    .rank-3 { background-color: #CD7F32; color: #000; } /* Bronze */
    .rank-rest { background-color: #1565C0; }
    
    .metric-value {
        font-size: 1.2rem;
        font-weight: bold;
        color: var(--secondary-color);
    }
    
    .score-high { color: #00C853; }
    .score-med { color: #FF9100; }
    .score-low { color: #D50000; }
    </style>
""", unsafe_allow_html=True)

# --- Load Models & Data ---
@st.cache_resource
def load_models():
    try:
        wait_model     = joblib.load('models/wait_time_model.pkl')
        peak_model     = joblib.load('models/high_utilization_model.pkl')
        duration_model = joblib.load('models/duration_model.pkl')
        price_model    = joblib.load('models/price_model.pkl')
        return wait_model, peak_model, duration_model, price_model
    except Exception as e:
        st.error(f"Models not found. Please run train_models.ipynb first. Error: {str(e)}")
        st.stop()

@st.cache_data
def load_data():
    data_path = 'data/ev_stations.parquet'
    if not os.path.exists(data_path):
        st.error(
            "Dataset file 'data/ev_stations.parquet' not found. "
            "Please upload or place the preprocessed dataset in the 'data' folder on the server."
        )
        st.stop()
            
    df = pd.read_parquet(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp', ascending=False)
    df = df.drop_duplicates(subset=['station_id'], keep='first')
    
    cols_to_drop = [
        'timestamp', 'latitude', 'longitude', 'location_type',
        'amenities_nearby', 'ports_available', 'precipitation_mm',
        'temperature_f', 'weather_condition', 'gas_price_per_gallon',
        'local_event', 'is_weekend', 'month'
    ]
    return df.drop(columns=cols_to_drop)

# --- UI Setup ---

# Load data once for dropdown options
df_options = load_data()
cities_list = sorted(df_options['city'].dropna().unique().tolist())
charger_types_list = sorted(df_options['charger_type'].dropna().unique().tolist())
raw_networks = df_options['network'].dropna().unique().tolist()
networks_list = sorted([n for n in raw_networks if n.lower() not in ('other', 'others')])

# Section 1: Header
st.markdown("<h1 class='main-header'>⚡ EV Charging Station Recommender</h1>", unsafe_allow_html=True)
st.subheader("Find the best charging station for your needs")
st.divider()

# Section 2: User Input Sidebar
st.sidebar.title("Your Preferences")

default_city_idx = cities_list.index("Atlanta") if "Atlanta" in cities_list else 0
city = st.sidebar.selectbox("City", options=cities_list, index=default_city_idx)

charger_type = st.sidebar.selectbox("Charger Type", options=charger_types_list)

min_power_kw = st.sidebar.slider("Minimum Power (kW)", 
                                min_value=7.2, max_value=350.0, value=50.0, step=0.1)

network = st.sidebar.selectbox("Preferred Network", options=networks_list)

day_of_week_str = st.sidebar.selectbox("Day of Week", options=[
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
])

# Map day of week to numeric
days_map = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2,
    "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
}
day_of_week = days_map[day_of_week_str]

hour_of_day = st.sidebar.slider("Hour of Day", 
                                min_value=0, max_value=23, value=17, step=1)

# Format hour for display
display_hour = datetime.strptime(str(hour_of_day), "%H").strftime("%I %p")
st.sidebar.caption(f"Selected Time: {display_hour}")

find_button = st.sidebar.button("⚡ Find Best Stations",
                               type="primary",
                               use_container_width=True)

# --- Logic: Pipeline Execution ---

if find_button:
    with st.spinner("Finding best stations for you..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Load Resources
            wait_model, peak_model, duration_model, price_model = load_models()
            df = load_data()
            
            # Task 1: Filter by Region
            status_text.text("Filtering by region...")
            # Infer the corresponding state for the selected city from the dataset
            inferred_state = df[df['city'] == city]['state'].iloc[0] if not df[df['city'] == city].empty else ""
            df = filter_by_region(df, city, inferred_state)
            progress_bar.progress(12)
            
            if df.empty:
                st.error("No stations found in this region. Try a different city or state.")
                st.stop()
            
            # Task 2: Hardware Compatibility
            status_text.text("Checking hardware compatibility...")
            df = hardware_compatibility(df, charger_type, min_power_kw, network)
            progress_bar.progress(25)
            
            if df.empty:
                st.error("No compatible stations found. Try different charger type or lower minimum power.")
                st.stop()
            
            stations_found_count = len(df)
            
            # Task 3: Predict Wait Time
            status_text.text("Predicting wait times...")
            df = predict_wait_time(df, wait_model)
            progress_bar.progress(37)
            
            # Task 4: Predict High Utilization
            status_text.text("Analyzing utilization...")
            df = predict_high_utilization(df, peak_model, day_of_week, hour_of_day)
            progress_bar.progress(50)
            
            # Task 5: Compute Reliability
            status_text.text("Computing reliability...")
            df = compute_reliability(df)
            progress_bar.progress(62)
            
            # Task 6: Predict Duration
            status_text.text("Estimating session duration...")
            df = predict_duration(df, duration_model)
            progress_bar.progress(75)
            
            # Task 7: Estimate Cost
            status_text.text("Calculating costs...")
            df = estimate_cost(df, price_model)
            progress_bar.progress(87)
            
            # Task 8: Ranking
            status_text.text("Ranking stations...")
            top_5 = hybrid_ranking(df)
            progress_bar.progress(100)
            
            # Clear progress indicators
            status_text.text("Done! Top 5 stations found.")
            # Small delay to let user see "Done!"
            import time
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()
            
            # --- Results Display ---
            st.success("Results ready!")
            
            # Summary Metrics
            best_station = top_5.iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Stations Found", stations_found_count)
            m2.metric("Best Score", f"{int(best_station['final_score']*100)}%")
            m3.metric("Estimated Cost", f"${best_station['estimated_total_cost']:.2f}")
            m4.metric("Est. Duration", f"{int(best_station['predicted_duration_mins'])} mins")
            
            st.divider()
            
            # Top 5 Cards
            for i, (idx, row) in enumerate(top_5.iterrows()):
                with st.container():
                    # Card Header with Rank and Score
                    score = row['final_score']
                    score_color = "score-high" if score > 0.7 else "score-med" if score > 0.4 else "score-low"
                    
                    st.markdown(f"""
                        <div class="station-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <strong style="font-size: 1.2rem;">{row['station_name']}</strong> 
                                    <span style="color: #666;">({row['station_id']})</span>
                                </div>
                                <div class="{score_color}" style="font-size: 1.5rem; font-weight: bold;">
                                    {score*100:.1f}% Match
                                </div>
                            </div>
                            <hr style="margin: 10px 0;">
                            <div style="display: flex; gap: 40px;">
                                <div style="flex: 1;">
                                    <p><b>Location:</b> {row['city']}, {row['state']}</p>
                                    <p><b>Type:</b> {row['charger_type']} ({row['power_output_kw']} kW)</p>
                                    <p><b>Network:</b> {row['network']}</p>
                                </div>
                                <div style="flex: 1;">
                                    <p><b>Est. Cost:</b> <span class="metric-value">${row['estimated_total_cost']:.2f}</span></p>
                                    <p><b>Est. Duration:</b> <span class="metric-value">{int(row['predicted_duration_mins'])} mins</span></p>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please check your inputs and try again.")
            st.exception(e)
else:
    # Initial state
    st.info("👈 Use the sidebar to set your preferences and find the best stations!")
    
    # Show some sample data or instructions
    st.image("https://images.unsplash.com/photo-1593941707882-a5bba14938c7?auto=format&fit=crop&q=80&w=2072", 
             caption="Charging the future", use_container_width=True)
