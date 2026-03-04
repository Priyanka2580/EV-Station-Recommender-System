# EV Charging Station Recommender Backend Pipeline

This is a complete backend data processing and ML pipeline in Python for recommending EV charging stations based on various parameters like wait time, reliability, and cost.

## Project Folder Structure
- `main.ipynb`: Runs the full pipeline.
- `data/`: Contains the dataset `ev_stations.csv`.
- `models/`: Contains the training notebook and saved `.pkl` models.
- `tasks/`: Individual notebooks for each task in the pipeline.
- `requirements.txt`: Project dependencies.

## Setup Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place dataset at:
   `data/ev_stations.csv` (Automatically handled if already present in root as `ev_charging_station_data.csv`)

3. Train models (runs EDA, feature engineering, GridSearchCV, saves .pkl files):
   Open `models/train_models.ipynb` and run all cells.

4. Run the full pipeline:
   Open `main.ipynb` and run all cells.

## Features
- Comprehensive EDA and automated model training with GridSearchCV.
- 8-stage pipeline from filtering to hybrid ranking.
- Overfitting detection and mitigation.
- Self-contained ML pipelines using `scikit-learn` Pipelines.
