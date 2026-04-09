import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier

def load_splits():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '../data/')
    train = pd.read_parquet(os.path.join(data_dir, 'train.parquet'))
    val = pd.read_parquet(os.path.join(data_dir, 'val.parquet'))
    test = pd.read_parquet(os.path.join(data_dir, 'test.parquet'))
    return train, val, test

print("Loading data...")
train_df, val_df, test_df = load_splits()

# Simple Target Engineering
for df in [train_df, val_df, test_df]:
    df['has_wait'] = (df['estimated_wait_time_mins'] > 0).astype(int)

# 1. Features (Simplified - back to your original list + 'network')
features_a = ['power_output_kw', 'ports_total', 'traffic_congestion_index', 'is_peak_hour', 'charger_type', 'pricing_type', 'network']

X_train, y_train = train_df[features_a], train_df['has_wait']
X_test, y_test = test_df[features_a], test_df['has_wait']

# 2. Simple Preprocessor (Exactly like your original)
preprocessor_a = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ]), ['power_output_kw', 'ports_total', 'traffic_congestion_index']),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ]), ['charger_type', 'pricing_type', 'network'])
])

# 3. Simple Model (Your original + scale_pos_weight)
model_a = Pipeline([
    ('preprocessor', preprocessor_a),
    ('classifier', XGBClassifier(
        n_estimators=100, 
        max_depth=6, 
        learning_rate=0.1, 
        scale_pos_weight=5, # <--- The only "extra" thing needed to fix recall
        use_label_encoder=False, 
        eval_metric='logloss', 
        random_state=42
    ))
])

print("Training Simple Model...")
model_a.fit(X_train, y_train)

# 4. Evaluation (Standard)
y_pred = model_a.predict(X_test)

print("\n--- Model A: Simple & Improved Results ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Wait', 'Has Wait']))

cm = confusion_matrix(y_test, y_pred)
print("\n--- Confusion Matrix ---")
print(f"{'':>12} | {'Pred No Wait':>12} | {'Pred Has Wait':>12}")
print("-" * 45)
print(f"{'Act No Wait':>12} | {cm[0][0]:>12} | {cm[0][1]:>12}")
print(f"{'Act Has Wait':>12} | {cm[1][0]:>12} | {cm[1][1]:>12}")
