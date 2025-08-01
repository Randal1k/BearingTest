import os
import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import gc  # For garbage collection

# Global variables for RUL prediction
rul_model = None
rul_scaler = None
rul_feature_names = []
rul_training_metadata = {}


def generate_synthetic_rul_data(feature_list, labels, time_horizon_days=365):
    """
    Generate synthetic RUL data based on condition labels
    This is a placeholder function - replace with real data when available
    """
    rul_data = []

    # Define typical RUL ranges for each condition (in days)
    rul_lookup = {
        'normal': 330,
        'unbalance': 150,
        'misalignment': 90,
        'bearing': 15
    }

    for features, label in zip(feature_list, labels):
        base_rul = rul_lookup.get(label, 30)

        # Wear calculation: higher RMS and STD indicate more wear
        rms_avg = np.mean([features.get(f'rms_{a}', 0) for a in 'xyz'])
        std_avg = np.mean([features.get(f'std_{a}', 0) for a in 'xyz'])

        wear_index = (rms_avg + std_avg) / 2  # Can be rescaled
        severity_factor = min(wear_index * 2, 1.0)  # 0–1 scale

        # Reduce RUL proportional to wear (up to 40%)
        adjusted_rul = base_rul * (1 - severity_factor * 0.4)
        adjusted_rul = max(adjusted_rul, 1)

        rul_data.append(adjusted_rul)


    return np.array(rul_data)


def prepare_rul_training_data(features, labels, rul_targets):
    """Prepare features for RUL training with memory optimization"""
    global rul_scaler, rul_feature_names

    print(f"Preparing RUL training data for {len(features)} samples...")

    # Convert to DataFrame in chunks if large dataset
    if len(features) > 2000:
        print("Large dataset detected - using memory-efficient processing...")
        # Process in smaller chunks to avoid memory issues
        chunk_size = 1000
        df_chunks = []

        for i in range(0, len(features), chunk_size):
            chunk_features = features[i:i + chunk_size]
            chunk_df = pd.DataFrame(chunk_features)
            df_chunks.append(chunk_df)

        df_features = pd.concat(df_chunks, ignore_index=True)
        del df_chunks  # Free memory
        gc.collect()
    else:
        df_features = pd.DataFrame(features)

    rul_feature_names = df_features.columns.tolist()

    # Fill missing values
    print("Filling missing values...")
    df_features = df_features.fillna(df_features.mean())

    # Add condition as a feature (encode labels)
    print("Encoding condition labels...")
    label_map = {'normal': 0, 'unbalance': 1, 'misalignment': 2, 'bearing': 3}
    condition_encoded = np.array([label_map[label] for label in labels])
    df_features['condition'] = condition_encoded
    rul_feature_names.append('condition')

    # Scale features
    print("Scaling features...")
    rul_scaler = StandardScaler()
    X_scaled = rul_scaler.fit_transform(df_features)

    print(f"RUL data preparation complete: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")

    rul_targets = np.nan_to_num(rul_targets, nan=1.0, posinf=365.0, neginf=1.0)
    return X_scaled, np.array(rul_targets), rul_feature_names


def train_rul_model(X, y, model_type='random_forest', progress_callback=None):
    """Train RUL model with memory and runtime optimization"""
    import time
    global rul_model, rul_training_metadata

    print(f"[RUL] Starting training - shape: {X.shape}, target: {y.shape}")
    start_time = time.time()

    if progress_callback:
        progress_callback(0.0, 1.0)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if model_type == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor
        print("[RUL] Training Random Forest with optimized settings...")

        rul_model = RandomForestRegressor(
            n_estimators=25,       # zmniejszona liczba drzew
            max_depth=10,          # ograniczona głębokość
            n_jobs=-1,             # wykorzystaj wszystkie rdzenie
            verbose=2,             # pokaż każdy etap
            random_state=42
        )

        rul_model.fit(X_train, y_train)

    else:
        raise ValueError("Only 'random_forest' is supported safely for large data.")

    # Ewaluacja
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y_pred = rul_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    rul_training_metadata = {
        'training_date': datetime.now().isoformat(),
        'model_type': model_type,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'feature_count': X.shape[1],
        'mae': mae,
        'rmse': rmse,
        'r2_score': r2
    }

    print(f"[RUL] MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")

    # ⏳ czas trwania
    elapsed = time.time() - start_time
    print(f"[RUL] Training completed in {elapsed:.2f} seconds.")

    # ✅ Zapisz tymczasowy model
    with open("res/model/random_forest/__rul_model_temp.pkl", "wb") as f:
        pickle.dump(rul_model, f)
        print("[RUL] Temporary model saved.")

    if progress_callback:
        progress_callback(1.0, 1.0)

    return rul_model

def predict_rul(filepath, condition_result=None):
    """
    Predict RUL for a given file
    This function is designed to work with the integrated model system
    """
    global rul_model, rul_scaler, rul_feature_names

    if rul_model is None:
        raise ValueError("No RUL model loaded. Load RUL model first.")

    # Import your existing functions
    import tool_monitor as tm

    # Load and process signal
    df = tm.load_file(filepath)
    if df is None:
        return None

    features = tm.process_signal(df)

    # Calculate wear index
    rms_avg = np.mean([features.get(f'rms_{a}', 0) for a in 'xyz'])
    std_avg = np.mean([features.get(f'std_{a}', 0) for a in 'xyz'])
    wear_index = (rms_avg + std_avg) / 2

    # Add condition as feature if provided
    if condition_result:
        condition_map = {'normal': 0, 'unbalance': 1, 'misalignment': 2, 'bearing': 3}
        condition_encoded = condition_map.get(condition_result['condition'], 0)
    else:
        # Default to normal if no condition provided
        condition_encoded = 0

    features['condition'] = condition_encoded

    # Prepare features for prediction
    df_features = pd.DataFrame([features])
    df_features = df_features.reindex(columns=rul_feature_names, fill_value=0)

    # Scale features
    X_scaled = rul_scaler.transform(df_features)

    # Predict RUL
    rul_days = rul_model.predict(X_scaled)[0]
    rul_days = max(rul_days, 1)  # Minimum 1 day

    # Calculate confidence interval
    if hasattr(rul_model, 'estimators_'):
        # Get predictions from all trees (Random Forest)
        tree_preds = [tree.predict(X_scaled)[0] for tree in rul_model.estimators_]
        std_dev = np.std(tree_preds)
        confidence_interval = (max(rul_days - 1.96 * std_dev, 1), rul_days + 1.96 * std_dev)
        reliability = 'High' if std_dev < 15 else 'Medium' if std_dev < 40 else 'Low'
    else:
        # For SVM, use a simple heuristic
        mae = rul_training_metadata.get('mae', 25)
        confidence_interval = (max(rul_days - mae, 1), rul_days + mae)
        reliability = 'Medium'  # No way to assess from variance

    # Convert to readable format
    estimated_date = datetime.now() + timedelta(days=int(rul_days))

    # Generate maintenance recommendations based on RUL and condition
    recommendations = generate_maintenance_recommendations(rul_days, condition_result)

    return {
        'rul_days': float(rul_days),
        'rul_hours': float(rul_days * 24),
        'estimated_failure_date': estimated_date.strftime('%Y-%m-%d'),
        'confidence_interval_days': (float(confidence_interval[0]), float(confidence_interval[1])),
        'reliability_level': reliability,
        'wear_index': round(wear_index, 4),
        'recommendations': recommendations
    }


def generate_maintenance_recommendations(rul_days, condition_result=None):
    """Generate intelligent maintenance recommendations based on RUL and condition"""
    recommendations = []

    # Base recommendations on RUL
    if rul_days > 180:  # > 6 months
        base_recs = [
            "Tool in good condition",
            "Continue regular monitoring",
            "Next inspection in 1-2 months"
        ]
    elif rul_days > 90:  # 3-6 months
        base_recs = [
            "Plan maintenance within 2-3 months",
            "Increase monitoring frequency",
            "Check for early warning signs"
        ]
    elif rul_days > 30:  # 1-3 months
        base_recs = [
            "Schedule maintenance soon",
            "Monitor weekly",
            "Prepare replacement parts"
        ]
    else:  # < 1 month
        base_recs = [
            "URGENT: Schedule immediate maintenance",
            "Tool may fail soon",
            "Consider stopping operation if critical"
        ]

    recommendations.extend(base_recs)

    # Add condition-specific recommendations
    if condition_result:
        condition = condition_result['condition']
        confidence = condition_result['confidence']

        if condition == 'unbalance' and confidence > 0.7:
            recommendations.append("Check rotor balance and alignment")
            recommendations.append("Inspect shaft coupling")
        elif condition == 'misalignment' and confidence > 0.7:
            recommendations.append("Check shaft alignment")
            recommendations.append("Verify mounting and foundation")
        elif condition == 'bearing' and confidence > 0.7:
            recommendations.append("Inspect bearing condition immediately")
            recommendations.append("Check lubrication system")
            if rul_days > 30:  # Override RUL if bearing issues detected
                recommendations.insert(0, "WARNING: Bearing issues detected - reduce RUL estimate")

    return recommendations


def load_rul_training_data(folder_path, progress_callback=None):
    """
    Load training data for RUL prediction
    This function integrates with your existing data loading
    """
    # Import your existing functions
    import tool_monitor as tm

    # Load features and labels using existing function
    feature_list, labels = tm.load_data_with_progress(folder_path, progress_callback)

    if not feature_list:
        return None, None, None

    # Generate RUL targets (replace this with real RUL data when available)
    print("Generating synthetic RUL targets...")
    rul_targets = generate_synthetic_rul_data(feature_list, labels)

    print(f"Generated RUL data:")
    print(
        f"  Normal tools: {np.mean([rul for rul, label in zip(rul_targets, labels) if label == 'normal']):.1f} days avg")
    print(
        f"  Unbalance tools: {np.mean([rul for rul, label in zip(rul_targets, labels) if label == 'unbalance']):.1f} days avg")
    print(
        f"  Misalignment tools: {np.mean([rul for rul, label in zip(rul_targets, labels) if label == 'misalignment']):.1f} days avg")
    print(
        f"  Bearing issues: {np.mean([rul for rul, label in zip(rul_targets, labels) if label == 'bearing']):.1f} days avg")

    return feature_list, labels, rul_targets


def save_rul_model(filepath):
    """Save RUL model (legacy function - for standalone RUL models)"""
    global rul_model, rul_scaler, rul_feature_names, rul_training_metadata

    if rul_model is None:
        raise ValueError("No RUL model to save. Train model first.")

    # Ensure proper directory structure
    if not filepath.startswith('res/model/'):
        if '/' in filepath:
            filepath = os.path.join('res/model', filepath)
        else:
            filepath = os.path.join('res/model', filepath)

    # Create directory
    model_dir = os.path.dirname(filepath)
    os.makedirs(model_dir, exist_ok=True)

    # Generate unique filename
    original_filepath = filepath
    counter = 1
    while os.path.exists(filepath):
        base_name = os.path.splitext(original_filepath)[0]
        extension = os.path.splitext(original_filepath)[1]
        filepath = f"{base_name}_rul_{counter:03d}{extension}"
        counter += 1

    # Prepare model data
    model_data = {
        'model': rul_model,
        'scaler': rul_scaler,
        'feature_names': rul_feature_names,
        'training_metadata': rul_training_metadata
    }

    # Save model
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"RUL model saved to {filepath}")

    return filepath


def load_rul_model(filepath):
    """Load RUL model (legacy function - for standalone RUL models)"""
    global rul_model, rul_scaler, rul_feature_names, rul_training_metadata

    try:
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        rul_model = model_data['model']
        rul_scaler = model_data['scaler']
        rul_feature_names = model_data['feature_names']
        rul_training_metadata = model_data.get('training_metadata', {})

        print(f"RUL model loaded from {filepath}")
        print(f"Model type: {rul_training_metadata.get('model_type', 'Unknown')}")
        print(f"MAE: {rul_training_metadata.get('mae', 0):.2f} days")

        return True

    except FileNotFoundError:
        print(f"RUL model file {filepath} not found!")
        return False
    except Exception as e:
        print(f"Error loading RUL model: {e}")
        return False


def init_rul_training(folder_path, model_type='random_forest', progress_callback=None):
    """Initialize RUL training process (legacy function)"""
    print("Starting RUL model training...")

    # Load data
    feature_list, labels, rul_targets = load_rul_training_data(folder_path, progress_callback)

    if feature_list is None:
        print("No data loaded for RUL training.")
        return None

    print(f"Loaded {len(feature_list)} samples for RUL training")

    # Prepare training data
    X, y, feature_names = prepare_rul_training_data(feature_list, labels, rul_targets)

    # Train model
    train_rul_model(X, y, model_type, progress_callback)

    # Save model
    model_filename = f'{model_type}/tool_rul_{model_type}.pkl'
    saved_filepath = save_rul_model(model_filename)

    print(f"RUL training complete! Model saved as {saved_filepath}")
    return saved_filepath


def init_rul_prediction(filepath):
    """Initialize RUL prediction (legacy function)"""
    success = load_rul_model(filepath)
    if success:
        print("RUL model loaded and ready for predictions!")
        return predict_rul
    else:
        return None


# Legacy functions for backward compatibility
def get_rul_model():
    """Get current RUL model"""
    return rul_model


def get_rul_scaler():
    """Get current RUL scaler"""
    return rul_scaler


def get_rul_feature_names():
    """Get current RUL feature names"""
    return rul_feature_names


def get_rul_metadata():
    """Get current RUL training metadata"""
    return rul_training_metadata