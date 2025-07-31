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
    rul_ranges = {
        'normal': (300, 365),  # 10-12 months
        'unbalance': (90, 180),  # 3-6 months
        'misalignment': (60, 120),  # 2-4 months
        'bearing': (1, 30)  # 1-30 days
    }

    for i, (features, label) in enumerate(zip(feature_list, labels)):
        min_rul, max_rul = rul_ranges.get(label, (1, 30))

        # Add some randomness and feature-based adjustment
        base_rul = np.random.uniform(min_rul, max_rul)

        # Adjust based on feature severity (higher vibration = lower RUL)
        severity_indicators = [
            features.get('rms_x', 0),
            features.get('rms_y', 0),
            features.get('rms_z', 0),
            features.get('std_x', 0),
            features.get('std_y', 0),
            features.get('std_z', 0)
        ]

        avg_severity = np.mean(severity_indicators)
        # Normalize severity (assuming values are typically 0-1 after normalization)
        severity_factor = min(avg_severity * 2, 1.0)  # Cap at 1.0

        # Reduce RUL based on severity
        adjusted_rul = base_rul * (1 - severity_factor * 0.3)  # Reduce by up to 30%
        adjusted_rul = max(adjusted_rul, 1)  # Minimum 1 day

        rul_data.append(adjusted_rul)

    return np.array(rul_data)


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


def prepare_rul_training_data(features, labels, rul_targets):
    """Prepare features for RUL training"""
    global rul_scaler, rul_feature_names

    df_features = pd.DataFrame(features)
    rul_feature_names = df_features.columns.tolist()

    # Fill missing values
    df_features = df_features.fillna(df_features.mean())

    # Add condition as a feature (encode labels)
    label_map = {'normal': 0, 'unbalance': 1, 'misalignment': 2, 'bearing': 3}
    condition_encoded = np.array([label_map[label] for label in labels])
    df_features['condition'] = condition_encoded
    rul_feature_names.append('condition')

    # Scale features
    rul_scaler = StandardScaler()
    X_scaled = rul_scaler.fit_transform(df_features)

    return X_scaled, np.array(rul_targets), rul_feature_names


def train_rul_model(X, y, model_type='random_forest'):
    """Train RUL prediction model"""
    global rul_model, rul_training_metadata

    print(f"Training RUL {model_type} model...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if model_type == 'random_forest':
        rul_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            min_samples_split=5,
            min_samples_leaf=2
        )
        rul_model.fit(X_train, y_train)

    elif model_type == 'svm':
        # Grid search for SVM regression
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'epsilon': [0.01, 0.1, 1.0]
        }

        svr_base = SVR(kernel='rbf')
        print("Performing grid search for optimal SVR parameters...")

        grid_search = GridSearchCV(
            svr_base,
            param_grid,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        rul_model = grid_search.best_estimator_

        print(f"Best SVR parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {-grid_search.best_score_:.2f} MAE")

    else:
        raise ValueError("Supported models: 'random_forest', 'svm'")

    # Evaluate model
    y_pred = rul_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Store training metadata
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

    if model_type == 'svm':
        rul_training_metadata['best_params'] = grid_search.best_params_
        rul_training_metadata['best_cv_score'] = -grid_search.best_score_

    print(f"RUL Model Performance:")
    print(f"  Mean Absolute Error: {mae:.2f} days")
    print(f"  Root Mean Square Error: {rmse:.2f} days")
    print(f"  R² Score: {r2:.3f}")

    return rul_model


def save_rul_model(filepath):
    """Save RUL model"""
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

    # Save summary
    base_name = os.path.splitext(filepath)[0]
    summary_path = f"{base_name}_rul_summary.txt"

    with open(summary_path, 'w') as f:
        f.write("TOOL RUL PREDICTION MODEL - SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Training Date: {rul_training_metadata.get('training_date', 'Unknown')}\n")
        f.write(f"Model Type: {rul_training_metadata.get('model_type', 'Unknown')}\n")
        f.write(f"Training Samples: {rul_training_metadata.get('training_samples', 'Unknown')}\n")
        f.write(f"Test Samples: {rul_training_metadata.get('test_samples', 'Unknown')}\n")
        f.write(f"Feature Count: {rul_training_metadata.get('feature_count', 'Unknown')}\n\n")

        f.write("MODEL PERFORMANCE:\n")
        f.write(f"  Mean Absolute Error: {rul_training_metadata.get('mae', 0):.2f} days\n")
        f.write(f"  Root Mean Square Error: {rul_training_metadata.get('rmse', 0):.2f} days\n")
        f.write(f"  R² Score: {rul_training_metadata.get('r2_score', 0):.3f}\n\n")

        if rul_training_metadata.get('model_type') == 'svm':
            f.write("SVR PARAMETERS:\n")
            best_params = rul_training_metadata.get('best_params', {})
            for param, value in best_params.items():
                f.write(f"  {param}: {value}\n")
            f.write(f"Best CV Score: {rul_training_metadata.get('best_cv_score', 0):.2f} MAE\n\n")

        f.write("FEATURE NAMES:\n")
        for i, name in enumerate(rul_feature_names, 1):
            f.write(f"{i:2d}. {name}\n")

    print(f"RUL model summary saved to {summary_path}")
    return filepath


def load_rul_model(filepath):
    """Load RUL model"""
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


def predict_rul(filepath, condition_result=None):
    """Predict RUL for a given file"""
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

    # Calculate confidence interval (for random forest)
    if hasattr(rul_model, 'estimators_'):
        # Get predictions from all trees
        tree_predictions = [tree.predict(X_scaled)[0] for tree in rul_model.estimators_]
        std_dev = np.std(tree_predictions)
        confidence_interval = (max(rul_days - 1.96 * std_dev, 1), rul_days + 1.96 * std_dev)
    else:
        # For SVM, use a simple heuristic
        mae = rul_training_metadata.get('mae', 30)
        confidence_interval = (max(rul_days - mae, 1), rul_days + mae)

    # Convert to readable format
    estimated_date = datetime.now() + timedelta(days=int(rul_days))

    # Generate maintenance recommendations based on RUL
    recommendations = []
    if rul_days > 180:  # > 6 months
        recommendations = [
            "Tool in good condition",
            "Continue regular monitoring",
            "Next inspection in 1-2 months"
        ]
    elif rul_days > 90:  # 3-6 months
        recommendations = [
            "Plan maintenance within 2-3 months",
            "Increase monitoring frequency",
            "Check for early warning signs"
        ]
    elif rul_days > 30:  # 1-3 months
        recommendations = [
            "Schedule maintenance soon",
            "Monitor weekly",
            "Prepare replacement parts"
        ]
    else:  # < 1 month
        recommendations = [
            "URGENT: Schedule immediate maintenance",
            "Tool may fail soon",
            "Consider stopping operation if critical"
        ]

    return {
        'rul_days': float(rul_days),
        'rul_hours': float(rul_days * 24),
        'estimated_failure_date': estimated_date.strftime('%Y-%m-%d'),
        'confidence_interval_days': (float(confidence_interval[0]), float(confidence_interval[1])),
        'reliability_level': 'High' if std_dev < 30 else 'Medium' if std_dev < 60 else 'Low',
        'recommendations': recommendations
    }


def init_rul_training(folder_path, model_type='random_forest', progress_callback=None):
    """Initialize RUL training process"""
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
    train_rul_model(X, y, model_type)

    # Save model
    model_filename = f'{model_type}/tool_rul_{model_type}.pkl'
    saved_filepath = save_rul_model(model_filename)

    print(f"RUL training complete! Model saved as {saved_filepath}")
    return saved_filepath


def init_rul_prediction(filepath):
    """Initialize RUL prediction"""
    success = load_rul_model(filepath)
    if success:
        print("RUL model loaded and ready for predictions!")
        return predict_rul
    else:
        return None