# imports
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas as pn
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from datetime import datetime
import pickle
import time

from sklearn.svm import SVC
from tqdm import tqdm

model = None
scaler = None
feature_names = []
training_metadata = {}


def load_file(filename):
    try:
        file = pn.read_csv(filename, header=None, names=["t", "x", "y", "z"])
        return file
    except Exception as e:
        print(f'Error Loading {filename}: {e}')
        return None


def load_features(filename):
    try:
        column_names = ['mean_x', 'std_x', 'rms_x', 'p2p_x', 'if_x', 'skewness_x', 'kurtosis_x',
                        'crest_factor_x', 'shape_factor_x',
                        'mean_y', 'std_y', 'rms_y', 'p2p_y', 'if_y', 'skewness_y', 'kurtosis_y',
                        'crest_factor_y', 'shape_factor_y',
                        'mean_z', 'std_z', 'rms_z', 'p2p_z', 'if_z', 'skewness_z', 'kurtosis_z',
                        'crest_factor_z', 'shape_factor_z']
        file = pn.read_csv(filename, header=None, names=column_names)

        if not all(col in file.columns for col in column_names):
            print(f"Error: CSV must contain columns: {column_names}")
            return None

        return file
    except Exception as e:
        print(f'Error Loading {filename}: {e}')
        return None


def get_folder_name(folderName):
    folder = folderName.lower()
    if folder == 'normal':
        return 'normal'
    elif folder == 'bearing':
        return 'bearing'
    elif folder == 'unbalance':
        return 'unbalance'
    elif folder == 'misalignment':
        return 'misalignment'
    else:
        return 'normal'


def load_data_with_progress(folderPath, progress_callback=None):
    """Load data with progress updates for integrated training"""
    allFiles = []
    failedFile = []
    featureList = []
    labels = []

    # Count files first
    for subfolder in os.listdir(folderPath):
        subfolderPath = os.path.join(folderPath, subfolder)
        if os.path.isdir(subfolderPath):
            csvFiles = [f for f in os.listdir(subfolderPath) if f.endswith('.csv')]
            for filename in csvFiles:
                filePath = os.path.join(subfolderPath, filename)
                allFiles.append((filePath, subfolder))

    total_files = len(allFiles)
    start_time = time.time()

    print(f"Processing {total_files} files for integrated training...")

    # Process with progress updates
    for i, (filePath, folderName) in enumerate(allFiles):
        try:
            # Your existing processing code
            signal = load_file(filePath)
            if signal is None:
                failedFile.append(filePath)
                continue

            features = process_signal(signal)
            featureList.append(features)
            label = get_folder_name(folderName)
            labels.append(label)

            # Update progress
            if progress_callback:
                elapsed = time.time() - start_time
                progress_callback(i + 1, total_files, elapsed)

        except Exception as e:
            print(f"Error processing {filePath}: {e}")
            failedFile.append(filePath)

    if failedFile:
        print(f"Warning: {len(failedFile)} files failed to process")

    return featureList, labels


def load_data(folderPath):
    allFiles = []
    failedFile = []
    featureList = []
    labels = []

    for subfolder in os.listdir(folderPath):
        subfolderPath = os.path.join(folderPath, subfolder)

        if os.path.isdir(subfolderPath):
            csvFiles = [f for f in os.listdir(subfolderPath) if f.endswith('.csv')]

            for filename in csvFiles:
                filePath = os.path.join(subfolderPath, filename)
                allFiles.append((filePath, subfolder))

    # Process
    for filePath, folderName in tqdm(allFiles, desc="Processing files"):
        try:
            # Load signal data
            signal = load_file(filePath)
            if signal is None:
                failedFile.append(filePath)
                continue

            # Extract features
            features = process_signal(signal)
            featureList.append(features)

            # Get label from folder name
            label = get_folder_name(folderName)
            labels.append(label)

        except Exception as e:
            print(f"Error processing {filePath}: {e}")
            failedFile.append(filePath)

    save_features(featureList)
    return featureList, labels


def signal_FFT(data):
    windowed_signal = data * np.hanning(len(data))
    N = len(data)
    fp = 1 / 20000
    yf = fft(windowed_signal)
    xf = fftfreq(N, fp)
    positive_idx = xf >= 0
    frequencies = xf[positive_idx]
    magnitude = np.abs(yf[positive_idx])

    return frequencies, magnitude


'''
    Filter for signal
        data        = signal to filter
        fs          = sampling frequency
        freqLimit   = cutoff frequency
        filterType  = filter type // low or lp - lowpass, high or hp - highpass
        order       = filter order
'''


def signal_filter(data, fs, freqLimit, filterType, order):
    Wn = freqLimit / (fs / 2)
    b, a = butter(order, Wn, btype=filterType, analog=False)
    return filtfilt(b, a, data)


def signal_normalization(data):
    # return (data - np.min(data)) / (np.max(data) - np.min(data))
    # return (data - np.mean(data)) / np.std(data)
    return data / np.max(data)


def signal_average(data):
    return np.mean(data)


def signal_std_deviation(data):
    return np.std(data)


def signal_RMS(data):
    return np.sqrt(signal_average(data ** 2))


def signal_P2P(data):
    return np.max(data) - np.min(data)


def signal_IF(data):
    return np.max(data) / signal_average(data)


def signal_skewness(data):
    return skew(data, axis=0, bias=True)


def signal_kurtosis(data):
    return kurtosis(data, axis=0, bias=True)


def signal_crestFactor(data):
    return abs(np.max(data)) / signal_RMS(data)


def signal_shapeFactor(data):
    return 1 / signal_average(data)


def calculate_features(data, axis):
    features = {}

    features[f'mean_{axis}'] = signal_average(data)
    features[f'std_{axis}'] = signal_std_deviation(data)
    features[f'rms_{axis}'] = signal_RMS(data)
    features[f'p2p_{axis}'] = signal_P2P(data)
    features[f'if_{axis}'] = signal_IF(data)
    features[f'skewness_{axis}'] = signal_skewness(data)
    features[f'kurtosis_{axis}'] = signal_kurtosis(data)
    features[f'crest_factor_{axis}'] = signal_crestFactor(data)
    features[f'shape_factor_{axis}'] = signal_shapeFactor(data)

    return features


def process_signal(data):
    allFeatures = {}
    for axis in ['x', 'y', 'z']:
        time_signal = data[axis]

        fftFreq, fftMag = signal_FFT(time_signal)
        normalizedSignal = signal_normalization(fftMag)

        feature = calculate_features(normalizedSignal, axis)
        allFeatures.update(feature)

    return allFeatures


def moving_average(data, window_size=15):
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='same')


def plot_axis_features(axis, featureList, labels):
    axis_features = {}

    if featureList:
        firstSample = featureList[0]
        featureNames = [key for key in firstSample.keys() if key.endswith(f'_{axis}')]

        # Extract values for each X-axis feature
        for Name in featureNames:
            axis_features[Name] = []

            for sample in featureList:
                axis_features[Name].append(sample.get(Name, 0))

    fig, axs = plt.subplots(3, 3, figsize=(15, 8), dpi=100)
    axs = axs.flatten()

    startBear = labels.count('bearing')
    startMisal = startBear + labels.count('misalignment')
    startNorm = startMisal + labels.count('normal')
    startUnbal = startNorm + labels.count('unbalance')
    print(startBear)
    print(startNorm)
    print(startUnbal)
    print(startMisal)

    for i, (plotname, values) in enumerate(axis_features.items()):
        bearing = moving_average(values[:1000])
        misal = moving_average(values[1000:2000])
        normal = moving_average(values[2000:3000])
        unbal = moving_average(values[3000:4000])

        x = np.arange(1000)

        axs[i].plot(x, bearing, label='Bearing')
        axs[i].plot(x, normal, label='Normal')
        axs[i].plot(x, unbal, label='Unbalance')
        axs[i].plot(x, misal, label='Misalignment')

        axs[i].set_title(plotname)
        axs[i].set_xlabel('Index')
        axs[i].set_ylabel('Value')
        axs[i].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    axs[0].legend(loc='upper right', fontsize='small')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

    return axis_features


def plot_axis_features_from_file(fig, axis, feature, segment_size):
    plotsNames = ['mean', 'std', 'rms', 'p2p', 'if', 'skewness', 'kurtosis', 'crest_factor', 'shape_factor']
    column_names = [f'{name}_{axis}' for name in plotsNames]
    data = feature[column_names]

    segments = {
        'Bearing': data.iloc[0:segment_size],
        'Unbalance': data.iloc[segment_size:2 * segment_size],
        'Normal': data.iloc[2 * segment_size:3 * segment_size],
        'Misalignment': data.iloc[3 * segment_size:4 * segment_size],
    }

    fig.subplots_adjust(hspace=0.4)
    axes = fig.subplots(3, 3).ravel()

    for i, feature_name in enumerate(column_names):
        ax = axes[i]
        x_vals = np.arange(segment_size)

        for label, segment in segments.items():
            values = segment[feature_name].values
            if len(values) == 0:
                continue
            smoothed = moving_average(values)
            ax.plot(x_vals, smoothed, label=label)

        ax.set_title(feature_name.replace(f"_{axis}", "").capitalize())
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize='small', bbox_to_anchor=(0.5, 0))
    fig.suptitle(f'Porównanie cech dla osi {axis.upper()}', fontsize=16)


def prepare_training_data(features, labels):
    """Prepare training data with improved error handling"""
    global scaler, feature_names

    print(f"Preparing training data for {len(features)} samples...")

    dfFeature = pd.DataFrame(features)
    feature_names = dfFeature.columns.tolist()

    # Handle missing values
    dfFeature = dfFeature.fillna(dfFeature.mean())

    # Check for any remaining issues
    if dfFeature.isnull().any().any():
        print("Warning: Still have NaN values after filling with mean")
        dfFeature = dfFeature.fillna(0)

    scaler = StandardScaler()
    XScaled = scaler.fit_transform(dfFeature)

    labelMap = {'normal': 0, 'unbalance': 1, 'misalignment': 2, 'bearing': 3}
    y = np.array([labelMap[label] for label in labels])

    print(f"Features prepared: {len(feature_names)} features")
    print(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    return XScaled, y, feature_names


def train_model(X, y, modelType='random_forest'):
    """Train model with improved logging for integration"""
    global model, training_metadata

    print(f"Training {modelType} model for integrated system...")

    # Split data for training and testing
    XTrain, XTest, yTrain, yTest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if modelType == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,  # Number of trees
            max_depth=10,  # Maximum tree depth
            random_state=42,  # For reproducible results
            class_weight='balanced'  # Handle unbalanced classes
        )
        model.fit(XTrain, yTrain)

    elif modelType == 'svm':
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
        svm_base = SVC(random_state=42, class_weight='balanced', probability=True)

        print("Performing grid search for optimal SVM parameters...")
        grid_search = GridSearchCV(
            svm_base,
            param_grid,
            cv=5,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(XTrain, yTrain)
        model = grid_search.best_estimator_

        print(f"Best SVM parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.3f}")

    else:
        raise ValueError("Supported models: 'random_forest', 'svm'")

    # Test the model
    yPred = model.predict(XTest)

    # Store training metadata
    training_metadata = {
        'training_date': datetime.now().isoformat(),
        'model_type': modelType,
        'training_samples': len(XTrain),
        'test_samples': len(XTest),
        'feature_count': X.shape[1],
        'classification_report': classification_report(yTest, yPred,
                                                       target_names=['normal', 'unbalance', 'misalignment', 'bearing'],
                                                       output_dict=True)
    }

    if modelType == 'svm':
        training_metadata['best_params'] = grid_search.best_params_
        training_metadata['best_cv_score'] = grid_search.best_score_

    print("Condition Model Performance:")
    print(classification_report(yTest, yPred,
                                target_names=['normal', 'unbalance', 'misalignment', 'bearing']))

    return model


def generate_unique_filename(base_path, extension):
    """Generate a unique filename by adding a counter if file exists"""
    if not os.path.exists(base_path + extension):
        return base_path + extension

    counter = 1
    while True:
        new_path = f"{base_path}_{counter:02d}{extension}"
        if not os.path.exists(new_path):
            return new_path
        counter += 1


def save_features(data, filename="features"):
    """Save features to CSV with improved error handling"""
    try:
        df = pd.DataFrame(data)
        output_path = f'{filename}.csv'
        df.to_csv(output_path, index=None, header=None)
        print(f"Features saved to {output_path}")
    except Exception as e:
        print(f"Error saving features: {e}")


def save_model(filepath, feature_list, saveReadable=True):
    """Save model with integration support"""
    global model, scaler, feature_names, training_metadata

    if model is None:
        raise ValueError("No model to save. Train model first.")

    # Ensure the filepath uses the res/model directory structure
    if not filepath.startswith('res/model/'):
        # If user provides a relative path like "RandomForest/test1.pkl"
        if '/' in filepath:
            filepath = os.path.join('res/model', filepath)
        else:
            # If just a filename, put it directly in res/model
            filepath = os.path.join('res/model', filepath)

    # Create directory structure if it doesn't exist
    model_dir = os.path.dirname(filepath)
    os.makedirs(model_dir, exist_ok=True)

    # Generate unique filename if file already exists
    original_filepath = filepath
    counter = 1
    while os.path.exists(filepath):
        base_name = os.path.splitext(original_filepath)[0]
        extension = os.path.splitext(original_filepath)[1]
        filepath = f"{base_name}_{counter:03d}{extension}"
        counter += 1

    if filepath != original_filepath:
        print(f"File {original_filepath} already exists. Saving as {filepath}")

    # Prepare model data for PKL
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'training_metadata': training_metadata
    }

    # Save PKL file (for deployment)
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Condition model saved to {filepath}")

    if saveReadable:
        base_name = os.path.splitext(filepath)[0]

        # 1. Save feature importance to CSV (for Random Forest)
        if hasattr(model, 'feature_importances_'):
            feature_importance_df = pd.DataFrame({
                'feature_name': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            importance_path = f"{base_name}_feature_importance.csv"
            # Ensure unique filename for importance file too
            counter = 1
            original_importance_path = importance_path
            while os.path.exists(importance_path):
                base_importance = os.path.splitext(original_importance_path)[0]
                importance_path = f"{base_importance}_{counter:03d}.csv"
                counter += 1

            feature_importance_df.to_csv(importance_path, index=False)
            print(f"Feature importance saved to {importance_path}")

        # 2. Save model summary to text file
        summary_path = f"{base_name}_summary.txt"
        # Ensure unique filename for summary file too
        counter = 1
        original_summary_path = summary_path
        while os.path.exists(summary_path):
            base_summary = os.path.splitext(original_summary_path)[0]
            summary_path = f"{base_summary}_{counter:03d}.txt"
            counter += 1

        with open(summary_path, 'w') as f:
            f.write("TOOL CONDITION MONITOR - MODEL SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Training Date: {training_metadata.get('training_date', 'Unknown')}\n")
            f.write(f"Model Type: {training_metadata.get('model_type', 'Unknown')}\n")
            f.write(f"Training Samples: {training_metadata.get('training_samples', 'Unknown')}\n")
            f.write(f"Test Samples: {training_metadata.get('test_samples', 'Unknown')}\n")
            f.write(f"Feature Count: {training_metadata.get('feature_count', 'Unknown')}\n\n")

            # Add SVM-specific info
            if training_metadata.get('model_type') == 'svm':
                f.write("SVM PARAMETERS:\n")
                best_params = training_metadata.get('best_params', {})
                for param, value in best_params.items():
                    f.write(f"  {param}: {value}\n")
                f.write(f"Best CV Score: {training_metadata.get('best_cv_score', 'Unknown'):.3f}\n\n")

            f.write("FEATURE NAMES:\n")
            for i, name in enumerate(feature_names, 1):
                f.write(f"{i:2d}. {name}\n")

            f.write("\nMODEL PERFORMANCE:\n")
            if 'classification_report' in training_metadata:
                report = training_metadata['classification_report']
                f.write(f"Overall Accuracy: {report['accuracy']:.3f}\n\n")

                for class_name in ['normal', 'unbalance', 'misalignment', 'bearing']:
                    if class_name in report:
                        f.write(f"{class_name.capitalize()}:\n")
                        f.write(f"  Precision: {report[class_name]['precision']:.3f}\n")
                        f.write(f"  Recall: {report[class_name]['recall']:.3f}\n")
                        f.write(f"  F1-Score: {report[class_name]['f1-score']:.3f}\n")
                        f.write(f"  Support: {report[class_name]['support']}\n\n")

        print(f"Model summary saved to {summary_path}")

    # Save features for integrated system
    if feature_list:
        save_features(feature_list, base_name)

    return filepath


def load_model(filepath=None):
    """Load model with integration support"""
    global model, scaler, feature_names, training_metadata

    if filepath is None:
        models_dir = 'res/model'
        if not os.path.exists(models_dir):
            print(f"Models directory {models_dir} not found!")
            return

        pkl_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        if not pkl_files:
            print(f"No .pkl files found in {models_dir}")
            return

        # Sort by modification time, get the most recent
        pkl_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
        filepath = os.path.join(models_dir, pkl_files[0])
        print(f"Loading most recent model: {filepath}")

    try:
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']
        training_metadata = model_data.get('training_metadata', {})

        print(f"Condition model loaded from {filepath}")
        print(f"Model type: {training_metadata.get('model_type', 'Unknown')}")
        print(f"Trained on: {training_metadata.get('training_date', 'Unknown')}")
        print(f"Features: {len(feature_names)}")

        return True

    except FileNotFoundError:
        print(f"Model file {filepath} not found!")
        return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def predict_condition(filepath):
    """Predict condition with enhanced recommendations"""
    global model, scaler, feature_names

    if model is None:
        raise ValueError("No model loaded. Load model first.")

    # Process the signal
    df = load_file(filepath)
    if df is None:
        return None

    features = process_signal(df)

    # Convert to array and scale
    df_features = pd.DataFrame([features])
    df_features = df_features.reindex(columns=feature_names, fill_value=0)
    X_scaled = scaler.transform(df_features)

    # Make prediction
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]

    # Convert to readable format
    condition_map = {0: 'normal', 1: 'unbalance', 2: 'misalignment', 3: 'bearing'}
    predicted_condition = condition_map[prediction]

    # Generate enhanced recommendations
    recommendations = generate_condition_recommendations(predicted_condition, probabilities)

    return {
        'condition': predicted_condition,
        'confidence': float(np.max(probabilities)),
        'probabilities': {
            'normal': float(probabilities[0]),
            'unbalance': float(probabilities[1]),
            'misalignment': float(probabilities[2]),
            'bearing': float(probabilities[3])
        },
        'recommendations': recommendations,
    }


def generate_condition_recommendations(condition, probabilities):
    """Generate intelligent condition-based recommendations"""
    recommendations = []
    confidence = np.max(probabilities)

    if condition == 'normal':
        if confidence > 0.9:
            recommendations = [
                "Tool condition is excellent",
                "Continue normal operation",
                "Next inspection in 2-3 months"
            ]
        else:
            recommendations = [
                "Tool condition appears normal",
                "Monitor for any changes",
                "Consider more frequent inspections"
            ]
    elif condition == 'unbalance':
        recommendations = [
            "Unbalance detected in the tool",
            "Check rotor balance and shaft alignment",
            "Inspect coupling and mounting",
            "Schedule balancing service"
        ]
        if confidence > 0.8:
            recommendations.insert(1, "High confidence - immediate attention required")
    elif condition == 'misalignment':
        recommendations = [
            "Shaft misalignment detected",
            "Check shaft alignment with laser alignment tool",
            "Verify foundation and mounting integrity",
            "Inspect coupling for wear"
        ]
        if confidence > 0.8:
            recommendations.insert(1, "High confidence - alignment correction needed")
    elif condition == 'bearing':
        recommendations = [
            "Bearing fault detected - URGENT",
            "Inspect bearing condition immediately",
            "Check lubrication system",
            "Plan for bearing replacement"
        ]
        if confidence > 0.7:
            recommendations.insert(1, "CRITICAL: Consider immediate shutdown if essential")

    return recommendations


def init_prediction(filepath=None):
    """Initialize prediction with integration support"""
    success = load_model(filepath)
    if success:
        print("Condition model loaded and ready for predictions!")
        return predict_condition
    else:
        return None


def init_training(folderPath, model_type='random_forest'):
    """Initialize training (legacy function for standalone use)"""
    feature_list, labels = load_data(folderPath)

    if len(feature_list) == 0:
        print("No data processed. Check your folder structure.")
        return

    # Show dataset statistics
    label_counts = {label: labels.count(label) for label in set(labels)}
    print(f"Dataset statistics:")
    print(f"Total samples: {len(feature_list)}")
    print(f"Label distribution: {label_counts}")

    # Prepare and train
    X, y, feature_names = prepare_training_data(feature_list, labels)
    train_model(X, y, model_type)

    # Save the model with readable files
    model_filename = f'{model_type}/tool_monitor_{model_type}.pkl'
    saved_filepath = save_model(model_filename, feature_list, saveReadable=True)
    print(f"Training complete! Model saved as {saved_filepath}")


# Legacy functions for backward compatibility
def get_model():
    """Get current condition model"""
    return model


def get_scaler():
    """Get current condition scaler"""
    return scaler


def get_feature_names():
    """Get current condition feature names"""
    return feature_names


def get_training_metadata():
    """Get current condition training metadata"""
    return training_metadata


def test_print():
    signal = load_file('res/balans_2.csv')

    xfreq, xmag = signal_FFT(signal.x)
    yfreq, ymag = signal_FFT(signal.y)
    zfreq, zmag = signal_FFT(signal.z)

    xnorm = signal_normalization(xmag)
    ynorm = signal_normalization(ymag)
    znorm = signal_normalization(zmag)

    plt.figure("Analiza sygnału")
    plt.subplot(3, 3, 1)
    plt.plot(signal.t, signal.x)
    plt.grid()
    plt.subplot(3, 3, 2)
    plt.plot(signal.t, signal.y)
    plt.grid()
    plt.subplot(3, 3, 3)
    plt.plot(signal.t, signal.z)
    plt.grid()
    plt.subplot(3, 3, 4)
    plt.plot(xfreq, xmag)
    plt.grid()
    plt.subplot(3, 3, 5)
    plt.plot(yfreq, ymag)
    plt.grid()
    plt.subplot(3, 3, 6)
    plt.plot(zfreq, zmag)
    plt.grid()
    plt.subplot(3, 3, 7)
    plt.plot(xfreq, xnorm)
    plt.grid()
    plt.subplot(3, 3, 8)
    plt.plot(yfreq, ynorm)
    plt.grid()
    plt.subplot(3, 3, 9)
    plt.plot(zfreq, znorm)
    plt.grid()

# Example usage (commented out for integration)
# Main program
# init_training('res/data2')
# init_training('res/data','svm')
# predict = init_prediction('res/model/random_forest/tool_monitor_random_forest.pkl')
# result = predict('res/lozysko_2.csv')
# print(result)