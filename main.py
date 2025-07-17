#imports
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas as pn
from scipy.fft import fft,fftfreq
from scipy.signal import butter,filtfilt
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datetime import datetime
from scipy import stats
import pickle
import json
from tqdm import tqdm

model = None
scaler = None
feature_names = []
training_metadata = {}

def load_file(filename):
    try:
        file = pn.read_csv(filename, header=None, names=["t", "x", "y", "z"])

        return file.sort_values('t')
    except Exception as e:
        print(f'Error Loading {filename}: {e}')
        return None

def load_features(filename):
        try:
            file = pn.read_csv(filename, header=None, names= ['mean_x', 'std_x', 'rms_x', 'p2p_x', 'if_x', 'skewness_x', 'kurtosis_x', 'crest_factor_x', 'shape_factor_x',
                                'mean_y', 'std_y', 'rms_y', 'p2p_y', 'if_y', 'skewness_y', 'kurtosis_y', 'crest_factor_y', 'shape_factor_y',
                                'mean_z', 'std_z', 'rms_z', 'p2p_z', 'if_z', 'skewness_z', 'kurtosis_z', 'crest_factor_z', 'shape_factor_z'])
            required_columns = ['mean_x', 'std_x', 'rms_x', 'p2p_x', 'if_x', 'skewness_x', 'kurtosis_x', 'crest_factor_x', 'shape_factor_x',
                                'mean_y', 'std_y', 'rms_y', 'p2p_y', 'if_y', 'skewness_y', 'kurtosis_y', 'crest_factor_y', 'shape_factor_y',
                                'mean_z', 'std_z', 'rms_z', 'p2p_z', 'if_z', 'skewness_z', 'kurtosis_z', 'crest_factor_z', 'shape_factor_z']

            if not all(col in file.columns for col in required_columns):
                print(f"Error: CSV must contain columns: {required_columns}")
                return None

            return file.sort_values('mean_x')
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

def load_data(folderPath):
    allFiles = []

    failedFile = []
    featureList = []
    labels = []

    for subfolder in os.listdir(folderPath):
        subfolderPath = os.path.join(folderPath,subfolder)

        if os.path.isdir(subfolderPath):
            csvFiles = [f for f in os.listdir(subfolderPath) if f.endswith('.csv')]

            for filename in csvFiles:
                filePath = os.path.join(subfolderPath, filename)
                allFiles.append((filePath, subfolder))

    #Process

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

    return featureList, labels

def signal_FFT(data):
    windowed_signal = data * np.hanning(len(data))
    N = len(data)
    fp = 1/20000
    yf = fft(windowed_signal)
    xf = fftfreq(N,fp)
    positive_idx = xf >= 0
    frequencies = xf[positive_idx]
    magnitude = np.abs(yf[positive_idx])

    return frequencies,magnitude

'''
    Filtr do sygnału
        data        = sygnal do filtracji
        fs          = czest. probkowania
        freqLimit   = czest. odciecia
        filterType  = typ filtra // low lub lp - dolnoprzepustowy, high lub hp - gornoprzepustowy
        order       = rzad filtra
'''
def signal_filter(data,fs,freqLimit, filterType, order):
    Wn = freqLimit/ (fs/2)
    b,a = butter(order, Wn, btype=filterType, analog=False)
    return filtfilt(b,a,data)

def signal_normalization(data):
    #return (data - np.min(data)) / (np.max(data) - np.min(data))
    #return (data - np.mean(data)) / np.std(data)
    return data / np.max(data)
def signal_average(data):
    return np.mean(data)

def signal_std_deviation(data):
    return np.std(data)

def signal_RMS(data):
    return  np.sqrt(signal_average(data**2))

def signal_P2P(data):
    return np.max(data) - np.min(data)

def signal_IF(data):
    return np.max(data) / signal_average(data)

def signal_skewness(data):
    return skew(data,axis=0,bias=True)

def signal_kurtosis(data):
    return kurtosis(data,axis=0,bias=True)

def signal_crestFactor(data):
    return abs(np.max(data)) / signal_RMS(data)

def signal_shapeFactor(data):
    return 1/signal_average(data)

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
def save_features(data):
    df = pd.DataFrame(data).to_csv('res/features.csv', index=None, header=None)

def moving_average(data, window_size=5):
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

    fig, axs = plt.subplots(3,3, figsize=(15,8), dpi=100)
    axs = axs.flatten()

    startBear = labels.count('bearing')
    startMisal = startBear + labels.count('misalignment')
    startNorm = startMisal + labels.count('normal')
    startUnbal =  startNorm + labels.count('unbalance')
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

def plot_axis_features_from_file(axis,feature):
    plotsNames = ['mean', 'std', 'rms', 'p2p', 'if', 'skewness', 'kurtosis', 'crest_factor', 'shape_factor']

    column_names = [f'{name}_{axis}' for name in plotsNames]
    data = feature[column_names]

    segment_size = 1000

    segments = {
        'Bearing': data.iloc[0:segment_size],
        'Unbalance': data.iloc[segment_size:2 * segment_size],
        'Normal': data.iloc[2 * segment_size:3 * segment_size],
        'Misalignment': data.iloc[3 * segment_size:4 * segment_size],
    }

    fig, axes = plt.subplots(3, 3, figsize=(14, 9))
    axes = axes.ravel()

    for i, feature in enumerate(column_names):
        ax = axes[i]
        x_vals = np.arange(segment_size)

        for label, segment in segments.items():
            smoothed = moving_average(segment[feature].values)
            ax.plot(x_vals, smoothed, label=label)

        ax.set_title(feature.replace(f"_{axis}", "").capitalize())
        ax.grid(True)

    axes[0].legend(loc='upper right', fontsize='small')
    fig.suptitle(f'Porównanie cech dla osi {axis.upper()}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def prepare_training_data(features, labels):
    global scaler, feature_names
    dfFeature = pd.DataFrame(features)
    feature_names = dfFeature.columns.tolist();

    dfFeature = dfFeature.fillna(dfFeature.mean())

    scaler = StandardScaler()
    XScaled = scaler.fit_transform(dfFeature)

    labelMap = {'normal': 0, 'unbalance': 1, 'misalignment': 2, 'bearing': 3}
    y = np.array([labelMap[label] for label in labels])

    return XScaled, y, feature_names

def train_model(X,y, modelType='random_forest'):

    global model, training_metadata

    if modelType == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,  # Number of trees
            max_depth=10,  # Maximum tree depth
            random_state=42,  # For reproducible results
            class_weight='balanced'  # Handle unbalanced classes
        )
    else:
        raise ValueError("Only 'random_forest' supported in simplified version")

    # Split data for training and testing
    XTrain, XTest, yTrain, yTest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train the model
    print("Training model...")
    model.fit(XTrain, yTrain)

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

    print("Model Performance:")
    print(classification_report(yTest, yPred,
                                target_names=['normal', 'unbalance', 'misalignment', 'bearing']))

    return model

def save_model(filepath='trained_model.pkl', saveReadable=True):
    global model, scaler, feature_names, training_metadata

    if model is None:
        raise ValueError("No model to save. Train model first.")

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
    print(f"Model saved to {filepath}")

    if saveReadable:
        base_name = os.path.splitext(filepath)[0]

        # 1. Save feature importance to CSV
        if hasattr(model, 'feature_importances_'):
            feature_importance_df = pd.DataFrame({
                'feature_name': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            importance_path = f"{base_name}_feature_importance.csv"
            feature_importance_df.to_csv(importance_path, index=False)
            print(f"Feature importance saved to {importance_path}")

        # 2. Save scaler parameters to CSV
        if scaler is not None:
            scaler_df = pd.DataFrame({
                'feature_name': feature_names,
                'mean': scaler.mean_,
                'scale': scaler.scale_,
                'variance': scaler.var_
            })

            scaler_path = f"{base_name}_scaler_params.csv"
            scaler_df.to_csv(scaler_path, index=False)
            print(f"Scaler parameters saved to {scaler_path}")

        # 3. Save model metadata to JSON
        metadata_path = f"{base_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(training_metadata, f, indent=2)
        print(f"Model metadata saved to {metadata_path}")

        # 4. Save model summary to text file
        summary_path = f"{base_name}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("TOOL CONDITION MONITOR - MODEL SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Training Date: {training_metadata.get('training_date', 'Unknown')}\n")
            f.write(f"Model Type: {training_metadata.get('model_type', 'Unknown')}\n")
            f.write(f"Training Samples: {training_metadata.get('training_samples', 'Unknown')}\n")
            f.write(f"Test Samples: {training_metadata.get('test_samples', 'Unknown')}\n")
            f.write(f"Feature Count: {training_metadata.get('feature_count', 'Unknown')}\n\n")

            f.write("FEATURE NAMES:\n")
            for i, name in enumerate(feature_names, 1):
                f.write(f"{i:2d}. {name}\n")

            f.write("\nMODEL PERFORMANCE:\n")
            if 'classification_report' in training_metadata:
                report = training_metadata['classification_report']
                f.write(f"Overall Accuracy: {report['accuracy']:.3f}\n\n")

                for class_name in ['Good', 'Warning', 'Faulty']:
                    class_key = class_name.lower()
                    if class_key in report:
                        f.write(f"{class_name}:\n")
                        f.write(f"  Precision: {report[class_key]['precision']:.3f}\n")
                        f.write(f"  Recall: {report[class_key]['recall']:.3f}\n")
                        f.write(f"  F1-Score: {report[class_key]['f1-score']:.3f}\n")
                        f.write(f"  Support: {report[class_key]['support']}\n\n")

        print(f"Model summary saved to {summary_path}")

def load_model(filepath='trained_tool_monitor.pkl'):
    global model, scaler, feature_names, training_metadata

    try:
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']
        training_metadata = model_data.get('training_metadata', {})

        print(f"Model loaded from {filepath}")
        print(f"Trained on: {training_metadata.get('training_date', 'Unknown')}")
        print(f"Features: {len(feature_names)}")

    except FileNotFoundError:
        print(f"Model file {filepath} not found!")
    except Exception as e:
        print(f"Error loading model: {e}")

def predict_condition(filepath):
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
    condition_map = {0: 'normal', 1 : 'unbalance' , 2: 'misalignment' , 3 : 'bearing'}
    predicted_condition = condition_map[prediction]

    # Generate recommendations
    recommendations = []
    if predicted_condition == 'normal':
        recommendations = [
            "DObra kondycja",
        ]
    elif predicted_condition == 'unbalance':
        recommendations = [
            "Łożysko wymaga sprawdzenia",
            "Należy sprwadzić balans łożyska",
            "wykonać kolejny pomiar za 1-2 tygodnie",
        ]
    elif predicted_condition == 'misalignment':
        recommendations = [
            "Łożysko wymaga sprawdzenia",
            "Należy sprwadzić połączenia",
            "wykonać kolejny pomiar za 1-2 tygodnie",
        ]
    elif predicted_condition == 'bearing':
        recommendations = [
            "Zepsute, natychmiastowa wymiana",
        ]

    return {
        'condition': predicted_condition,
        'confidence': float(np.max(probabilities)),
        'probabilities': {
            'good': float(probabilities[0]),
            'warning': float(probabilities[1]),
            'faulty': float(probabilities[2])
        },
        'recommendations': recommendations,
        'extracted_features': features,
        'prediction_timestamp': datetime.now().isoformat()
    }

def init_prediction(filepath='trained_tool_monitor.pkl'):
    load_model(filepath)

    print("Model loaded and ready for predictions!")

    # Return prediction function
    return predict_condition

def init_training(folderPath):
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
    train_model(X, y, 'random_forest')

    # Save the model with readable files
    save_model('trained_tool_monitor.pkl', saveReadable=True)

    print("Training complete!")

# Main program
# signal = load_file('res/normalne11.csv')
#
# xFFT_freq,xFFT_mag = signal_FFT(signal.x)
# yFFT_freq,yFFT_mag = signal_FFT(signal.y)
# zFFT_freq,zFFT_mag = signal_FFT(signal.z)
#
#
# xFiltr = signal_filter(signal.x, 20000, 500,'lp',10)
# yFiltr = signal_filter(signal.y, 20000, 500,'lp',10)
# zFiltr = signal_filter(signal.z, 20000, 500,'lp',10)
#
#
# # FFT po filtracji
# xFiltrFFT,txFiltrFFT = signal_FFT(xFiltr)
# yFiltrFFT,tyFiltrFFT = signal_FFT(yFiltr)
# zFiltrFFT,tzFiltrFFT = signal_FFT(zFiltr)
#
# # Normalizacja
# xNorm = signal_normalization(xFFT_mag)
# yNorm = signal_normalization(yFFT_mag)
# zNorm = signal_normalization(zFFT_mag)



#init_training('res/data')

predict = init_prediction('trained_tool_monitor.pkl')

result = predict('res/os_3.csv')
print(result)




# features, labels = load_data('res/data/')
# save_features(features)



# features2 = load_features('res/features.csv')
# labels = features.columns.tolist()
# features = features.to_numpy()
# print(features2)

# plot_axis_features('x',features,labels)
#plot_axis_features_from_file('x',features2)

#print(xFeatures)
# print("Features =================================")
#print(features)
# print("Labels =================================")
#print(labels)

# xFeatures = plot_axis_features('x', features, labels)
print("")

