import yaml
import os


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found!")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing config file: {e}")


def get_config_value(config, key_path):
    """Get configuration value using dot notation (e.g., 'models.random_forest.n_estimators')"""
    keys = key_path.split('.')
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            raise KeyError(f"Configuration key '{key_path}' not found!")

    return value


# Load configuration once at module import
CONFIG = load_config()


# Helper functions for common config access patterns
def get_model_params(model_type, model_category):
    """Get model parameters for specific model type and category"""
    return get_config_value(CONFIG, f'models.{model_type}.{model_category}')


def get_rul_params():
    """Get RUL configuration parameters"""
    return get_config_value(CONFIG, 'rul')


def get_training_params():
    """Get training configuration parameters"""
    return get_config_value(CONFIG, 'training')


def get_gui_params():
    """Get GUI configuration parameters"""
    return get_config_value(CONFIG, 'gui')


def get_signal_processing_params():
    """Get signal processing configuration parameters"""
    return get_config_value(CONFIG, 'signal_processing')


def get_recommendations(category):
    """Get recommendations for specific category"""
    return get_config_value(CONFIG, f'recommendations.{category}')


def get_paths():
    """Get file and directory paths"""
    return get_config_value(CONFIG, 'paths')


def get_memory_params():
    """Get memory management parameters"""
    return get_config_value(CONFIG, 'memory')


def get_performance_params():
    """Get performance configuration parameters"""
    return get_config_value(CONFIG, 'performance')