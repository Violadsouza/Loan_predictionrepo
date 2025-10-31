import pandas as pd
from scipy.stats import ks_2samp
import joblib
import logging
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def detect_data_drift(baseline_path, new_path, threshold=0.05):
    baseline = pd.read_csv(baseline_path)
    new_data = pd.read_csv(new_path)

    drifted_features = []
    for col in baseline.columns:
        stat, p_value = ks_2samp(baseline[col], new_data[col])
        if p_value < threshold:
            drifted_features.append(col)
            logging.warning(f'Drift detected in feature: {col} (p={p_value:.4f})')
    if not drifted_features:
        logging.info('No data drift detected.')
    return drifted_features

def monitor_model_performance(model_path, X_test_path, y_test_path, drift_features=[], accuracy_threshold=0.8):
    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.flatten()
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f'Model accuracy: {accuracy:.4f}')
    if accuracy < accuracy_threshold:
        logging.warning(f'Model accuracy below threshold {accuracy_threshold}')
    if drift_features:
        logging.warning(f'Drift detected in features: {drift_features}')

if __name__ == "__main__":
    baseline_path = 'X_train.csv'
    new_data_path = 'X_test.csv'
    model_path = 'best_loan_model.pkl'
    y_test_path = 'y_test.csv'

    drifted_features = detect_data_drift(baseline_path, new_data_path)
    monitor_model_performance(model_path, new_data_path, y_test_path, drifted_features)
