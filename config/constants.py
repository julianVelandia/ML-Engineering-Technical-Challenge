import os

BASE_DIR = '/opt/airflow/dags'

UCIREPO_DATASET_ID_CREDIT_CARD_CLIENTS = 350
FEATURES_FILENAME = 'credit_card_clients_features.csv'
TARGETS_FILENAME = 'credit_card_clients_targets.csv'

DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

CLEAN_FEATURES_FILENAME = 'credit_card_clients_features_clean.csv'
CLEAN_TARGETS_FILENAME = 'credit_card_clients_targets_clean.csv'
BEST_MODEL_FILENAME = 'random_forest_best_model.joblib'
BEST_METRICS_FILENAME = 'best_metrics.json'

IMPUTER_FILENAME = 'imputer.joblib'
SCALER_FILENAME = 'scaler.joblib'

INITIAL_METRICS_FILENAME = 'initial_metrics.json'
MODEL_FILENAME = 'random_forest_model.joblib'

REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

EDA_FILE = os.path.join(REPORTS_DIR, 'eda_credit_card_clients.md')

RANDOM_FOREST_MODEL_FILENAME = 'random_forest_best_model.joblib'
NUEVOS_DATOS_FILENAME = 'nuevos_datos_etiquetados.csv'
DATOS_ANTERIORES_FILENAME = 'datos_anteriores_etiquetados.csv'

CURRENT_METRICS_FILENAME = 'current_metrics.json'
RETRAIN_FLAG_FILENAME = 'retrain_flag.txt'

NUEVOS_CLIENTES_FILENAME = os.path.join(DATA_DIR, 'nuevos_clientes.csv')
NUEVOS_CLIENTES_PREDICCION_FILENAME = os.path.join(DATA_DIR, 'nuevos_clientes_con_prediccion.csv')
