from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from config.constants import UCIREPO_DATASET_ID_CREDIT_CARD_CLIENTS
from tasks.cargar_datos_ucirepo import cargar_datos_ucirepo
from tasks.eda_default_credit_card_clients import generar_eda

with DAG('eda_credit_card_clients',
         default_args={'retries': 1},
         description='Pipeline para generar el EDA del dataset de Credit Card Clients',
         schedule_interval='@once',
         start_date=datetime(2024, 1, 1),
         catchup=False) as dag:
    dataset_id = UCIREPO_DATASET_ID_CREDIT_CARD_CLIENTS

    cargar_datos_task = PythonOperator(
        task_id='cargar_datos',
        python_callable=cargar_datos_ucirepo,
        op_kwargs={'dataset_id': dataset_id}
    )

    generar_eda_task = PythonOperator(
        task_id='generar_eda',
        python_callable=generar_eda
    )

    cargar_datos_ucirepo >> generar_eda_task
