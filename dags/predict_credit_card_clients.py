from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from tasks.predecir_nuevos_datos import predecir_nuevos_datos

with DAG('predecir_nuevos_datos',
         default_args={'retries': 1},
         schedule_interval=None,
         start_date=datetime(2025, 1, 1),
         catchup=False) as dag:
    predecir_nuevos_datos_task = PythonOperator(
        task_id='predecir_nuevos_datos',
        python_callable=predecir_nuevos_datos,
    )

    predecir_nuevos_datos_task
