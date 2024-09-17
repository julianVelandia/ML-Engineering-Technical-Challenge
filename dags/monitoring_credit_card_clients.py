from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from config.constants import DATA_DIR, RETRAIN_FLAG_FILENAME, NUEVOS_DATOS_FILENAME
from tasks.monitorear_modelo import monitorear_modelo
from tasks.reentrenar_modelo import reentrenar_modelo

with DAG('monitoring_credit_card_clients',
         default_args={'retries': 1},
         description='DAG para monitorear y reentrenar el modelo de clientes de tarjetas de crédito',
         schedule_interval='@daily',
         start_date=datetime(2024, 1, 1),
         catchup=False) as dag:
    monitorear_rendimiento_task = PythonOperator(
        task_id='monitorear_rendimiento',
        python_callable=monitorear_modelo,
    )

    reentrenar_modelo_task = PythonOperator(
        task_id='reentrenar_modelo',
        python_callable=reentrenar_modelo
    )

    monitorear_rendimiento_task >> reentrenar_modelo_task
