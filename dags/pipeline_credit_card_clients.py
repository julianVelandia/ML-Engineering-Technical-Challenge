from datetime import datetime

from airflow import DAG
from airflow.operators.python_operator import PythonOperator

from config.constants import UCIREPO_DATASET_ID_CREDIT_CARD_CLIENTS
from tasks.ajustar_hiperparametros import ajustar_hiperparametros
from tasks.cargar_datos_ucirepo import cargar_datos_ucirepo
from tasks.entrenar_modelo import entrenar_modelo
from tasks.generar_informes import generar_informes
from tasks.limpiar_datos import limpiar_datos



with DAG('pipeline_credit_card_clients',
         default_args={'retries': 1},
         schedule_interval=None,
         start_date=datetime(2025, 1, 1),
         catchup=False) as dag:
    dataset_id = UCIREPO_DATASET_ID_CREDIT_CARD_CLIENTS

    cargar_datos_task = PythonOperator(
        task_id='cargar_datos',
        python_callable=cargar_datos_ucirepo,
        op_kwargs={'dataset_id': dataset_id}
    )

    limpiar_datos_task = PythonOperator(
        task_id='limpiar_datos',
        python_callable=limpiar_datos
    )

    entrenar_modelo_task = PythonOperator(
        task_id='entrenar_modelo',
        python_callable=entrenar_modelo
    )

    ajustar_hiperparametros_task = PythonOperator(
        task_id='ajustar_hiperparametros',
        python_callable=ajustar_hiperparametros
    )

    generar_informes_task = PythonOperator(
        task_id='generar_informes',
        python_callable=generar_informes
    )

    cargar_datos_task >> limpiar_datos_task >> entrenar_modelo_task >> ajustar_hiperparametros_task

    ajustar_hiperparametros_task >> generar_informes_task
