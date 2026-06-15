from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# Absolute imports across project space
from src.monitoring.store_predictions import update_store
from src.monitoring.drift_detector import calculate_drift

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "mlops_drift_monitoring",
    default_args=default_args,
    description="Batch process production telemetry and test for feature drift",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2026, 1, 1),
    catchup=False,
) as dag:

    task_update_store = PythonOperator(
        task_id="update_monitoring_store",
        python_callable=update_store,
    )

    task_calculate_drift = PythonOperator(
        task_id="calculate_dataset_drift",
        python_callable=calculate_drift,
    )

    task_update_store >> task_calculate_drift