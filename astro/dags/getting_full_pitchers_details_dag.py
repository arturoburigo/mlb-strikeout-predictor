# dags/pitcher_scraper_dag.py
from airflow.decorators import dag
from airflow.operators.python import PythonOperator
from pendulum import datetime
from include.get_pitchers_records.fully_pitchers_record import run_pitcher_scraper
import pendulum

local_tz = pendulum.timezone("America/Toronto")

@dag(
    'getting_full_pitchers_details_dag',
    start_date=datetime(2024, 1, 1, tz=local_tz),
    schedule=None,
    catchup=False,
    tags=["betting", "odds", "mlb"],
)
def getting_full_pitchers_details_dag():
    
    scrape_pitchers_data = PythonOperator(
        task_id='scrape_pitchers_data',
        python_callable=run_pitcher_scraper,
    )

    scrape_pitchers_data

getting_full_pitchers_details_dag()