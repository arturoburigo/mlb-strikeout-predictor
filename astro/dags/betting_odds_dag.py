"""
## Betting Odds Data Pipeline DAG

This DAG runs two scripts in sequence to collect and merge betting odds data:

1. betting_pitcher_odds_today.py - Fetches MLB schedule and betting props data (or generates fake data)
2. get_and_merge_betano_pitchers_odds.py - Scrapes Betano odds and merges with existing data

The DAG runs daily to collect the latest betting odds for MLB pitchers.
"""

from airflow.decorators import dag, task
from pendulum import datetime
import subprocess
import sys
import os
from pathlib import Path
import pendulum
from dotenv import load_dotenv

local_tz = pendulum.timezone("America/Toronto")

@dag(
    'betting_odds_pipeline',
    start_date=datetime(2024, 1, 1, tz=local_tz),
    schedule="@daily",
    catchup=False,
    tags=["betting", "odds", "mlb"],
)
def betting_odds_pipeline():
    
    @task
    def fetch_betting_odds():
        """
        Runs betting_pitcher_odds_today.py to fetch MLB schedule and betting props data.
        This script will use the real API if BETTINGPROS_API_KEY is available.
        """
        try:
            # Get the path to the script
            script_path = Path(__file__).parent.parent / "include" / "Getting Odds" / "betting_pitcher_odds_today.py"
            
            # Load environment variables from .env file in astro directory
            env_path = Path(__file__).parent / ".env"
            load_dotenv(env_path)
            
            # Debug prints
            print("[DAG DEBUG] Script path:", script_path)
            print("[DAG DEBUG] Script exists:", script_path.exists())
            print("[DAG DEBUG] ENV file path:", env_path)
            print("[DAG DEBUG] ENV file exists:", env_path.exists())
            print("[DAG DEBUG] BETTINGPROS_API_KEY:", os.getenv("BETTINGPROS_API_KEY"))
            
            # Set up environment for subprocess
            env = os.environ.copy()
            
            # Run the script
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,  # Set working directory to project root
                env=env  # Pass environment variables
            )
            
            if result.returncode != 0:
                print(f"Script stderr: {result.stderr}")
                raise Exception(f"Script failed with return code {result.returncode}")
            
            print(f"Betting odds script completed successfully. Output: {result.stdout}")
            return "success"
            
        except Exception as e:
            print(f"Error running betting odds script: {str(e)}")
            raise
    
    # Define task dependencies
    betting_odds_result = fetch_betting_odds()

    
    # Set up the dependencies
    betting_odds_result

# Instantiate the DAG
betting_odds_pipeline() 