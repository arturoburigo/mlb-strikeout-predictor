"""
## Betting Odds Data Pipeline DAG

This DAG runs two scripts in sequence to collect and merge betting odds data:

1. betting_pitcher_odds_today.py - Fetches MLB schedule and betting props data
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

local_tz = pendulum.timezone("America/Toronto")

default_args=dict(
    start_date=datetime(2016, 1, 1, tz=local_tz),
    owner='airflow'
)
# Define the basic parameters of the DAG
@dag(
    'betting_odds_pipeline',
    default_args=default_args
)
def betting_odds_pipeline():
    
    @task
    def fetch_betting_odds(execution_date=None):
        """
        Runs betting_pitcher_odds_today.py to fetch MLB schedule and betting props data.
        This script fetches data from BettingPros API and MLB API.
        """
        try:
            # Get the path to the script
            script_path = Path(__file__).parent.parent / "include" / "Getting Odds" / "betting_pitcher_odds_today.py"
            
            # Load environment variables from .env file
            from dotenv import load_dotenv
            import os
            
            # Load .env file from project root
            env_path = Path(__file__).parent.parent.parent / ".env"
            load_dotenv(env_path)
            
            # Debug prints
            print("[DAG DEBUG] CWD:", os.getcwd())
            print("[DAG DEBUG] ENV BETTINGPROS_API_KEY:", os.environ.get("BETTINGPROS_API_KEY"))
            print(f"[DAG DEBUG] Execution date: {execution_date}")
            
            # Get the API key
            api_key = os.getenv("BETTINGPROS_API_KEY")
            if not api_key:
                raise Exception("BETTINGPROS_API_KEY not found in environment")
            
            # Set up environment for subprocess
            env = os.environ.copy()
            env["BETTINGPROS_API_KEY"] = api_key
            if execution_date:
                env["EXECUTION_DATE"] = execution_date
            
            # Run the script
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,  # Set working directory to project root
                env=env  # Pass environment variables
            )
            
            if result.returncode != 0:
                raise Exception(f"Script failed with return code {result.returncode}. Error: {result.stderr}")
            
            print(f"Betting odds script completed successfully. Output: {result.stdout}")
            return "success"
            
        except Exception as e:
            print(f"Error running betting odds script: {str(e)}")
            raise
    
    @task
    def scrape_and_merge_betano_odds():
        """
        Runs get_and_merge_betano_pitchers_odds.py to scrape Betano odds and merge with existing data.
        This script scrapes Betano website and merges the data with the betting data from the previous task.
        """
        try:
            # Get the path to the script
            script_path = Path(__file__).parent.parent / "include" / "Getting Odds" / "get_and_merge_betano_pitchers_odds.py"
            
            # Load environment variables from .env file
            from dotenv import load_dotenv
            import os
            
            # Load .env file from project root
            env_path = Path(__file__).parent.parent.parent / ".env"
            load_dotenv(env_path)
            
            # Set up environment for subprocess
            env = os.environ.copy()
            
            # Run the script with headless mode and merge enabled
            result = subprocess.run(
                [sys.executable, str(script_path), "--no-headless", "--save-debug-csv"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,  # Set working directory to project root
                env=env  # Pass environment variables
            )
            
            if result.returncode != 0:
                raise Exception(f"Script failed with return code {result.returncode}. Error: {result.stderr}")
            
            print(f"Betano odds script completed successfully. Output: {result.stdout}")
            return "success"
            
        except Exception as e:
            print(f"Error running Betano odds script: {str(e)}")
            raise
    
    # Define task dependencies - fetch_betting_odds must complete before scrape_and_merge_betano_odds
    betting_odds_result = fetch_betting_odds()
    betano_odds_result = scrape_and_merge_betano_odds()
    
    # Set up the dependency
    betting_odds_result >> betano_odds_result


# Instantiate the DAG
betting_odds_pipeline() 