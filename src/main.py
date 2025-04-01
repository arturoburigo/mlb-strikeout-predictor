import warnings
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz
from scrapping.get_pitcher_lastseason import load_pitcher_data
from model_training import train_model
from predictions import process_betting_data
from feature_engineering import main as engineer_features
from scrapping.betting_odds_today import main as get_betting_odds
from data_utils import load_data
from email_ml_predictions import send_prediction_email
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Get schedule times from environment variables
DATA_PIPELINE_HOUR = int(os.getenv('DATA_PIPELINE_HOUR', 10))
DATA_PIPELINE_MINUTE = int(os.getenv('DATA_PIPELINE_MINUTE', 0))
EMAIL_HOUR = int(os.getenv('EMAIL_HOUR', 12))
EMAIL_MINUTE = int(os.getenv('EMAIL_MINUTE', 0))

def run_data_pipeline():
    """
    Main execution function to run the data processing pipeline in sequence.
    Each step waits for the previous one to complete before proceeding.
    """
    try:
        # Step 1: Get betting odds for today
        logger.info("=== Step 1: Getting betting odds for today ===")
        get_betting_odds()
        
        # Step 2: Get pitcher data from last season
        logger.info("=== Step 2: Getting pitcher data from last season ===")
        load_pitcher_data()
        
        # Step 3: Data utilities
        logger.info("=== Step 3: Running data utilities ===")
        pitchers_df, k_percentage_df, betting_file_used = load_data()
        logger.info(f"Loaded data for {pitchers_df['Pitcher'].nunique()} pitchers")
        
        # Step 4: Feature engineering
        logger.info("=== Step 4: Running feature engineering ===")
        engineered_data = engineer_features(pitchers_df, k_percentage_df)
        logger.info(f"Engineered data contains {len(engineered_data)} rows with features")
        
        # Step 5: Model training - passing both original dataframes to the updated train_model function
        logger.info("=== Step 5: Training model ===")
        model, model_results = train_model(pitchers_df, k_percentage_df)
        
        # Step 6: Generate predictions
        logger.info("=== Step 6: Generating predictions ===")
        results_df = process_betting_data(
            model=model,
            pitchers_df=pitchers_df,
            k_percentage_df=k_percentage_df,
            betting_data_path=betting_file_used,
        )
        
        logger.info("=== Data pipeline completed successfully ===")
        return True
        
    except Exception as e:
        logger.error(f"Data pipeline failed with error: {str(e)}", exc_info=True)
        return False

def run_email_pipeline():
    """
    Function to send the email with predictions
    """
    try:
        logger.info("=== Sending email with predictions ===")
        send_prediction_email()
        logger.info("=== Email sent successfully ===")
        return True
    except Exception as e:
        logger.error(f"Email pipeline failed with error: {str(e)}", exc_info=True)
        return False

def schedule_pipeline():
    """
    Schedule the pipeline to run at configured times in US time every day
    """
    scheduler = BlockingScheduler()
    
    # Schedule the data pipeline at configured time (using Eastern Time)
    et_timezone = pytz.timezone('America/New_York')
    scheduler.add_job(
        run_data_pipeline,
        trigger=CronTrigger(hour=DATA_PIPELINE_HOUR, minute=DATA_PIPELINE_MINUTE, timezone=et_timezone),
        id='daily_data_pipeline',
        name=f'Run ML data pipeline daily at {DATA_PIPELINE_HOUR:02d}:{DATA_PIPELINE_MINUTE:02d} ET',
        replace_existing=True
    )
    
    # Schedule the email at configured time
    scheduler.add_job(
        run_email_pipeline,
        trigger=CronTrigger(hour=EMAIL_HOUR, minute=EMAIL_MINUTE, timezone=et_timezone),
        id='daily_email_pipeline',
        name=f'Send email daily at {EMAIL_HOUR:02d}:{EMAIL_MINUTE:02d} ET',
        replace_existing=True
    )
    
    logger.info(f"Pipeline scheduled to run daily at {DATA_PIPELINE_HOUR:02d}:{DATA_PIPELINE_MINUTE:02d} ET and send email at {EMAIL_HOUR:02d}:{EMAIL_MINUTE:02d} ET")
    scheduler.start()

def get_next_run_time(current_time, target_hour, target_minute):
    """
    Calculate the next run time for a given target hour and minute
    """
    target_time = current_time.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
    
    # If current time is past target time, set target to tomorrow
    if current_time > target_time:
        target_time = target_time + timedelta(days=1)
    
    return target_time

if __name__ == "__main__":
    # Create necessary directories
    Path('logs').mkdir(exist_ok=True)
    
    # Get current time in Eastern Time
    et_timezone = pytz.timezone('America/New_York')
    current_time = datetime.now(et_timezone)
    
    # Calculate next run times for both jobs
    next_data_pipeline = get_next_run_time(current_time, DATA_PIPELINE_HOUR, DATA_PIPELINE_MINUTE)
    next_email = get_next_run_time(current_time, EMAIL_HOUR, EMAIL_MINUTE)
    
    # Calculate wait times
    wait_seconds_data = (next_data_pipeline - current_time).total_seconds()
    wait_hours_data = wait_seconds_data / 3600
    
    wait_seconds_email = (next_email - current_time).total_seconds()
    wait_hours_email = wait_seconds_email / 3600
    
    logger.info(f"Waiting until {next_data_pipeline.strftime('%Y-%m-%d %H:%M')} ET to start the pipeline...")
    logger.info(f"Approximately {wait_hours_data:.1f} hours until next pipeline run")
    logger.info(f"Email will be sent at {next_email.strftime('%Y-%m-%d %H:%M')} ET")
    logger.info(f"Approximately {wait_hours_email:.1f} hours until next email")
    
    # Start the scheduler which will wait until the scheduled time
    schedule_pipeline()