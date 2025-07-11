import warnings
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz
from scrapping.get_pitcher_data_lastseason import load_pitcher_data
from scrapping.get_pitcher_lastgame import load_last_pitcher_game
from model.model_training import train_model
from scrapping.model.making_predictions import process_betting_data
from feature_engineering import main as engineer_features
from scrapping.betting_pitcher_odds_today import main as get_betting_odds
from data_utils import load_data
from email_ml_predictions import send_prediction_email
from email_ml_results import send_results_email
from utils.upload_results_to_s3_Bucket import upload_game_results
from cleanup_files import cleanup_csv_files
from telegramSender import main as run_telegram_bot
import os
from dotenv import load_dotenv
import pandas as pd
import glob
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

# Get schedule times from environment variables
DATA_PIPELINE_HOUR = int(os.getenv('DATA_PIPELINE_HOUR', 19))
DATA_PIPELINE_MINUTE = int(os.getenv('DATA_PIPELINE_MINUTE', 10))
EMAIL_HOUR = int(os.getenv('EMAIL_HOUR', 22))
EMAIL_MINUTE = int(os.getenv('EMAIL_MINUTE', 40))
RESULTS_EMAIL_HOUR = int(os.getenv('RESULTS_EMAIL_HOUR', 11))
RESULTS_EMAIL_MINUTE = int(os.getenv('RESULTS_EMAIL_MINUTE', 42))
AWS_UPLOAD_HOUR = int(os.getenv('AWS_UPLOAD_HOUR', 12))
AWS_UPLOAD_MINUTE = int(os.getenv('AWS_UPLOAD_MINUTE', 0))
CLEANUP_HOUR = int(os.getenv('CLEANUP_HOUR', 13))
CLEANUP_MINUTE = int(os.getenv('CLEANUP_MINUTE', 0))
TELEGRAM_HOUR = int(os.getenv('TELEGRAM_HOUR', 22))
TELEGRAM_MINUTE = int(os.getenv('TELEGRAM_MINUTE', 45))

# Debug logging for environment variables
logger.info("=== Environment Variables Debug ===")
logger.info(f"DATA_PIPELINE_HOUR from env: {os.getenv('DATA_PIPELINE_HOUR')}")
logger.info(f"DATA_PIPELINE_MINUTE from env: {os.getenv('DATA_PIPELINE_MINUTE')}")
logger.info(f"EMAIL_HOUR from env: {os.getenv('EMAIL_HOUR')}")
logger.info(f"EMAIL_MINUTE from env: {os.getenv('EMAIL_MINUTE')}")
logger.info(f"RESULTS_EMAIL_HOUR from env: {os.getenv('RESULTS_EMAIL_HOUR')}")
logger.info(f"RESULTS_EMAIL_MINUTE from env: {os.getenv('RESULTS_EMAIL_MINUTE')}")
logger.info(f"AWS_UPLOAD_HOUR from env: {os.getenv('AWS_UPLOAD_HOUR')}")
logger.info(f"AWS_UPLOAD_MINUTE from env: {os.getenv('AWS_UPLOAD_MINUTE')}")
logger.info(f"CLEANUP_HOUR from env: {os.getenv('CLEANUP_HOUR')}")
logger.info(f"CLEANUP_MINUTE from env: {os.getenv('CLEANUP_MINUTE')}")
logger.info(f"TELEGRAM_HOUR from env: {os.getenv('TELEGRAM_HOUR')}")
logger.info(f"TELEGRAM_MINUTE from env: {os.getenv('TELEGRAM_MINUTE')}")
logger.info(f"Final values:")
logger.info(f"DATA_PIPELINE: {DATA_PIPELINE_HOUR:02d}:{DATA_PIPELINE_MINUTE:02d}")
logger.info(f"EMAIL: {EMAIL_HOUR:02d}:{EMAIL_MINUTE:02d}")
logger.info(f"RESULTS_EMAIL: {RESULTS_EMAIL_HOUR:02d}:{RESULTS_EMAIL_MINUTE:02d}")
logger.info(f"AWS_UPLOAD: {AWS_UPLOAD_HOUR:02d}:{AWS_UPLOAD_MINUTE:02d}")
logger.info(f"CLEANUP: {CLEANUP_HOUR:02d}:{CLEANUP_MINUTE:02d}")
logger.info(f"TELEGRAM: {TELEGRAM_HOUR:02d}:{TELEGRAM_MINUTE:02d}")

# Suppress warnings
warnings.filterwarnings('ignore')


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
        
        # Get today's date for the prediction file
        today_date = datetime.now().strftime("%Y-%m-%d")
        csv_filename = f"predicted_{today_date}.csv"
        
        # Get email configuration from environment variables
        destination_email = os.getenv("RECEIVER_EMAIL")
        cc_emails = os.getenv("CC_EMAILS", "").split(",")
        cc_emails = [email.strip() for email in cc_emails if email.strip()]
        
        if not destination_email:
            logger.error("RECEIVER_EMAIL environment variable not set")
            return False
            
        if not os.path.exists(csv_filename):
            logger.error(f"Today's prediction file not found: {csv_filename}")
            return False
            
        # Read the predictions file
        predictions_df = pd.read_csv(csv_filename)
        
        # Send the email
        success = send_prediction_email(
            receiver_email=destination_email,
            predictions_df=predictions_df,
            csv_path=csv_filename,
            cc_emails=cc_emails
        )
        
        if success:
            logger.info("=== Email sent successfully ===")
        else:
            logger.error("=== Failed to send email ===")
            
        return success
    except Exception as e:
        logger.error(f"Email pipeline failed with error: {str(e)}", exc_info=True)
        return False

def run_results_pipeline():
    """
    Function to get pitcher data and send results email
    """
    try:
        # First, get pitcher data
        logger.info("=== Getting pitcher data from last game ===")
        load_last_pitcher_game()
        
        # Then send results email
        logger.info("=== Sending email with results analysis ===")
        
        # Get yesterday's date for the files
        today_date = datetime.now()
        yesterday_date = today_date - pd.Timedelta(days=1)
        yesterday_date_str = yesterday_date.strftime("%Y-%m-%d")
        
        predictions_filename = f"predicted_{yesterday_date_str}.csv"
        results_filename = f"game_results_{yesterday_date_str}.csv"
        
        if not os.path.exists(predictions_filename) or not os.path.exists(results_filename):
            logger.error(f"Required files not found for {yesterday_date_str}")
            logger.error(f"Looking for: {predictions_filename} and {results_filename}")
            return False
        
        predictions_df = pd.read_csv(predictions_filename)
        results_df = pd.read_csv(results_filename)
        
        destination_email = os.getenv("RECEIVER_EMAIL")
        cc_emails = os.getenv("CC_EMAILS", "").split(",")
        cc_emails = [email.strip() for email in cc_emails if email.strip()]
        
        success = send_results_email(
            receiver_email=destination_email,
            predictions_df=predictions_df,
            results_df=results_df,
            cc_emails=cc_emails
        )
        
        if success:
            logger.info("=== Results email sent successfully ===")
        else:
            logger.error("=== Failed to send results email ===")
            
        return success
    except Exception as e:
        logger.error(f"Results pipeline failed with error: {str(e)}", exc_info=True)
        return False

def run_aws_upload():
    """
    Function to upload game results to AWS S3
    """
    try:
        logger.info("=== Uploading game results to AWS S3 ===")
        success = upload_game_results()
        if success:
            logger.info("=== AWS upload completed successfully ===")
        else:
            logger.error("=== Failed to upload to AWS S3 ===")
        return success
    except Exception as e:
        logger.error(f"AWS upload failed with error: {str(e)}", exc_info=True)
        return False

def run_cleanup():
    """
    Function to clean up CSV files
    """
    try:
        logger.info("=== Running CSV file cleanup ===")
        success = cleanup_csv_files()
        if success:
            logger.info("=== Cleanup completed successfully ===")
        else:
            logger.error("=== Failed to clean up files ===")
        return success
    except Exception as e:
        logger.error(f"Cleanup failed with error: {str(e)}", exc_info=True)
        return False

def run_telegram_pipeline():
    """
    Run the Telegram bot to send MLB predictions
    """
    try:
        logger.info("=== Running Telegram notification pipeline ===")
        run_telegram_bot()
        logger.info("=== Telegram notification pipeline completed successfully ===")
    except Exception as e:
        logger.error(f"Error in Telegram notification pipeline: {str(e)}")

def schedule_pipeline():
    """
    Schedule the pipeline to run at configured times in US time every day
    """
    scheduler = BlockingScheduler()
    
    # Schedule the data pipeline at configured time (using Eastern Time)
    et_timezone = pytz.timezone('America/New_York')
    
    # Data pipeline
    scheduler.add_job(
        run_data_pipeline,
        trigger=CronTrigger(hour=DATA_PIPELINE_HOUR, minute=DATA_PIPELINE_MINUTE, timezone=et_timezone),
        id='daily_data_pipeline',
        name=f'Run ML data pipeline daily at {DATA_PIPELINE_HOUR:02d}:{DATA_PIPELINE_MINUTE:02d} ET',
        replace_existing=True
    )
    
    # Predictions email
    scheduler.add_job(
        run_email_pipeline,
        trigger=CronTrigger(hour=EMAIL_HOUR, minute=EMAIL_MINUTE, timezone=et_timezone),
        id='daily_email_pipeline',
        name=f'Send predictions email daily at {EMAIL_HOUR:02d}:{EMAIL_MINUTE:02d} ET',
        replace_existing=True
    )
    
    # Results pipeline
    scheduler.add_job(
        run_results_pipeline,
        trigger=CronTrigger(hour=RESULTS_EMAIL_HOUR, minute=RESULTS_EMAIL_MINUTE, timezone=et_timezone),
        id='daily_results_pipeline',
        name=f'Run results pipeline daily at {RESULTS_EMAIL_HOUR:02d}:{RESULTS_EMAIL_MINUTE:02d} ET',
        replace_existing=True
    )
    
    # AWS upload
    scheduler.add_job(
        run_aws_upload,
        trigger=CronTrigger(hour=AWS_UPLOAD_HOUR, minute=AWS_UPLOAD_MINUTE, timezone=et_timezone),
        id='daily_aws_upload',
        name=f'Upload to AWS daily at {AWS_UPLOAD_HOUR:02d}:{AWS_UPLOAD_MINUTE:02d} ET',
        replace_existing=True
    )
    
    # File cleanup
    scheduler.add_job(
        run_cleanup,
        trigger=CronTrigger(hour=CLEANUP_HOUR, minute=CLEANUP_MINUTE, timezone=et_timezone),
        id='daily_cleanup',
        name=f'Run cleanup daily at {CLEANUP_HOUR:02d}:{CLEANUP_MINUTE:02d} ET',
        replace_existing=True
    )
    
    # Telegram notification
    scheduler.add_job(
        run_telegram_pipeline,
        trigger=CronTrigger(hour=TELEGRAM_HOUR, minute=TELEGRAM_MINUTE, timezone=et_timezone),
        id='daily_telegram_pipeline',
        name=f'Send Telegram notification daily at {TELEGRAM_HOUR:02d}:{TELEGRAM_MINUTE:02d} ET',
        replace_existing=True
    )
    
    logger.info(f"Pipeline scheduled to run daily at {DATA_PIPELINE_HOUR:02d}:{DATA_PIPELINE_MINUTE:02d} ET")
    logger.info(f"Predictions email will be sent at {EMAIL_HOUR:02d}:{EMAIL_MINUTE:02d} ET")
    logger.info(f"Results pipeline will run at {RESULTS_EMAIL_HOUR:02d}:{RESULTS_EMAIL_MINUTE:02d} ET")
    logger.info(f"AWS upload will run at {AWS_UPLOAD_HOUR:02d}:{AWS_UPLOAD_MINUTE:02d} ET")
    logger.info(f"Cleanup will run at {CLEANUP_HOUR:02d}:{CLEANUP_MINUTE:02d} ET")
    logger.info(f"Telegram notification will be sent at {TELEGRAM_HOUR:02d}:{TELEGRAM_MINUTE:02d} ET")
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
    
    # Calculate next run times for all jobs
    next_data_pipeline = get_next_run_time(current_time, DATA_PIPELINE_HOUR, DATA_PIPELINE_MINUTE)
    next_email = get_next_run_time(current_time, EMAIL_HOUR, EMAIL_MINUTE)
    next_results_email = get_next_run_time(current_time, RESULTS_EMAIL_HOUR, RESULTS_EMAIL_MINUTE)
    next_aws_upload = get_next_run_time(current_time, AWS_UPLOAD_HOUR, AWS_UPLOAD_MINUTE)
    next_cleanup = get_next_run_time(current_time, CLEANUP_HOUR, CLEANUP_MINUTE)
    next_telegram = get_next_run_time(current_time, TELEGRAM_HOUR, TELEGRAM_MINUTE)
    
    # Calculate wait times
    wait_seconds_data = (next_data_pipeline - current_time).total_seconds()
    wait_hours_data = wait_seconds_data / 3600
    
    wait_seconds_email = (next_email - current_time).total_seconds()
    wait_hours_email = wait_seconds_email / 3600
    
    wait_seconds_results = (next_results_email - current_time).total_seconds()
    wait_hours_results = wait_seconds_results / 3600
    
    wait_seconds_aws = (next_aws_upload - current_time).total_seconds()
    wait_hours_aws = wait_seconds_aws / 3600
    
    wait_seconds_cleanup = (next_cleanup - current_time).total_seconds()
    wait_hours_cleanup = wait_seconds_cleanup / 3600
    
    wait_seconds_telegram = (next_telegram - current_time).total_seconds()
    wait_hours_telegram = wait_seconds_telegram / 3600
    
    logger.info(f"Waiting until {next_data_pipeline.strftime('%Y-%m-%d %H:%M')} ET to start the pipeline...")
    logger.info(f"Approximately {wait_hours_data:.1f} hours until next pipeline run")
    logger.info(f"Email will be sent at {next_email.strftime('%Y-%m-%d %H:%M')} ET")
    logger.info(f"Approximately {wait_hours_email:.1f} hours until next email")
    logger.info(f"Results email will be sent at {next_results_email.strftime('%Y-%m-%d %H:%M')} ET")
    logger.info(f"Approximately {wait_hours_results:.1f} hours until next results email")
    logger.info(f"AWS upload will run at {next_aws_upload.strftime('%Y-%m-%d %H:%M')} ET")
    logger.info(f"Approximately {wait_hours_aws:.1f} hours until next AWS upload")
    logger.info(f"Cleanup will run at {next_cleanup.strftime('%Y-%m-%d %H:%M')} ET")
    logger.info(f"Approximately {wait_hours_cleanup:.1f} hours until next cleanup")
    logger.info(f"Telegram notification will be sent at {next_telegram.strftime('%Y-%m-%d %H:%M')} ET")
    logger.info(f"Approximately {wait_hours_telegram:.1f} hours until next telegram notification")
    
    # Start the scheduler which will wait until the scheduled time
    schedule_pipeline()