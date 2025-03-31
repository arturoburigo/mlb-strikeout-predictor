import warnings
import logging
import sys
from datetime import datetime
from pathlib import Path
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

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

def run_pipeline():
    """
    Main execution function to run the entire pipeline in sequence.
    Each step waits for the previous one to complete before proceeding.
    """
    try:
        # Step 1: Get betting odds for today
        logger.info("=== Step 1: Getting betting odds for today ===")
        from scrapping.betting_odds_today import main as get_betting_odds
        get_betting_odds()
        
        # Step 2: Get pitcher data from last season
        logger.info("=== Step 2: Getting pitcher data from last season ===")
        from scrapping.get_pitcher_lastseason import get_pitcher_data
        get_pitcher_data()
        
        # Step 3: Data utilities
        logger.info("=== Step 3: Running data utilities ===")
        from data_utils import load_data
        pitchers_df, k_percentage_df, betting_file_used = load_data()
        
        # Step 4: Feature engineering
        logger.info("=== Step 4: Running feature engineering ===")
        from feature_engineering import engineer_features
        engineered_data = engineer_features(pitchers_df, k_percentage_df)
        
        # Step 5: Model training
        logger.info("=== Step 5: Training model ===")
        from model_training import train_model
        model, model_results = train_model(engineered_data)
        
        # Step 6: Generate predictions
        logger.info("=== Step 6: Generating predictions ===")
        from predictions import process_betting_data
        results_df = process_betting_data(
            model=model,
            pitchers_df=pitchers_df,
            k_percentage_df=k_percentage_df,
            betting_data_path=betting_file_used,
            output_dir='predictions'
        )
        
        # Step 7: Send email with predictions
        logger.info("=== Step 7: Sending email with predictions ===")
        from email_ml_predictions import send_predictions_email
        send_predictions_email(results_df)
        
        logger.info("=== Pipeline completed successfully ===")
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        return False

def schedule_pipeline():
    """
    Schedule the pipeline to run at 10 AM US time every day
    """
    scheduler = BlockingScheduler()
    
    # Schedule the pipeline to run at 10 AM US time (using Eastern Time)
    et_timezone = pytz.timezone('America/New_York')
    scheduler.add_job(
        run_pipeline,
        trigger=CronTrigger(hour=10, minute=0, timezone=et_timezone),
        id='daily_pipeline',
        name='Run ML pipeline daily at 10 AM ET',
        replace_existing=True
    )
    
    logger.info("Pipeline scheduled to run daily at 10 AM ET")
    scheduler.start()

if __name__ == "__main__":
    # Create necessary directories
    Path('predictions').mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    
    # Run the pipeline immediately and then schedule it
    logger.info("Running initial pipeline execution...")
    if run_pipeline():
        logger.info("Initial pipeline execution successful. Starting scheduler...")
        schedule_pipeline()
    else:
        logger.error("Initial pipeline execution failed. Please check the logs.")