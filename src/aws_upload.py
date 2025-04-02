import os
import boto3
import pandas as pd
from datetime import datetime
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def upload_game_results():
    """
    Upload yesterday's game results to AWS S3 bucket.
    Returns True if successful, False otherwise.
    """
    try:
        # Get yesterday's date for the file
        today_date = datetime.now()
        yesterday_date = today_date - pd.Timedelta(days=1)
        yesterday_date_str = yesterday_date.strftime("%Y-%m-%d")
        game_results_filename = f"game_results_{yesterday_date_str}.csv"
        
        # Check if file exists
        if not os.path.exists(game_results_filename):
            logger.error(f"Game results file not found: {game_results_filename}")
            return False
            
        # Get AWS credentials from environment variables
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_region = os.getenv('AWS_REGION', 'us-east-1')
        bucket_name = os.getenv('AWS_BUCKET_NAME')
        
        if not all([aws_access_key_id, aws_secret_access_key, bucket_name]):
            logger.error("AWS credentials not properly configured in environment variables")
            return False
            
        # Create S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region
        )
        
        # Upload file to S3
        s3_key = game_results_filename
        s3_client.upload_file(game_results_filename, bucket_name, s3_key)
        
        logger.info(f"Successfully uploaded {game_results_filename} to S3 bucket {bucket_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error uploading to S3: {str(e)}")
        return False

if __name__ == "__main__":
    upload_game_results() 