import os
import glob
import logging
from datetime import datetime
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def cleanup_csv_files():
    """
    Clean up all CSV files except team_strikeout_percentage.csv.
    Returns True if successful, False otherwise.
    """
    try:
        # Get all CSV files in the current directory
        csv_files = glob.glob('*.csv')
        
        # Keep track of files to delete
        files_to_delete = []
        
        # Identify files to delete (excluding team_strikeout_percentage.csv)
        for file in csv_files:
            if file != 'team_strikeout_percentage.csv':
                files_to_delete.append(file)
        
        # Delete the files
        for file in files_to_delete:
            try:
                os.remove(file)
                logger.info(f"Successfully deleted: {file}")
            except Exception as e:
                logger.error(f"Error deleting {file}: {str(e)}")
                return False
        
        logger.info(f"Cleanup completed. Deleted {len(files_to_delete)} files.")
        return True
        
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        return False

if __name__ == "__main__":
    cleanup_csv_files() 