import os
import requests
import pandas as pd
from datetime import datetime
import glob
from dotenv import load_dotenv
from predictions import get_top_picks

load_dotenv()

def send_predictions_to_telegram(predictions_df=None, csv_path=None):
    """
    Send MLB predictions to a Telegram group.
    
    Args:
        predictions_df (pd.DataFrame): DataFrame containing predictions
        csv_path (str): Path to predictions CSV file (optional if df is provided)
        
    Returns:
        bool: True if message sent successfully
    """
    # Get Telegram credentials from environment variables
    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not telegram_bot_token or not telegram_chat_id:
        print("Error: Telegram credentials not configured in environment variables")
        print(f"Bot token: {'set' if telegram_bot_token else 'not set'}")
        print(f"Chat ID: {'set' if telegram_chat_id else 'not set'}")
        return False
    
    # Load data if CSV path is provided but no DataFrame
    if predictions_df is None and csv_path:
        try:
            predictions_df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            predictions_df = None
    
    # Get top picks
    top_picks_df, top_picks_text = get_top_picks(predictions_df, verbose=False)
    
    # Create message content
    if top_picks_df is None:
        message = "⚾ MLB Strikeout Predictions\n\nNo predictions available today."
    else:
        message = f"⚾ Today's Top {len(top_picks_df)} MLB Strikeout Predictions:\n\n"
        message += top_picks_text
    
    # Send message to Telegram
    try:
        url = f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage"
        payload = {
            "chat_id": telegram_chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        
        response = requests.post(url, data=payload)
        response.raise_for_status()
        
        print(f"Successfully sent predictions to Telegram group")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Error sending message to Telegram: {str(e)}")
        return False

if __name__ == "__main__":
    today_date = datetime.now().strftime("%Y-%m-%d")
    csv_filename = f"predicted_{today_date}.csv"
    
    if not os.path.exists(csv_filename):
        print(f"Today's prediction file not found, looking for most recent...")
        prediction_files = glob.glob("predicted_*.csv")
        
        if prediction_files:
            prediction_files.sort(reverse=True)
            csv_filename = prediction_files[0]
            print(f"Using most recent file: {csv_filename}")
        else:
            print("Error: No prediction files found")
            exit(1)
    
    try:
        predictions_df = pd.read_csv(csv_filename)
        send_predictions_to_telegram(
            predictions_df=predictions_df,
            csv_path=csv_filename
        )
    except Exception as e:
        print(f"Error processing prediction file: {e}") 