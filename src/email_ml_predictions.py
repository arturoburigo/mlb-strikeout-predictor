import smtplib
import glob
from datetime import datetime
import os
from email.message import EmailMessage
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from predictions import get_top_picks
import pandas as pd

load_dotenv()

def create_email_content(predictions_df=None, csv_path=None):
    """
    Create email subject and body content with top picks information.
    
    Args:
        predictions_df (pd.DataFrame): DataFrame containing predictions
        csv_path (str): Path to predictions CSV file (optional if df is provided)
        
    Returns:
        tuple: (subject, html_body, text_body)
    """
    # Load data if CSV path is provided but no DataFrame
    if predictions_df is None and csv_path:
        try:
            predictions_df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            predictions_df = None
    
    # Get top picks
    top_picks_df, top_picks_text = get_top_picks(predictions_df, verbose=False)
    
    # Create email content
    subject = "⚾ Daily MLB Strikeout Predictions"
    
    if top_picks_df is None:
        html_body = "<h2>MLB Strikeout Predictions</h2><p>No predictions available today.</p>"
        text_body = "MLB Strikeout Predictions\n\nNo predictions available today."
    else:
        # HTML version
        html_body = f"""
        <h2>⚾ Today's Top {len(top_picks_df)} MLB Strikeout Predictions</h2>
        <table border="1" cellpadding="5" cellspacing="0">
            <tr>
                <th>Pitcher (Team)</th>
                <th>API Prediction</th>
                <th>ML Prediction</th>
                <th>ML Recommendation</th>
                <th>ML Confidence</th>
            </tr>
        """
        
        for _, row in top_picks_df.iterrows():
            html_body += f"""
            <tr>
                <td>{row['Player']} ({row['Team']})</td>
                <td>{row['API Projected Value']:.1f}</td>
                <td>{row['ML Predict Value']:.1f}</td>
                <td>{row['ML Recommend Side']} {row['Over Line']}</td>
                <td>{row['ML Confidence Percentage']:.1f}%</td>
            </tr>
            """
        
        html_body += "</table>"
        html_body += "<p>See attached CSV for complete predictions.</p>"
        
        # Plain text version
        text_body = f"Today's Top {len(top_picks_df)} MLB Strikeout Predictions:\n\n"
        text_body += top_picks_text
        text_body += "\nSee attached CSV for complete predictions."
    
    return subject, html_body, text_body


def send_prediction_email(receiver_email, predictions_df=None, csv_path=None, cc_emails=None):
    """
    Send prediction email with top picks and CSV attachment, with optional CC recipients.
    
    Args:
        receiver_email (str): Email address to send to
        predictions_df (pd.DataFrame): Predictions DataFrame (optional)
        csv_path (str): Path to predictions CSV (required if no df provided)
        cc_emails (list): List of email addresses to CC (optional)
        
    Returns:
        bool: True if email sent successfully
    """
    sender_email = os.getenv("EMAIL_SENDER")
    sender_password = os.getenv("EMAIL_PASSWORD")
    sender_name = "MLB PREDICT"
    
    print(f"Sender Email: {sender_email}")
    print(f"Receiver Email: {receiver_email}")
    if cc_emails:
        print(f"CC Emails: {', '.join(cc_emails)}")
    
    if not sender_email or not sender_password:
        print("Error: Email credentials not configured in environment variables")
        print(f"Sender email: {'set' if sender_email else 'not set'}")
        print(f"Sender password: {'set' if sender_password else 'not set'}")
        return False
    
    if predictions_df is None and csv_path is None:
        print("Error: Must provide either predictions_df or csv_path")
        return False
    
    try:
        # Create email content
        subject, html_body, text_body = create_email_content(predictions_df, csv_path)
        
        # Create multipart message
        msg = MIMEMultipart("alternative")
        msg["From"] = sender_name
        msg["To"] = receiver_email
        msg["Subject"] = subject
        
        # Add CC recipients if provided
        if cc_emails:
            msg["Cc"] = ", ".join(cc_emails)  # Join multiple emails with comma
        
        # Attach both HTML and plain text versions
        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))
        
        # Attach CSV file if path provided
        if csv_path:
            try:
                with open(csv_path, "rb") as f:
                    part = MIMEText(f.read().decode("utf-8"), "csv")
                    part.add_header(
                        "Content-Disposition",
                        f"attachment; filename={os.path.basename(csv_path)}",
                    )
                    msg.attach(part)
            except Exception as e:
                print(f"Warning: Could not attach CSV file - {str(e)}")
        
        # Debug print before sending
        print(f"Attempting to send email from {sender_email} to {receiver_email}")
        if cc_emails:
            print(f"CC: {', '.join(cc_emails)}")
        print(f"Subject: {subject}")
        print(f"Attachment: {csv_path if csv_path else 'None'}")
        
        # Send email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            
            # Combine To and CC recipients for the actual send
            all_recipients = [receiver_email]
            if cc_emails:
                all_recipients.extend(cc_emails)
            
            server.send_message(msg, to_addrs=all_recipients)
        
        print(f"Successfully sent predictions to {receiver_email}")
        if cc_emails:
            print(f"CC'd: {', '.join(cc_emails)}")
        return True
        
    except smtplib.SMTPAuthenticationError:
        print("Error: Authentication failed. Please check your email credentials.")
        print("Note: You might need to use an 'App Password' if you have 2FA enabled.")
        return False
    except smtplib.SMTPException as e:
        print(f"SMTP Error: {str(e)}")
        return False
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False
    
def send_prediction_email(receiver_email, predictions_df=None, csv_path=None, cc_emails=None):
    """
    Send prediction email with top picks and CSV attachment, with optional CC recipients.
    
    Args:
        receiver_email (str): Email address to send to
        predictions_df (pd.DataFrame): Predictions DataFrame (optional)
        csv_path (str): Path to predictions CSV (required if no df provided)
        cc_emails (list): List of email addresses to CC (optional)
        
    Returns:
        bool: True if email sent successfully
    """
    sender_email = os.getenv("EMAIL_SENDER")
    sender_password = os.getenv("EMAIL_PASSWORD")
    sender_name = "MLB PREDICT"
    
    print(f"Sender Email: {sender_email}")
    print(f"Receiver Email: {receiver_email}")
    if cc_emails:
        print(f"CC Emails: {', '.join(cc_emails)}")
    
    if not sender_email or not sender_password:
        print("Error: Email credentials not configured in environment variables")
        print(f"Sender email: {'set' if sender_email else 'not set'}")
        print(f"Sender password: {'set' if sender_password else 'not set'}")
        return False
    
    if predictions_df is None and csv_path is None:
        print("Error: Must provide either predictions_df or csv_path")
        return False
    
    try:
        # Create email content
        subject, html_body, text_body = create_email_content(predictions_df, csv_path)
        
        # Create multipart message
        msg = MIMEMultipart("alternative")
        msg["From"] = sender_name
        msg["To"] = receiver_email
        msg["Subject"] = subject
        
        # Add CC recipients if provided
        if cc_emails:
            msg["Cc"] = ", ".join(cc_emails)  # Join multiple emails with comma
        
        # Attach both HTML and plain text versions
        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))
        
        # Attach CSV file if path provided
        if csv_path:
            try:
                with open(csv_path, "rb") as f:
                    part = MIMEText(f.read().decode("utf-8"), "csv")
                    part.add_header(
                        "Content-Disposition",
                        f"attachment; filename={os.path.basename(csv_path)}",
                    )
                    msg.attach(part)
            except Exception as e:
                print(f"Warning: Could not attach CSV file - {str(e)}")
        
        # Debug print before sending
        print(f"Attempting to send email from {sender_email} to {receiver_email}")
        if cc_emails:
            print(f"CC: {', '.join(cc_emails)}")
        print(f"Subject: {subject}")
        print(f"Attachment: {csv_path if csv_path else 'None'}")
        
        # Send email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            
            # Combine To and CC recipients for the actual send
            all_recipients = [receiver_email]
            if cc_emails:
                all_recipients.extend(cc_emails)
            
            server.send_message(msg, to_addrs=all_recipients)
        
        print(f"Successfully sent predictions to {receiver_email}")
        if cc_emails:
            print(f"CC'd: {', '.join(cc_emails)}")
        return True
        
    except smtplib.SMTPAuthenticationError:
        print("Error: Authentication failed. Please check your email credentials.")
        print("Note: You might need to use an 'App Password' if you have 2FA enabled.")
        return False
    except smtplib.SMTPException as e:
        print(f"SMTP Error: {str(e)}")
        return False
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False
    
if __name__ == "__main__":
    destination_email = os.getenv("RECEIVER_EMAIL")
    cc_emails = os.getenv("CC_EMAILS", "").split(",")  # Assuming CC_EMAILS is comma-separated
    
    # Clean up CC emails (remove empty strings from env var)
    cc_emails = [email.strip() for email in cc_emails if email.strip()]
    
    today_date = datetime.now().strftime("%Y-%m-%d")
    csv_filename = f"   predicted_{today_date}.csv"
    
    if not os.path.exists(csv_filename):
        print(f"Today's prediction file not found, looking for most recent...")
        prediction_files = glob.glob("data_predicted_*.csv")
        
        if prediction_files:
            prediction_files.sort(reverse=True)
            csv_filename = prediction_files[0]
            print(f"Using most recent file: {csv_filename}")
        else:
            print("Error: No prediction files found")
            exit(1)
    
    try:
        predictions_df = pd.read_csv(csv_filename)
        send_prediction_email(
            receiver_email=destination_email,
            predictions_df=predictions_df,
            csv_path=csv_filename,
            cc_emails=cc_emails
        )
    except Exception as e:
        print(f"Error processing prediction file: {e}")