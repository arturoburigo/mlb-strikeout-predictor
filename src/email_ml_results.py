import smtplib
import glob
from datetime import datetime
import os
from email.message import EmailMessage
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

def analyze_predictions(predictions_df, results_df):
    """
    Analyze predictions vs actual results and calculate potential profits.
    Only includes predictions with ML Confidence Percentage > 68%.
    
    Args:
        predictions_df (pd.DataFrame): DataFrame containing predictions
        results_df (pd.DataFrame): DataFrame containing actual results
        
    Returns:
        tuple: (correct_predictions, incorrect_predictions, total_profit, profit_by_pitcher)
    """
    # Merge predictions with results
    merged_df = predictions_df.merge(
        results_df[['Name_abbreviation', 'REAL SO']],
        left_on='Name_abbreviation',
        right_on='Name_abbreviation'
    )
    
    # Filter for high confidence predictions (> 68%)
    merged_df = merged_df[merged_df['ML Confidence Percentage'] > 68]
    
    # Calculate correct and incorrect predictions
    correct_predictions = []
    incorrect_predictions = []
    profit_by_pitcher = []
    
    for _, row in merged_df.iterrows():
        ml_side = row['ML Recommend Side']
        ml_so_pred = row['ML Predict Value']
        over_line = row['Over Line']
        actual_so = row['REAL SO']
        
        # Use appropriate odds based on recommendation
        odds = row['Under Odds'] if ml_side == 'Under' else row['Over Odds']
        confidence = row['ML Confidence Percentage']
        
        # Determine if prediction was correct
        # For UNDER: actual SO must be LESS than the line
        # For OVER: actual SO must be GREATER than the line
        if ml_side == 'Under':  # Under prediction
            is_correct = actual_so < over_line
        else:  # Over prediction
            is_correct = actual_so > over_line
        
        # Calculate profit for $10 bet
        # If actual SO equals the line, it's a push (no profit/loss)
        if actual_so == over_line:
            profit = 0
        else:
            profit = 10 * (odds - 1) if is_correct else -10
        
        result = {
            'Pitcher': row['Player'],
            'Team': row['Team'],
            'ML Side': ml_side,
            'ML SO Pred': f"{ml_so_pred:.1f}",
            'Line': over_line,
            'Actual': actual_so,
            'Odds': odds,
            'Profit': profit,
            'Confidence': confidence
        }
        
        if is_correct:
            correct_predictions.append(result)
        else:
            incorrect_predictions.append(result)
        
        profit_by_pitcher.append(result)
    
    # Calculate total profit
    total_profit = sum(p['Profit'] for p in profit_by_pitcher)
    
    return correct_predictions, incorrect_predictions, total_profit, profit_by_pitcher

def create_email_content(predictions_df, results_df):
    """
    Create email subject and body content with results analysis.
    
    Args:
        predictions_df (pd.DataFrame): DataFrame containing predictions
        results_df (pd.DataFrame): DataFrame containing actual results
        
    Returns:
        tuple: (subject, html_body, text_body)
    """
    # Analyze predictions
    correct_predictions, incorrect_predictions, total_profit, profit_by_pitcher = analyze_predictions(predictions_df, results_df)
    
    # Sort predictions by confidence (highest to lowest)
    correct_predictions.sort(key=lambda x: x['Confidence'], reverse=True)
    incorrect_predictions.sort(key=lambda x: x['Confidence'], reverse=True)
    
    # Create email content
    subject = "⚾ MLB Strikeout Predictions Results"
    
    # HTML version
    html_body = f"""
    <h2>⚾ MLB Strikeout Predictions Results</h2>
    
    <h3>Summary</h3>
    <p>Total Predictions: {len(profit_by_pitcher)}</p>
    <p>Correct Predictions: {len(correct_predictions)}</p>
    <p>Incorrect Predictions: {len(incorrect_predictions)}</p>
    <p>Accuracy Rate: {(len(correct_predictions) / len(profit_by_pitcher) * 100):.1f}%</p>
    <p>Total Profit (with $10 bets): ${total_profit:.2f}</p>
    
    <h3>Correct Predictions</h3>
    <table border="1" cellpadding="5" cellspacing="0">
        <tr>
            <th>Pitcher (Team)</th>
            <th>ML Side</th>
            <th>Line</th>
            <th>Game SO</th>
            <th>ML SO Pred</th>
            <th>Confidence</th>
            <th>Odds</th>
            <th>Profit</th>
        </tr>
    """
    
    for pred in correct_predictions:
        html_body += f"""
        <tr>
            <td>{pred['Pitcher']} ({pred['Team']})</td>
            <td>{pred['ML Side']}</td>
            <td>{pred['Line']}</td>
            <td>{pred['Actual']}</td>
            <td>{pred['ML SO Pred']}</td>
            <td>{pred['Confidence']:.1f}%</td>
            <td>{pred['Odds']:.1f}</td>
            <td>${pred['Profit']:.2f}</td>
        </tr>
        """
    
    html_body += "</table>"
    
    html_body += """
    <h3>Incorrect Predictions</h3>
    <table border="1" cellpadding="5" cellspacing="0">
        <tr>
            <th>Pitcher (Team)</th>
            <th>ML Side</th>
            <th>Line</th>
            <th>Game SO</th>
            <th>ML SO Pred</th>
            <th>Confidence</th>
            <th>Odds</th>
            <th>Loss</th>
        </tr>
    """
    
    for pred in incorrect_predictions:
        html_body += f"""
        <tr>
            <td>{pred['Pitcher']} ({pred['Team']})</td>
            <td>{pred['ML Side']}</td>
            <td>{pred['Line']}</td>
            <td>{pred['Actual']}</td>
            <td>{pred['ML SO Pred']}</td>
            <td>{pred['Confidence']:.1f}%</td>
            <td>{pred['Odds']:.1f}</td>
            <td>${abs(pred['Profit']):.2f}</td>
        </tr>
        """
    
    html_body += "</table>"
    
    # Plain text version
    text_body = f"""MLB Strikeout Predictions Results

Summary:
Total Predictions: {len(profit_by_pitcher)}
Correct Predictions: {len(correct_predictions)}
Incorrect Predictions: {len(incorrect_predictions)}
Accuracy Rate: {(len(correct_predictions) / len(profit_by_pitcher) * 100):.1f}%
Total Profit (with $10 bets): ${total_profit:.2f}

Correct Predictions:
"""
    
    for pred in correct_predictions:
        text_body += f"{pred['Pitcher']} ({pred['Team']}) | {pred['ML Side']} | {pred['Line']} | {pred['Actual']} | {pred['ML SO Pred']} | {pred['Confidence']:.1f}% | {pred['Odds']:.1f} | ${pred['Profit']:.2f}\n"
    
    text_body += "\nIncorrect Predictions:\n"
    
    for pred in incorrect_predictions:
        text_body += f"{pred['Pitcher']} ({pred['Team']}) | {pred['ML Side']} | {pred['Line']} | {pred['Actual']} | {pred['ML SO Pred']} | {pred['Confidence']:.1f}% | {pred['Odds']:.1f} | ${abs(pred['Profit']):.2f}\n"
    
    return subject, html_body, text_body

def send_results_email(receiver_email, predictions_df, results_df, cc_emails=None):
    """
    Send results email with analysis and CSV attachment.
    
    Args:
        receiver_email (str): Email address to send to
        predictions_df (pd.DataFrame): Predictions DataFrame
        results_df (pd.DataFrame): Results DataFrame
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
    
    try:
        # Create email content
        subject, html_body, text_body = create_email_content(predictions_df, results_df)
        
        # Create multipart message
        msg = MIMEMultipart("alternative")
        msg["From"] = sender_name
        msg["To"] = receiver_email
        msg["Subject"] = subject
        
        # Add CC recipients if provided
        if cc_emails:
            msg["Cc"] = ", ".join(cc_emails)
        
        # Attach both HTML and plain text versions
        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))
        
        # Send email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            
            # Combine To and CC recipients for the actual send
            all_recipients = [receiver_email]
            if cc_emails:
                all_recipients.extend(cc_emails)
            
            server.send_message(msg, to_addrs=all_recipients)
        
        print(f"Successfully sent results to {receiver_email}")
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
    cc_emails = os.getenv("CC_EMAILS", "").split(",")
    
    # Clean up CC emails
    cc_emails = [email.strip() for email in cc_emails if email.strip()]
    
    # Get yesterday's date for the files
    today_date = datetime.now()
    yesterday_date = today_date - pd.Timedelta(days=1)
    yesterday_date_str = yesterday_date.strftime("%Y-%m-%d")
    
    predictions_filename = f"predicted_{yesterday_date_str}.csv"
    results_filename = f"game_results_{yesterday_date_str}.csv"
    
    if not os.path.exists(predictions_filename) or not os.path.exists(results_filename):
        print(f"Error: Required files not found for {yesterday_date_str}")
        print(f"Looking for: {predictions_filename} and {results_filename}")
        exit(1)
    
    try:
        predictions_df = pd.read_csv(predictions_filename)
        results_df = pd.read_csv(results_filename)
        
        send_results_email(
            receiver_email=destination_email,
            predictions_df=predictions_df,
            results_df=results_df,
            cc_emails=cc_emails
        )
    except Exception as e:
        print(f"Error processing files: {e}") 