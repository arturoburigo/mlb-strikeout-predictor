import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os
from datetime import datetime
import re

load_dotenv()

def generate_analysis():
    """
    Generates a baseball analysis using ChatGPT and returns the formatted message.
    Also saves the analysis to a CSV file.
    """
    # Read your CSVs
    file_date = datetime.now().strftime("%Y-%m-%d")
    predicted_data = pd.read_csv(f"predicted_{file_date}.csv")

    # Convert data to text
    predicted_text = predicted_data.to_string(index=False)

    # Build the complete prompt
    prompt = f"""
    You are an expert baseball analyst and professional sports bettor with years of experience in MLB strikeout betting. Your analysis is known for its accuracy and profitability.

    I have provided you with a dataset of strikeout predictions for upcoming games, including both API-based predictions and machine learning model predictions.

    The dataset contains the following columns:
    Player, Name_abbreviation, Team, Over Line, Over Odds, Under Line, Under Odds, 
    API Projected Value, API Recommended Side, Streak, Streak Type, Diff,
    ML Strikeout Line, ML Predict Value, ML Recommend Side, ML Confidence Percentage, Pitcher 2023.

    {predicted_text}

    Based on this data, provide a comprehensive analysis following these steps:

    1. IDENTIFY THE BEST BETS:
       - Rank the top 5-7 most promising strikeout bets, considering both the API and ML predictions
       - For each bet, explain why it's a strong opportunity based on the data
       - Highlight any significant discrepancies between the API and ML predictions
       - Note any recent streaks or patterns that support your recommendation

    2. PROVIDE CONTEXT AND INSIGHTS:
       - Identify any pitchers with unusually high or low projected values compared to their lines
       - Point out any pitchers with high confidence ratings from the ML model
       - Highlight any pitchers with significant differences between their API and ML predictions
       - Note any pitchers who didn't play in 2023 (rookies or returning from injury)

    3. RISK ASSESSMENT:
       - For each recommended bet, provide a confidence level (1-10)
       - Explain the potential risks or factors that could affect the outcome
       - Suggest optimal bet sizing based on the odds and your confidence

    4. FORMAT YOUR RESPONSE:
       - Use Telegram formatting with bold, emojis, and clear section headers
       - Make your recommendations stand out visually
       - Be concise but thorough in your explanations

    5. INCLUDE A PLAIN DATA SECTION:
       After your analysis, output a plain extractable list in this exact format:

       --- PLAIN DATA START ---
       Player: [Player Name], Team: [Team Name], Bet: [Over/Under], Line: [Line], Odds: [Odds], API Predicted Value: [Value], ML Predicted Value: [Value], Confidence Level: [Percentage]
       --- PLAIN DATA END ---

    6. CALCULATE POTENTIAL RETURNS:
       - If betting $10 on each recommended bet, how many winning bets are needed to break even?
       - What would be the total profit if all bets hit, using the provided odds?
       - What would be the expected value based on your confidence levels?

    Focus on quality over quantity. It's better to have 3-5 high-confidence bets than 10 uncertain ones.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "ERROR: OPENAI_API_KEY is not configured in the environment."
    
    # Send to OpenAI
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    analysis_text = response.choices[0].message.content
    return analysis_text

def extract_recommended_bets(analysis_text):
    """
    Extract recommended bets from the plain text part of the analysis.
    """
    recommended_bets = []

    plain_section = re.search(
        r"--- PLAIN DATA START ---\s*(.*?)\s*--- PLAIN DATA END ---",
        analysis_text,
        re.DOTALL
    )

    if not plain_section:
        print("No plain data section found.")
        return recommended_bets

    plain_text = plain_section.group(1)

    pattern = (
        r"Player:\s*([^,]+),\s*Team:\s*([^,]+),\s*Bet:\s*(Over|Under),\s*"
        r"Line:\s*([\d.]+),\s*Odds:\s*([\d.]+),\s*API Predicted Value:\s*([\d.]+),\s*"
        r"ML Predicted Value:\s*([\d.]+),\s*Confidence Level:\s*([\d.]+)"
    )

    matches = re.findall(pattern, plain_text)

    for match in matches:
        player, team, bet_type, line, odds, api_pred, ml_pred, confidence = match
        recommended_bets.append({
            "Player": player.strip(),
            "Team": team.strip(),
            "Bet_Type": bet_type.strip(),
            "Line": float(line),
            "Odds": float(odds),
            "API_Predicted_Value": float(api_pred),
            "ML_Predicted_Value": float(ml_pred),
            "Confidence_Level": float(confidence)
        })

    return recommended_bets

def save_recommendations_to_csv(recommended_bets):
    """
    Save the recommended bets to a CSV file.
    """
    if not recommended_bets:
        print("No recommendations found to save.")
        return
    
    recommendations_df = pd.DataFrame(recommended_bets)
    
    file_date = datetime.now().strftime("%Y-%m-%d")
    output_filename = f"recommendations_{file_date}.csv"
    recommendations_df.to_csv(output_filename, index=False)
    print(f"Recommendations saved to {output_filename}")

if __name__ == "__main__":
    analysis = generate_analysis()
    print(analysis)
    
    recommendations = extract_recommended_bets(analysis)
    print(f"Extracted {len(recommendations)} recommended bets.")
    save_recommendations_to_csv(recommendations)