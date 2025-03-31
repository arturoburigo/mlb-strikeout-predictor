import numpy as np
import pandas as pd
import os
import re
from datetime import datetime
from feature_engineering import calculate_weighted_performance
from data_utils import load_data
import glob



def predict_strikeouts_with_confidence(model, pitchers_df, k_percentage_df, pitcher_name, opponent_team, strikeout_line):
    """
    Make prediction with confidence metrics for a pitcher vs opponent.
    
    Args:
        model: Trained ML pipeline
        pitchers_df: Pitchers historical data
        k_percentage_df: Team strikeout percentages
        pitcher_name: Pitcher abbreviation
        opponent_team: Opponent team abbreviation
        strikeout_line: Betting line for strikeouts
        
    Returns:
        Dictionary with prediction, recommendation and confidence metrics
        or None if prediction couldn't be made
    """
    # Create a copy to avoid SettingWithCopyWarning
    pitcher_data = pitchers_df[pitchers_df['Pitcher'] == pitcher_name].copy()
    
    if pitcher_data.empty:
        print(f"No data found for pitcher: {pitcher_name}")
        return None

    # Ensure we have required columns for rolling calculations
    required_columns = ['Season', 'SO', 'IP', 'Home']
    for col in required_columns:
        if col not in pitcher_data.columns:
            pitcher_data[col] = 0  # Initialize with default value
            
    # Calculate rolling features with proper sorting
    pitcher_data = pitcher_data.sort_values('Season')
    pitcher_data['SO_rolling_5'] = pitcher_data['SO'].rolling(5, min_periods=1).mean()
    pitcher_data['SO_rolling_10'] = pitcher_data['SO'].rolling(10, min_periods=1).mean()
    
    # Calculate home/away splits
    pitcher_data['Home_IP'] = pitcher_data[pitcher_data['Home'] == 1.0]['IP'].mean()
    pitcher_data['Away_IP'] = pitcher_data[pitcher_data['Home'] == 0.0]['IP'].mean()
    pitcher_data['Home_SO'] = pitcher_data[pitcher_data['Home'] == 1.0]['SO'].mean()
    pitcher_data['Away_SO'] = pitcher_data[pitcher_data['Home'] == 0.0]['SO'].mean()

    # Get opponent strikeout rate
    opponent_k = k_percentage_df.loc[k_percentage_df['Team'] == opponent_team, '%K'].mean()
    if np.isnan(opponent_k):
        opponent_k = k_percentage_df['%K'].mean()  # Fallback to league average
    
    # Calculate weighted performance metrics
    performance = calculate_weighted_performance(
        pitcher_data=pitcher_data,
        current_season=2024,
        last_season=2023 if not pitcher_data[pitcher_data['Season'] == 2023].empty else None
    )
    
    # Prepare features DataFrame
    features = pd.DataFrame([performance])
    
    # Add derived features with safety checks
    features['IP'] = features['IP'].replace(0, 1)  # Avoid division by zero
    features['SO_per_IP'] = features['SO'] / features['IP']
    features['BB_per_IP'] = features['BB'] / features['IP']
    features['K-BB%'] = features['SO_per_IP'] - features['BB_per_IP']
    features['Opp_K%'] = opponent_k
    features['Team_K%'] = pitcher_data['Team_K%'].iloc[0] if not pitcher_data.empty else k_percentage_df['%K'].mean()
    
    # Define and validate all required features
    model_features = [
        'IP', 'H', 'BB', 'ERA', 'FIP', 'SO_per_IP', 'BB_per_IP', 'K-BB%', 
        'Opp_K%', 'Team_K%', 'Home', 'SO_rolling_5', 'SO_rolling_10',
        'Home_IP', 'Away_IP', 'Home_SO', 'Away_SO'
    ]
    
    # Ensure all features exist in the DataFrame
    for feature in model_features:
        if feature not in features.columns:
            features[feature] = 0  # Initialize missing features with 0
            print(f"Warning: Initialized missing feature {feature} with zeros")
    
    input_features = features[model_features].fillna(0)
    
    try:
        # Make prediction
        predicted_strikeouts = model.predict(input_features)[0]
        
        # Calculate confidence metrics
        if hasattr(model, 'named_steps') and 'randomforestregressor' in model.named_steps:
            # For RandomForest - use tree variance
            predictions = [tree.predict(input_features) for tree in 
                         model.named_steps.randomforestregressor.estimators_]
            std_dev = np.std(predictions)
            confidence = max(0, min(1 - (std_dev / 3), 1))
        else:
            # For other models - use simple confidence based on line proximity
            std_dev = 0
            confidence = 0.8 - (abs(predicted_strikeouts - strikeout_line) / 10)
            confidence = max(0, min(confidence, 1))
        
        # Determine recommendation
        recommended_side = "Over" if predicted_strikeouts > strikeout_line else "Under"
        
        return {
            'predicted_value': float(predicted_strikeouts),
            'recommended_side': recommended_side,
            'confidence_percentage': float(confidence * 100),
            'std_dev': float(std_dev)
        }
        
    except Exception as e:
        print(f"Prediction failed for {pitcher_name}: {str(e)}")
        return None


def process_betting_data(model, pitchers_df, k_percentage_df, betting_data_path, output_dir=None):
    """
    Process betting data and add predictions using the trained model.
    
    Args:
        model: Trained ML pipeline
        pitchers_df: Pitchers historical data
        k_percentage_df: Team strikeout percentages
        betting_data_path: Path to betting data CSV
        output_dir: Directory to save the output file (defaults to current directory)
        
    Returns:
        Updated betting DataFrame with predictions (only rows with valid predictions and selected columns)
    """
    
    # Load betting data
    betting_data = pd.read_csv(betting_data_path)
    
    # Initialize prediction columns
    betting_data['ML Strikeout Line'] = (betting_data['Over Line'] + betting_data['Under Line']) / 2
    betting_data['ML Predict Value'] = np.nan
    betting_data['ML Recommend Side'] = np.nan
    betting_data['ML Confidence Percentage'] = np.nan
    betting_data['Pitcher 2023'] = False
    
    def has_2023_data(pitcher_name):
        """Check if pitcher has 2023 data available."""
        return not pitchers_df[(pitchers_df['Pitcher'] == pitcher_name) & 
                             (pitchers_df['Season'] == 2023)].empty
    
    for index, row in betting_data.iterrows():
        pitcher_name = row['Name_abbreviation']
        opponent_team = row['Opponent']
        strikeout_line = row['ML Strikeout Line']
        
        pitcher_2023 = has_2023_data(pitcher_name)
        betting_data.at[index, 'Pitcher 2023'] = pitcher_2023
        
        # Make prediction
        result = predict_strikeouts_with_confidence(
            model=model,
            pitchers_df=pitchers_df,
            k_percentage_df=k_percentage_df,
            pitcher_name=pitcher_name,
            opponent_team=opponent_team,
            strikeout_line=strikeout_line
        )
        
        if result:
            betting_data.at[index, 'ML Predict Value'] = result['predicted_value']
            betting_data.at[index, 'ML Recommend Side'] = result['recommended_side']
            betting_data.at[index, 'ML Confidence Percentage'] = result['confidence_percentage']
            
            # Print progress
            print(f"{pitcher_name} vs {opponent_team}: "
                  f"Line {strikeout_line:.1f} → Pred {result['predicted_value']:.1f} "
                  f"({result['recommended_side']}, {result['confidence_percentage']:.0f}%)")
        else:
            print(f"⚠️ No prediction for {pitcher_name} vs {opponent_team} - missing data")
    
    # Filter rows with valid predictions
    filtered_data = betting_data.dropna(subset=['ML Predict Value'])
    
    # Remove unwanted columns
    columns_to_drop = ['Opponent', 'Home Team', 'Away Team', 'Probability', 'Bet Rating']
    columns_to_keep = [col for col in filtered_data.columns if col not in columns_to_drop]
    filtered_data = filtered_data[columns_to_keep]
    
    # Extract date from input filename or use current date
    date_match = re.search(r'betting_data_(\d{4}-\d{2}-\d{2})\.csv', betting_data_path)
    if date_match:
        file_date = date_match.group(1)
    else:
        file_date = datetime.now().strftime("%Y-%m-%d")
    
    # Create output filename
    output_filename = f"predicted_{file_date}.csv"
    
    # Set output path
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
    else:
        output_path = output_filename
    
    # Save the filtered data
    filtered_data.to_csv(output_path, index=False)
    print(f"\nSaved predictions to {output_path}")
    print(f"Original rows: {len(betting_data)} | Filtered rows with predictions: {len(filtered_data)}")
    print(f"Columns kept: {len(columns_to_keep)} | Columns removed: {len(columns_to_drop)}")
    
    return filtered_data

def get_top_picks(predictions_df, n=10, verbose=True):
    """
    Get the top n highest confidence picks from the predictions DataFrame,
    showing both ML predicted value and API projected value.
    
    Args:
        predictions_df (pd.DataFrame): DataFrame containing predictions with required columns
        n (int): Number of top picks to return (default: 10)
        verbose (bool): Whether to print the picks (default: True)
        
    Returns:
        pd.DataFrame: DataFrame containing the top n picks
        str: Formatted string of the top picks (if verbose=True)
    """
    if predictions_df is None or predictions_df.empty:
        if verbose:
            print("No predictions available.")
        return None, "No predictions available."
    
    # Ensure required columns exist
    required_columns = ['Player', 'Team', 'ML Predict Value', 'ML Recommend Side', 
                       'Over Line', 'ML Confidence Percentage', 'API Projected Value']
    
    missing_cols = [col for col in required_columns if col not in predictions_df.columns]
    if missing_cols:
        if verbose:
            print(f"Missing required columns: {', '.join(missing_cols)}")
        return None, f"Missing required columns: {', '.join(missing_cols)}"
    
    # Get top n picks
    top_picks = predictions_df.sort_values('ML Confidence Percentage', ascending=False).head(n)
    
    # Create formatted string
    formatted_output = ""
    if verbose:
        formatted_output = f"\nTop {n} highest confidence picks:\n\n"
        for _, pick in top_picks.iterrows():
            formatted_output += (f"{pick['Player']} ({pick['Team']}): "
                               f"ML Pred: {pick['ML Predict Value']:.1f} | "
                               f"API Proj: {pick['API Projected Value']:.1f} | "
                               f"{pick['ML Recommend Side']} {pick['Over Line']} | "
                               f"Confidence: {pick['ML Confidence Percentage']:.1f}%\n")
        print(formatted_output)
    
    return top_picks, formatted_output


def main():
    """
    Main function to run the strikeout prediction workflow.
    Loads data, trains the model, and processes betting data.
    """
    try:
        pitchers_df, k_percentage_df, betting_file = load_data()
        print(f"Data loaded successfully using load_data() function.")
        print(f"Using betting file: {betting_file}")
    except (ImportError, ModuleNotFoundError):
        # Fallback to direct loading
        print("Fallback to direct data loading...")
        pitchers_df = pd.read_csv('pitchers_data.csv')
        k_percentage_df = pd.read_csv('team_strikeout_percentage.csv')
        
        # Find most recent betting file
        betting_files = glob.glob('betting_data_*.csv')
        if not betting_files:
            betting_file = 'betting_data.csv'
        else:
            betting_files.sort(key=lambda x: datetime.strptime(x.split('_')[2].split('.')[0], '%Y-%m-%d'), reverse=True)
            betting_file = betting_files[0]
        print(f"Data loaded directly from CSV files")
        print(f"Using betting file: {betting_file}")
    
    # Train or load model
    print("\nTraining model...")
    try:
        from model_training import train_model
        model, results = train_model(pitchers_df, k_percentage_df)
        
        # Create directory for model outputs if it doesn't exist
        #os.makedirs('predictions', exist_ok=True)
        
        # Print model performance summary
        print("\nModel performance summary:")
        best_model_name = max(results, key=lambda k: results[k]['CV_R2'])
        print(f"Best model: {best_model_name}")
        print(f"CV R2 Score: {results[best_model_name]['CV_R2']:.4f}")
        print(f"Test R2 Score: {results[best_model_name]['Test_R2']:.4f}")
        print(f"Test MAE: {results[best_model_name]['Test_MAE']:.4f}")
    except (ImportError, ModuleNotFoundError) as e:
        print(f"Error loading train_model function: {e}")
        print("Please ensure you've run model training first or have a trained model available.")
        return
    
    # Process betting data with predictions
    print("\nProcessing betting data and making predictions...")
    predictions = process_betting_data(
        model=model,
        pitchers_df=pitchers_df,
        k_percentage_df=k_percentage_df,
        betting_data_path=betting_file, 
    )
    
    # Display prediction summary
    if predictions is not None and not predictions.empty:
        total_predictions = len(predictions)
        over_count = (predictions['ML Recommend Side'] == 'Over').sum()
        under_count = (predictions['ML Recommend Side'] == 'Under').sum()
        high_confidence = (predictions['ML Confidence Percentage'] >= 70).sum()
        
        print("\nPrediction Summary:")
        print(f"Total predictions: {total_predictions}")
        print(f"Over recommendations: {over_count} ({over_count/total_predictions*100:.1f}%)")
        print(f"Under recommendations: {under_count} ({under_count/total_predictions*100:.1f}%)")
        print(f"High confidence picks (≥70%): {high_confidence} ({high_confidence/total_predictions*100:.1f}%)")
        
        # Display top 5 highest confidence picks
        if not predictions.empty:
            print("\nTop 5 highest confidence picks:")
            top_picks = predictions.sort_values('ML Confidence Percentage', ascending=False).head(5)
            for _, pick in top_picks.iterrows():
                print(f"{pick['Player']} ({pick['Team']}): {pick['ML Predict Value']:.1f} "
                      f"{pick['ML Recommend Side']} {pick['Over Line']} "
                      f"[{pick['ML Confidence Percentage']:.1f}% confidence]")
    
    print("\nWorkflow completed successfully!")
    return predictions


if __name__ == "__main__":
    main()