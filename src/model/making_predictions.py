import numpy as np
import pandas as pd
import os
import re
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

def create_features_for_prediction(pitchers_df, pitcher_name, opponent_team=None):
    """
    Create features for a specific pitcher using the cleaned data format.
    
    Args:
        pitchers_df (DataFrame): DataFrame with cleaned pitcher data
        pitcher_name (str): Pitcher abbreviation
        opponent_team (str, optional): Opponent team abbreviation
        
    Returns:
        DataFrame: Single row with features for prediction
    """
    # Filter data for the specific pitcher
    pitcher_data = pitchers_df[pitchers_df['Pitcher'] == pitcher_name].copy()
    
    if pitcher_data.empty:
        print(f"No data found for pitcher: {pitcher_name}")
        return None
    
    # Sort by opponent to get chronological order
    pitcher_data = pitcher_data.sort_values('Opp').reset_index(drop=True)
    
    # Calculate basic rate statistics
    pitcher_data['SO_per_IP'] = pitcher_data['SO'] / pitcher_data['IP']
    pitcher_data['BB_per_IP'] = (pitcher_data['BF'] - pitcher_data['Str']) / pitcher_data['IP']
    pitcher_data['Str_rate'] = pitcher_data['Str'] / pitcher_data['Pit']
    pitcher_data['StS_rate'] = pitcher_data['StS'] / pitcher_data['Pit']
    pitcher_data['StL_rate'] = pitcher_data['StL'] / pitcher_data['Pit']
    pitcher_data['Pit_per_IP'] = pitcher_data['Pit'] / pitcher_data['IP']
    pitcher_data['BF_per_IP'] = pitcher_data['BF'] / pitcher_data['IP']
    pitcher_data['SO_per_BF'] = pitcher_data['SO'] / pitcher_data['BF']
    pitcher_data['SO_per_Pit'] = pitcher_data['SO'] / pitcher_data['Pit']
    
    # Handle infinity and NaN values
    numeric_columns = pitcher_data.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        pitcher_data[col] = pitcher_data[col].replace([np.inf, -np.inf], 0)
        pitcher_data[col] = pitcher_data[col].fillna(0)
    
    # Clip extreme values
    pitcher_data['SO_per_IP'] = pitcher_data['SO_per_IP'].clip(0, 20)
    pitcher_data['BB_per_IP'] = pitcher_data['BB_per_IP'].clip(0, 10)
    pitcher_data['Str_rate'] = pitcher_data['Str_rate'].clip(0, 1)
    pitcher_data['StS_rate'] = pitcher_data['StS_rate'].clip(0, 1)
    pitcher_data['StL_rate'] = pitcher_data['StL_rate'].clip(0, 1)
    pitcher_data['Pit_per_IP'] = pitcher_data['Pit_per_IP'].clip(0, 50)
    pitcher_data['BF_per_IP'] = pitcher_data['BF_per_IP'].clip(0, 10)
    pitcher_data['SO_per_BF'] = pitcher_data['SO_per_BF'].clip(0, 1)
    pitcher_data['SO_per_Pit'] = pitcher_data['SO_per_Pit'].clip(0, 1)
    
    # Calculate rolling averages
    pitcher_data['SO_rolling_3'] = pitcher_data['SO'].rolling(3, min_periods=1).mean()
    pitcher_data['SO_rolling_5'] = pitcher_data['SO'].rolling(5, min_periods=1).mean()
    pitcher_data['IP_rolling_3'] = pitcher_data['IP'].rolling(3, min_periods=1).mean()
    pitcher_data['IP_rolling_5'] = pitcher_data['IP'].rolling(5, min_periods=1).mean()
    
    # Calculate pitcher averages
    pitcher_avg = pitcher_data.agg({
        'SO': 'mean',
        'IP': 'mean',
        'Pit': 'mean',
        'Str': 'mean',
        'StS': 'mean',
        'StL': 'mean',
        'BF': 'mean',
        'WPA': 'mean'
    })
    
    # Get the most recent game data
    latest_game = pitcher_data.iloc[-1]
    
    # Get opponent strikeout average if available
    if opponent_team and opponent_team != 'N/A':
        opp_so_avg = pitcher_data[pitcher_data['Opp'] == opponent_team]['opp_so_avg'].mean()
        if pd.isna(opp_so_avg):
            opp_so_avg = pitcher_data['opp_so_avg'].mean()
    else:
        opp_so_avg = pitcher_data['opp_so_avg'].mean()
    
    # Normalize opponent strikeout average
    opp_so_avg_norm = (opp_so_avg - pitcher_data['opp_so_avg'].mean()) / pitcher_data['opp_so_avg'].std()
    
    # Define feature columns in the same order as training
    feature_columns = [
        'IP', 'BF', 'Pit', 'Str', 'StS', 'StL', 'WPA',
        'SO_per_IP', 'BB_per_IP', 'Str_rate', 'StS_rate', 'StL_rate',
        'Pit_per_IP', 'BF_per_IP', 'SO_per_BF', 'SO_per_Pit',
        'SO_rolling_3', 'SO_rolling_5', 'IP_rolling_3', 'IP_rolling_5',
        'avg_SO', 'avg_IP', 'avg_Pit', 'avg_Str', 'avg_StS', 'avg_StL', 'avg_BF', 'avg_WPA',
        'SO_vs_avg', 'IP_vs_avg', 'Pit_vs_avg',
        'opp_so_avg', 'opp_so_avg_norm'
    ]
    
    # Create features for prediction
    features = pd.DataFrame([{
        'IP': latest_game['IP'],
        'BF': latest_game['BF'],
        'Pit': latest_game['Pit'],
        'Str': latest_game['Str'],
        'StS': latest_game['StS'],
        'StL': latest_game['StL'],
        'WPA': latest_game['WPA'],
        'SO_per_IP': latest_game['SO_per_IP'],
        'BB_per_IP': latest_game['BB_per_IP'],
        'Str_rate': latest_game['Str_rate'],
        'StS_rate': latest_game['StS_rate'],
        'StL_rate': latest_game['StL_rate'],
        'Pit_per_IP': latest_game['Pit_per_IP'],
        'BF_per_IP': latest_game['BF_per_IP'],
        'SO_per_BF': latest_game['SO_per_BF'],
        'SO_per_Pit': latest_game['SO_per_Pit'],
        'SO_rolling_3': latest_game['SO_rolling_3'],
        'SO_rolling_5': latest_game['SO_rolling_5'],
        'IP_rolling_3': latest_game['IP_rolling_3'],
        'IP_rolling_5': latest_game['IP_rolling_5'],
        'avg_SO': pitcher_avg['SO'],
        'avg_IP': pitcher_avg['IP'],
        'avg_Pit': pitcher_avg['Pit'],
        'avg_Str': pitcher_avg['Str'],
        'avg_StS': pitcher_avg['StS'],
        'avg_StL': pitcher_avg['StL'],
        'avg_BF': pitcher_avg['BF'],
        'avg_WPA': pitcher_avg['WPA'],
        'SO_vs_avg': latest_game['SO'] - pitcher_avg['SO'],
        'IP_vs_avg': latest_game['IP'] - pitcher_avg['IP'],
        'Pit_vs_avg': latest_game['Pit'] - pitcher_avg['Pit'],
        'opp_so_avg': opp_so_avg,
        'opp_so_avg_norm': opp_so_avg_norm
    }])
    
    # Ensure all features are present and in correct order
    for col in feature_columns:
        if col not in features.columns:
            features[col] = 0
    
    # Return only the feature columns in the correct order
    return features[feature_columns]

def predict_strikeouts_with_confidence(model, pitchers_df, pitcher_name, opponent_team, strikeout_line):
    """
    Make prediction with confidence metrics for a pitcher vs opponent.
    
    Args:
        model: Trained ML pipeline
        pitchers_df: Pitchers historical data (cleaned format)
        pitcher_name: Pitcher abbreviation
        opponent_team: Opponent team abbreviation
        strikeout_line: Betting line for strikeouts
        
    Returns:
        Dictionary with prediction, recommendation and confidence metrics
        or None if prediction couldn't be made
    """
    try:
        # Create features for prediction
        features = create_features_for_prediction(pitchers_df, pitcher_name, opponent_team)
        
        if features is None:
            return None
        
        # Make prediction
        predicted_strikeouts = float(model.predict(features)[0])
        
        # Calculate confidence based on model type
        if hasattr(model, 'named_steps'):
            model_step = None
            for step_name, step_model in model.named_steps.items():
                if hasattr(step_model, 'estimators_'):
                    model_step = step_model
                    break
            
            if model_step and hasattr(model_step, 'estimators_'):
                # For ensemble models - use tree variance
                try:
                    predictions = [float(tree.predict(features)[0]) for tree in model_step.estimators_]
                    std_dev = float(np.std(predictions))
                    confidence = max(0, min(1 - (std_dev / 3), 1))
                except:
                    # Fallback confidence calculation
                    std_dev = 0.0
                    confidence = 0.8 - (abs(predicted_strikeouts - strikeout_line) / 10)
                    confidence = max(0, min(confidence, 1))
            else:
                # For other models - use simple confidence based on line proximity
                std_dev = 0.0
                confidence = 0.8 - (abs(predicted_strikeouts - strikeout_line) / 10)
                confidence = max(0, min(confidence, 1))
        else:
            # Fallback confidence calculation
            std_dev = 0.0
            confidence = 0.8 - (abs(predicted_strikeouts - strikeout_line) / 10)
            confidence = max(0, min(confidence, 1))
        
        # Determine recommendation
        recommended_side = "Over" if predicted_strikeouts > strikeout_line else "Under"
        
        return {
            'predicted_value': predicted_strikeouts,
            'recommended_side': recommended_side,
            'confidence_percentage': float(confidence * 100),
            'std_dev': std_dev
        }
        
    except Exception as e:
        print(f"Prediction failed for {pitcher_name}: {str(e)}")
        return None

def process_betting_data(model, pitchers_df, betting_data_path, output_dir=None):
    """
    Process betting data and add predictions using the trained model.
    
    Args:
        model: Trained ML pipeline
        pitchers_df: Pitchers historical data (cleaned format)
        betting_data_path: Path to betting data CSV
        output_dir: Directory to save the output file (defaults to current directory)
        
    Returns:
        Updated betting DataFrame with predictions
    """
    
    # Load betting data
    betting_data = pd.read_csv(betting_data_path)
    
    # Initialize prediction columns
    betting_data['ML Predict Value'] = np.nan
    betting_data['ML Recommend Side'] = pd.Series(dtype='object')
    betting_data['ML Confidence Percentage'] = np.nan
    betting_data['Has Historical Data'] = False
    
    def has_historical_data(pitcher_name):
        """Check if pitcher has historical data available."""
        return not pitchers_df[pitchers_df['Pitcher'] == pitcher_name].empty
    
    for index, row in betting_data.iterrows():
        pitcher_name = row['Name_abbreviation']
        opponent_team = row['Opponent'] if row['Opponent'] != 'N/A' else None
        strikeout_line = row['Over Line']  # Use Over Line as the betting line
        
        has_data = has_historical_data(pitcher_name)
        betting_data.at[index, 'Has Historical Data'] = has_data
        
        if not has_data:
            print(f"⚠️ No historical data for {pitcher_name}")
            continue
        
        # Make prediction
        result = predict_strikeouts_with_confidence(
            model=model,
            pitchers_df=pitchers_df,
            pitcher_name=pitcher_name,
            opponent_team=opponent_team,
            strikeout_line=strikeout_line
        )
        
        if result:
            betting_data.at[index, 'ML Predict Value'] = result['predicted_value']
            betting_data.at[index, 'ML Recommend Side'] = result['recommended_side']
            betting_data.at[index, 'ML Confidence Percentage'] = result['confidence_percentage']
            
            # Print progress
            print(f"{pitcher_name} vs {opponent_team or 'Unknown'}: "
                  f"Line {strikeout_line:.1f} → Pred {result['predicted_value']:.1f} "
                  f"({result['recommended_side']}, {result['confidence_percentage']:.0f}%)")
        else:
            print(f"⚠️ No prediction for {pitcher_name} vs {opponent_team or 'Unknown'} - prediction failed")
    
    # Filter rows with valid predictions
    filtered_data = betting_data.dropna(subset=['ML Predict Value'])
    
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
    
    return filtered_data

def get_top_picks(predictions_df, n=10, verbose=True):
    """
    Get the top n highest confidence picks from the predictions DataFrame.
    
    Args:
        predictions_df (pd.DataFrame): DataFrame containing predictions
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
    
    # Get top n picks
    top_picks = predictions_df.sort_values('ML Confidence Percentage', ascending=False).head(n)
    
    # Create formatted string
    formatted_output = ""
    if verbose:
        formatted_output = f"\nTop {n} highest confidence picks:\n\n"
        for _, pick in top_picks.iterrows():
            formatted_output += (f"{pick['Player']} ({pick['Team']}): |"
                                f"Over Odds: {pick['Over Odds']:.2f} | "
                                f"Under Odds: {pick['Under Odds']:.2f} | "
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
        print("Loading cleaned pitcher data...")
        
        # Load the cleaned pitcher data
        pitchers_df = pd.read_csv('pitchers_data_with_opp_so_cleaned.csv')
        print(f"Loaded data for {pitchers_df['Pitcher'].nunique()} pitchers")
        print(f"Total records: {len(pitchers_df)}")
        
        # Train the model
        print("\nTraining machine learning models...")
        from src.model.model_training import train_model
        model, results = train_model(pitchers_df)
        
        # Print model performance summary
        print("\nModel performance summary:")
        best_model_name = max(results, key=lambda k: results[k]['CV_R2'])
        print(f"Best model: {best_model_name}")
        print(f"CV R2 Score: {results[best_model_name]['CV_R2']:.4f}")
        print(f"Test R2 Score: {results[best_model_name]['Test_R2']:.4f}")
        print(f"Test MAE: {results[best_model_name]['Test_MAE']:.4f}")
        
        # Find betting data file
        betting_files = [f for f in os.listdir('.') if f.startswith('betting_data_') and f.endswith('.csv')]
        if not betting_files:
            print("No betting data files found!")
            return
        
        # Use the most recent betting file
        betting_files.sort(reverse=True)
        betting_file = betting_files[0]
        print(f"\nUsing betting file: {betting_file}")
        
        # Process betting data with predictions
        print("\nProcessing betting data and making predictions...")
        predictions = process_betting_data(
            model=model,
            pitchers_df=pitchers_df,
            betting_data_path=betting_file
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
        
    except Exception as e:
        print(f"Error occurred during prediction workflow: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()