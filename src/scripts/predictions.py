import numpy as np
from feature_engineering import calculate_weighted_performance

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
    import pandas as pd
    import numpy as np
    from datetime import datetime
    import os
    import re
    
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
    
    # FILTRAR APENAS LINHAS COM PREVISÕES VÁLIDAS
    filtered_data = betting_data.dropna(subset=['ML Predict Value'])
    
    # REMOVER COLUNAS INDESEJADAS
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
    output_filename = f"betting_data_predicted_{file_date}.csv"
    
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