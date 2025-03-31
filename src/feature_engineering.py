import pandas as pd

def calculate_weighted_performance(pitcher_data, current_season, last_season=None):
    """
    Calculate weighted performance metrics for a pitcher based on current and previous seasons.
    
    Args:
        pitcher_data (DataFrame): Historical data for a specific pitcher
        current_season (int): Current season year
        last_season (int, optional): Previous season year
        
    Returns:
        dict: Dictionary of weighted performance metrics
    """
    # Create a copy to avoid the SettingWithCopyWarning
    current_season_data = pitcher_data[pitcher_data['Season'] == current_season].copy()
    
    # Check if we have current season data
    if current_season_data.empty:
        return {}
        
    # Get last 5 and 10 games data
    last_5_games = current_season_data.tail(5)
    last_10_games = current_season_data.tail(10)
    
    # Calculate rolling averages correctly using .loc
    current_season_data.loc[:, 'SO_rolling_5'] = current_season_data['SO'].rolling(5).mean()
    current_season_data.loc[:, 'SO_rolling_10'] = current_season_data['SO'].rolling(10).mean()
    
    # Calculate home/away splits
    home_stats = current_season_data[current_season_data['Home'] == 1.0].mean(numeric_only=True)
    away_stats = current_season_data[current_season_data['Home'] == 0.0].mean(numeric_only=True)

    if last_season is not None:
        last_season_data = pitcher_data[pitcher_data['Season'] == last_season]
        weight_current_season = 0.40
        weight_last_5_games = 0.25
        weight_last_10_games = 0.15
        weight_last_season = 0.20
    else:
        last_season_data = pd.DataFrame()
        weight_current_season = 0.50
        weight_last_5_games = 0.30
        weight_last_10_games = 0.20
        weight_last_season = 0.0

    # Define metrics only if they exist in our dataframes
    base_metrics = ['IP', 'H', 'BB', 'ERA', 'FIP', 'SO']
    rolling_metrics = []
    
    # Only add rolling metrics if they're successfully calculated
    if not current_season_data['SO_rolling_5'].isna().all():
        rolling_metrics.append('SO_rolling_5')
    if not current_season_data['SO_rolling_10'].isna().all():
        rolling_metrics.append('SO_rolling_10')
    
    metrics = base_metrics + rolling_metrics
    weighted_values = {}

    for metric in metrics:
        # Ensure last 5 and last 10 have the metric (they might not have rolling averages)
        metric_in_last5 = metric in last_5_games.columns
        metric_in_last10 = metric in last_10_games.columns
        metric_in_lastseason = metric in last_season_data.columns
        
        current_mean = current_season_data[metric].mean() if not current_season_data.empty else 0
        last_5_mean = last_5_games[metric].mean() if not last_5_games.empty and metric_in_last5 else 0
        last_10_mean = last_10_games[metric].mean() if not last_10_games.empty and metric_in_last10 else 0
        last_season_mean = last_season_data[metric].mean() if not last_season_data.empty and metric_in_lastseason else 0

        weighted_values[metric] = (
            weight_current_season * current_mean +
            weight_last_5_games * last_5_mean +
            weight_last_10_games * last_10_mean +
            weight_last_season * last_season_mean
        )
    
    # Add home/away splits
    weighted_values['Home_IP'] = home_stats.get('IP', 0)
    weighted_values['Away_IP'] = away_stats.get('IP', 0)
    weighted_values['Home_SO'] = home_stats.get('SO', 0)
    weighted_values['Away_SO'] = away_stats.get('SO', 0)
    weighted_values['Home'] = current_season_data['Home'].mean()
    
    return weighted_values

def main(pitchers_df, k_percentage_df):
    """
    Main function to calculate weighted performance metrics for pitchers.
    
    Args:
        pitchers_df (DataFrame): DataFrame containing pitcher data
        k_percentage_df (DataFrame): DataFrame containing strikeout percentage data
        
    Returns:
        DataFrame: Engineered features DataFrame
    """
    try:
        # Define current and previous seasons
        current_season = 2024
        last_season = 2023
        
        # Get unique pitchers
        unique_pitchers = pitchers_df['Pitcher'].unique()
        print(f"Found {len(unique_pitchers)} unique pitchers in the dataset")
        
        # Calculate weighted performance for each pitcher
        all_weighted_stats = {}
        for pitcher in unique_pitchers:
            pitcher_records = pitchers_df[pitchers_df['Pitcher'] == pitcher]
            
            # Only process pitchers with data in the current season
            if current_season in pitcher_records['Season'].values:
                weighted_stats = calculate_weighted_performance(
                    pitcher_records, 
                    current_season, 
                    last_season
                )
                if weighted_stats:  # Only add if we got results
                    all_weighted_stats[pitcher] = weighted_stats
        
        # Convert the dictionary to a DataFrame
        engineered_data = pd.DataFrame.from_dict(all_weighted_stats, orient='index')
        
        # Add derived features
        engineered_data['IP'] = engineered_data['IP'].replace(0, 1)  # Avoid division by zero
        engineered_data['SO_per_IP'] = engineered_data['SO'] / engineered_data['IP']
        engineered_data['BB_per_IP'] = engineered_data['BB'] / engineered_data['IP']
        engineered_data['K-BB%'] = engineered_data['SO_per_IP'] - engineered_data['BB_per_IP']
        
        # Merge with k_percentage_df if needed
        if not k_percentage_df.empty:
            engineered_data = engineered_data.merge(
                k_percentage_df,
                left_index=True,
                right_index=True
            )
            engineered_data.rename(columns={'%K': 'Team_K%'}, inplace=True)
            engineered_data['Opp_K%'] = engineered_data['Team_K%']  # Use same value for now
        
        # Ensure all required features exist
        required_features = [
            'IP', 'H', 'BB', 'ERA', 'FIP', 'SO_per_IP', 'BB_per_IP', 'K-BB%', 
            'Opp_K%', 'Team_K%', 'Home', 'SO_rolling_5', 'SO_rolling_10',
            'Home_IP', 'Away_IP', 'Home_SO', 'Away_SO'
        ]
        
        for feature in required_features:
            if feature not in engineered_data.columns:
                engineered_data[feature] = 0
        
        print(f"\nSuccessfully calculated weighted stats for {len(all_weighted_stats)} pitchers")
        return engineered_data
        
    except Exception as e:
        print(f"Error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()  # Print the full stack trace for debugging
        return None

if __name__ == "__main__":
    # Load data first
    from data_utils import load_data
    pitchers_df, k_percentage_df, _ = load_data()
    main(pitchers_df, k_percentage_df)