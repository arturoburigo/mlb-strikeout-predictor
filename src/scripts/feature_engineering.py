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

def main():
    """
    Main function to calculate weighted performance metrics for pitchers.
    Loads data from 'pitchers_data.csv' and analyzes 2024 season with 2023 as reference.
    """
    try:
        # Load pitcher data from CSV file
        print("Loading pitcher data from pitchers_data.csv...")
        pitcher_data = pd.read_csv('pitchers_data.csv')
        
        # Define current and previous seasons
        current_season = 2024
        last_season = 2023
        
        # Get unique pitchers
        unique_pitchers = pitcher_data['Pitcher'].unique()
        print(f"Found {len(unique_pitchers)} unique pitchers in the dataset")
        
        # Calculate weighted performance for each pitcher
        all_weighted_stats = {}
        for pitcher in unique_pitchers:
            pitcher_records = pitcher_data[pitcher_data['Pitcher'] == pitcher]
            
            # Only process pitchers with data in the current season
            if current_season in pitcher_records['Season'].values:
                weighted_stats = calculate_weighted_performance(
                    pitcher_records, 
                    current_season, 
                    last_season
                )
                if weighted_stats:  # Only add if we got results
                    all_weighted_stats[pitcher] = weighted_stats
                
        # Display results for a few pitchers as example
        sample_size = min(5, len(all_weighted_stats))
        print(f"\nDisplaying weighted performance metrics for {sample_size} sample pitchers:")
        
        for i, (pitcher, stats) in enumerate(list(all_weighted_stats.items())[:sample_size]):
            print(f"\n{pitcher}:")
            print(f"  Weighted SO: {stats['SO']:.2f}")
            print(f"  Weighted ERA: {stats['ERA']:.2f}")
            
            # Only print rolling stats if they exist
            if 'SO_rolling_5' in stats:
                print(f"  Last 5 games SO avg: {stats['SO_rolling_5']:.2f}")
            
            print(f"  Home/Away split (SO): {stats['Home_SO']:.2f}/{stats['Away_SO']:.2f}")
        
        print(f"\nSuccessfully calculated weighted stats for {len(all_weighted_stats)} pitchers")
        
        # Return the calculated stats for potential further use
        return all_weighted_stats
        
    except Exception as e:
        print(f"Error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()  # Print the full stack trace for debugging
        return None

if __name__ == "__main__":
    main()