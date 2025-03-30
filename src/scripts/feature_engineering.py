import pandas as pd
import numpy as np

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
    current_season_data = pitcher_data[pitcher_data['Season'] == current_season]
    last_5_games = current_season_data.tail(5)
    last_10_games = current_season_data.tail(10)
    
    # Calculate rolling averages
    current_season_data['SO_rolling_5'] = current_season_data['SO'].rolling(5).mean()
    current_season_data['SO_rolling_10'] = current_season_data['SO'].rolling(10).mean()
    
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

    metrics = ['IP', 'H', 'BB', 'ERA', 'FIP', 'SO', 'SO_rolling_5', 'SO_rolling_10']
    weighted_values = {}

    for metric in metrics:
        current_mean = current_season_data[metric].mean() if not current_season_data.empty else 0
        last_5_mean = last_5_games[metric].mean() if not last_5_games.empty else 0
        last_10_mean = last_10_games[metric].mean() if not last_10_games.empty else 0
        last_season_mean = last_season_data[metric].mean() if not last_season_data.empty else 0

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