import pandas as pd
import numpy as np

def calculate_simple_performance(pitcher_data, current_season):
    """
    Calculate simple performance metrics for a pitcher based on current season data.
    Focuses on the most important features without complex weighting.
    
    Args:
        pitcher_data (DataFrame): Historical data for a specific pitcher
        current_season (int): Current season year
        
    Returns:
        dict: Dictionary of performance metrics
    """
    # Get current season data
    current_season_data = pitcher_data[pitcher_data['Season'] == current_season].copy()
    
    # Check if we have current season data
    if current_season_data.empty:
        return {}
    
    # Calculate simple averages for key metrics
    performance_metrics = {}
    
    # Basic pitching metrics
    performance_metrics['avg_innings_pitched'] = current_season_data['IP'].mean()
    performance_metrics['avg_strikeouts'] = current_season_data['SO'].mean()
    performance_metrics['avg_walks'] = current_season_data['BB'].mean()
    performance_metrics['avg_era'] = current_season_data['ERA'].mean()
    performance_metrics['avg_fip'] = current_season_data['FIP'].mean()
    performance_metrics['avg_batters_faced'] = current_season_data['BF'].mean()
    performance_metrics['avg_total_pitches'] = current_season_data['Pit'].mean()
    performance_metrics['avg_strikes_thrown'] = current_season_data['Str'].mean()
    performance_metrics['avg_swinging_strikes'] = current_season_data['StS'].mean()
    performance_metrics['avg_looking_strikes'] = current_season_data['StL'].mean()
    
    # Calculate derived metrics
    if performance_metrics['avg_innings_pitched'] > 0:
        performance_metrics['strikeouts_per_inning'] = performance_metrics['avg_strikeouts'] / performance_metrics['avg_innings_pitched']
        performance_metrics['walks_per_inning'] = performance_metrics['avg_walks'] / performance_metrics['avg_innings_pitched']
        performance_metrics['k_minus_bb_rate'] = performance_metrics['strikeouts_per_inning'] - performance_metrics['walks_per_inning']
    else:
        performance_metrics['strikeouts_per_inning'] = 0
        performance_metrics['walks_per_inning'] = 0
        performance_metrics['k_minus_bb_rate'] = 0
    
    if performance_metrics['avg_batters_faced'] > 0:
        performance_metrics['strikeouts_per_batter'] = performance_metrics['avg_strikeouts'] / performance_metrics['avg_batters_faced']
        performance_metrics['swinging_strike_rate'] = performance_metrics['avg_swinging_strikes'] / performance_metrics['avg_batters_faced']
    else:
        performance_metrics['strikeouts_per_batter'] = 0
        performance_metrics['swinging_strike_rate'] = 0
    
    if performance_metrics['avg_total_pitches'] > 0:
        performance_metrics['strikeouts_per_pitch'] = performance_metrics['avg_strikeouts'] / performance_metrics['avg_total_pitches']
    else:
        performance_metrics['strikeouts_per_pitch'] = 0
    
    # Recent form (last 5 games)
    if len(current_season_data) >= 5:
        recent_5_games = current_season_data.tail(5)
        performance_metrics['avg_strikeouts_last_5_games'] = recent_5_games['SO'].mean()
    else:
        performance_metrics['avg_strikeouts_last_5_games'] = performance_metrics['avg_strikeouts']
    
    # Season average comparison
    performance_metrics['strikeouts_vs_season_avg'] = performance_metrics['avg_strikeouts'] - current_season_data['SO'].mean()
    performance_metrics['innings_vs_season_avg'] = performance_metrics['avg_innings_pitched'] - current_season_data['IP'].mean()
    
    # Consistency (standard deviation of SO)
    performance_metrics['strikeout_consistency'] = current_season_data['SO'].std()
    
    # Workload (total innings pitched)
    performance_metrics['total_innings_pitched_season'] = current_season_data['IP'].sum()
    
    # Game order (number of games played)
    performance_metrics['games_played_this_season'] = len(current_season_data)
    
    # Momentum (trend in last 3 games)
    if len(current_season_data) >= 3:
        last_3_games = current_season_data.tail(3)
        if len(last_3_games) >= 2:
            performance_metrics['strikeout_momentum_last_3_games'] = last_3_games['SO'].iloc[-1] - last_3_games['SO'].iloc[0]
        else:
            performance_metrics['strikeout_momentum_last_3_games'] = 0
    else:
        performance_metrics['strikeout_momentum_last_3_games'] = 0
    
    return performance_metrics

def main(pitchers_df, k_percentage_df=None):
    """
    Main function to calculate performance metrics for pitchers.
    
    Args:
        pitchers_df (DataFrame): DataFrame containing pitcher data
        k_percentage_df (DataFrame): (Unused) DataFrame containing strikeout percentage data
        
    Returns:
        DataFrame: Engineered features DataFrame with original data preserved
    """
    try:
        # Define current season
        current_season = 2025
        
        # Get unique pitchers
        unique_pitchers = pitchers_df['Pitcher_Name'].unique()
        print(f"Found {len(unique_pitchers)} unique pitchers in the dataset")
        
        # Calculate performance for each pitcher
        all_performance_stats = {}
        opp_kpct = {}
        for pitcher in unique_pitchers:
            pitcher_records = pitchers_df[pitchers_df['Pitcher_Name'] == pitcher]
            
            # Only process pitchers with data in the current season
            if current_season in pitcher_records['Season'].values:
                performance_stats = calculate_simple_performance(
                    pitcher_records, 
                    current_season
                )
                if performance_stats:  # Only add if we got results
                    all_performance_stats[pitcher] = performance_stats
                    # Use mean opp_so_avg for this pitcher in current season
                    opp_kpct[pitcher] = pitcher_records[pitcher_records['Season'] == current_season]['opp_so_avg'].mean()
        
        # Convert the dictionary to a DataFrame
        engineered_data = pd.DataFrame.from_dict(all_performance_stats, orient='index')
        
        # Add Opp_K% from opp_so_avg
        engineered_data['opponent_strikeout_percentage'] = engineered_data.index.map(opp_kpct)
        
        # Ensure all required features exist and handle NaN values
        required_features = [
            'avg_innings_pitched', 'avg_strikeouts', 'avg_walks', 'avg_era', 'avg_fip', 
            'strikeouts_per_inning', 'walks_per_inning', 'k_minus_bb_rate', 
            'opponent_strikeout_percentage', 'avg_batters_faced', 'avg_total_pitches',
            'avg_strikes_thrown', 'avg_swinging_strikes', 'avg_looking_strikes',
            'strikeouts_per_batter', 'swinging_strike_rate', 'strikeouts_per_pitch', 
            'avg_strikeouts_last_5_games', 'strikeouts_vs_season_avg',
            'innings_vs_season_avg', 'strikeout_consistency', 'total_innings_pitched_season', 
            'games_played_this_season', 'strikeout_momentum_last_3_games'
        ]
        
        for feature in required_features:
            if feature not in engineered_data.columns:
                engineered_data[feature] = 0
        
        # Fill NaN values with 0
        engineered_data = engineered_data.fillna(0)
        
        # Final check for any remaining infinity or extremely large values
        for col in engineered_data.columns:
            if engineered_data[col].dtype in [np.float64, np.int64]:
                engineered_data[col] = engineered_data[col].replace([np.inf, -np.inf], 0)
                # Clip extremely large values to reasonable ranges
                if col in ['avg_era', 'avg_fip']:
                    engineered_data[col] = engineered_data[col].clip(0, 20)
                elif col in ['strikeouts_per_inning', 'walks_per_inning', 'strikeouts_per_batter', 'swinging_strike_rate', 'strikeouts_per_pitch']:
                    engineered_data[col] = engineered_data[col].clip(0, 5)
                elif col in ['avg_strikeouts_last_5_games', 'avg_strikeouts']:
                    engineered_data[col] = engineered_data[col].clip(0, 20)
                elif col in ['avg_innings_pitched']:
                    engineered_data[col] = engineered_data[col].clip(0, 10)
                elif col in ['total_innings_pitched_season']:
                    engineered_data[col] = engineered_data[col].clip(0, 200)
                elif col in ['games_played_this_season']:
                    engineered_data[col] = engineered_data[col].clip(0, 50)
                elif col in ['strikeout_momentum_last_3_games']:
                    engineered_data[col] = engineered_data[col].clip(-10, 10)
        
        # Now merge the engineered features back with the original data
        # Reset index to make Pitcher_Name a column
        engineered_data = engineered_data.reset_index()
        engineered_data = engineered_data.rename(columns={'index': 'Pitcher_Name'})
        
        # Merge with original data to preserve all original columns
        final_data = pitchers_df.merge(
            engineered_data,
            on='Pitcher_Name',
            how='left'
        )
        
        # Fill NaN values in engineered features with 0
        engineered_columns = [col for col in engineered_data.columns if col != 'Pitcher_Name']
        for col in engineered_columns:
            if col in final_data.columns:
                final_data[col] = final_data[col].fillna(0)
        
        print(f"\nSuccessfully calculated performance stats for {len(all_performance_stats)} pitchers")
        print(f"Final data shape: {final_data.shape}")
        return final_data
        
    except Exception as e:
        print(f"Error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def engineer_features(csv_path='pitchers_data_with_opp_so.csv', output_path='pitchers_data_engineered.csv'):
    """
    Load data from CSV files and create engineered features.
    
    Args:
        csv_path (str): Path to the pitchers data CSV file
        output_path (str): Path to save the engineered features CSV file
        
    Returns:
        DataFrame: Engineered features dataframe with original data preserved
    """
    try:
        # Load the pitchers data
        pitchers_df = pd.read_csv(csv_path)
        
        # Create engineered features using the main function (no k_percentage_df needed)
        engineered_data = main(pitchers_df)
        
        # Save the engineered features to CSV if output_path is provided
        if output_path and engineered_data is not None:
            engineered_data.to_csv(output_path, index=False)
            print(f"Engineered features saved to {output_path}")
        
        return engineered_data
        
    except Exception as e:
        print(f"Error in engineer_features: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Test the feature engineering
    engineered_data = engineer_features()
    if engineered_data is not None:
        print(f"Feature engineering completed. Shape: {engineered_data.shape}")
        print(f"Original columns preserved: {[col for col in engineered_data.columns if not col.startswith(('avg_', 'strikeouts_', 'walks_', 'k_minus_', 'opponent_', 'swinging_', 'total_', 'games_', 'momentum_'))]}")
        print(f"New engineered features: {[col for col in engineered_data.columns if col.startswith(('avg_', 'strikeouts_', 'walks_', 'k_minus_', 'opponent_', 'swinging_', 'total_', 'games_', 'momentum_'))]}")