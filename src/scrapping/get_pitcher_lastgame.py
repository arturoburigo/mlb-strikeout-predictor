# %%
import pandas as pd
import os
import time
import random
import glob
from datetime import datetime

# Read the most recent betting_data file
betting_files = glob.glob('betting_data_*.csv')
    
if not betting_files:
    # Fallback to the default name if no date-specific files found
    betting_file = 'betting_data.csv'
    print(f"Using default file: {betting_file}")
else:
    # Sort files by date (assuming format betting_data_YYYY-MM-DD.csv)
    betting_files.sort(key=lambda x: datetime.strptime(x.split('_')[2].split('.')[0], '%Y-%m-%d'), reverse=True)
    betting_file = betting_files[0]
    print(f"Using most recent file: {betting_file}")

betting_data_df = pd.read_csv(betting_file)


season = 2025
print(f'Searching for data for season {season}')

# Create the list of pitchers from the Name_abbreviation column
pitchers = betting_data_df['Name_abbreviation'].tolist()
print(f'Number of pitchers: {len(pitchers)}')

# List to store the last game data for each pitcher
last_games = []

# Function to load the last game of a pitcher
def load_last_pitcher_game(pitcher, season):
    for id_suffix in ['01', '02', '03', '04']:
        url = f"https://www.baseball-reference.com/players/gl.fcgi?id={pitcher}{id_suffix}&t=p&year={season}"
        print(f"Trying URL: {url}")
        try:
            # Load pitcher's game log table
            pitcher_gl = pd.read_html(url, header=0, attrs={'id': 'pitching_gamelogs'})[0]
            
            # Check if there are valid data
            if pitcher_gl.empty:
                print(f"Empty table for {pitcher} with ID {id_suffix}")
                continue
                
            # Filter only rows with numeric Rk (valid game data)
            pitcher_gl = pitcher_gl[pitcher_gl['Rk'].apply(lambda x: str(x).isdigit())]
            
            if pitcher_gl.empty:
                print(f"No valid games for {pitcher} with ID {id_suffix}")
                continue
                
            # Get the last game (highest Rk value)
            last_game = pitcher_gl.loc[pitcher_gl['Rk'].astype(int).idxmax()]
            
            # Add pitcher and season information
            last_game['Season'] = season
            last_game['Pitcher'] = pitcher.lower()
            
            print(f"Last game found for {pitcher} in season {season}")
            return last_game.to_dict()
            
        except (ValueError, IndexError) as e:
            print(f"Error fetching data for {pitcher} with ID {id_suffix}: {e}")
            time.sleep(random.randint(3, 5))  # Reduce wait time
    
    print(f"No data found for {pitcher} in season {season}")
    return None

# Search for the last game of each pitcher
for pitcher in pitchers:
    last_game = load_last_pitcher_game(pitcher, season)
    if last_game:
        last_games.append(last_game)
    # Random wait to avoid server overload
    time.sleep(random.randint(3, 5))  # Reduce wait time

# Create DataFrame with the last games
if last_games:
    last_games_df = pd.DataFrame(last_games)
    
    # Select relevant columns, if present in the DataFrame
    columns_to_keep = [
        'Season',   # Season
        'Pitcher',  # Pitcher name
        'Date',     # Game date
        'Opp',      # Opponent
        'Home',     # If the game is home (1) or away (0)
        'IP',       # Innings pitched
        'H',        # Hits allowed
        'BB',       # Walks allowed
        'SO',       # Strikeouts
        'ERA',      # Earned Run Average
        'FIP',      # Fielding Independent Pitching
        'GB',       # Ground Balls
        'FB',       # Fly Balls
        'LD',       # Line Drives
        'PU',       # Pop Ups
        'WPA'       # Win Probability Added
    ]
    
    # Keep only columns that exist in the DataFrame
    available_columns = [col for col in columns_to_keep if col in last_games_df.columns]
    last_games_df = last_games_df[available_columns]
    
    # Process the Home column if it exists and has value 'Unnamed: 5'
    if 'Unnamed: 5' in last_games_df.columns:
        last_games_df.rename(columns={'Unnamed: 5': 'Home'}, inplace=True)
        last_games_df['Home'] = last_games_df['Home'].fillna(0)
        last_games_df['Home'] = last_games_df['Home'].apply(lambda x: 1 if x == '@' else 0)
        last_games_df['Home'] = last_games_df['Home'].astype(int)
    
    # Print the result
    print("\nLast games for each pitcher in 2024:")
    print(last_games_df)
    
    # Also print statistical information
    print("\nGeneral statistics:")
    print(last_games_df.describe())
else:
    print("No last game data was found for any pitcher.")

# Get today's date for the prediction file
today_date = datetime.now().strftime("%Y-%m-%d")
csv_filename = f"data_predicted_{today_date}.csv"

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

# Read the predicted data
predicted_df = pd.read_csv(csv_filename)

# Check if we have last games data
if not last_games:
    print("Warning: No pitcher last game data available. Creating file without strikeout information.")
    predicted_df.to_csv('game_results.csv', index=False)
    print("Created game_results.csv without strikeout data")
else:
    # Create a dictionary mapping pitcher names to their last game SO (strikeouts)
    pitcher_to_so = {}
    for game in last_games:
        pitcher = game['Pitcher'].lower()
        if 'SO' in game:
            pitcher_to_so[pitcher] = game['SO']
        else:
            print(f"Warning: No strikeout data for pitcher {pitcher}")
            pitcher_to_so[pitcher] = None

    # Add the last game strikeout column to the predicted data
    predicted_df['game_strikeout'] = predicted_df['Name_abbreviation'].str.lower().map(pitcher_to_so)

    # Save the combined data
    predicted_df.to_csv('game_results.csv', index=False)
    print("Successfully created game_results.csv with last game strikeout data")
    print(f"Added strikeout data for {len(pitcher_to_so)} pitchers")
