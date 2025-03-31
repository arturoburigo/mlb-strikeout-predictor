import pandas as pd
import glob
from datetime import datetime

def load_data():
    """
    Loads and preprocesses the most recent betting data along with pitcher and team statistics.
    
    Returns:
        tuple: (pitcher_data, k_percentage_df, betting_file) containing the preprocessed data
    """
    # Find the most recent betting data file
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
    
    # Load all datasets
    k_percentage_df = pd.read_csv('team_strikeout_percentage.csv')
    pitcher_data = pd.read_csv('pitchers_data.csv')
    betting_data = pd.read_csv(betting_file)
    
    # Merge and preprocess data
    pitcher_data = pitcher_data.merge(
        betting_data[['Name_abbreviation', 'Team']], 
        left_on='Pitcher', 
        right_on='Name_abbreviation', 
        how='left'
    )
    
    # Feature engineering
    pitcher_data['SO_per_IP'] = pitcher_data['SO'] / pitcher_data['IP']
    pitcher_data['BB_per_IP'] = pitcher_data['BB'] / pitcher_data['IP']
    pitcher_data['K-BB%'] = pitcher_data['SO_per_IP'] - pitcher_data['BB_per_IP']
    
    # Merge with team stats
    pitcher_data = pitcher_data.merge(k_percentage_df, on='Team', how='left')
    pitcher_data.rename(columns={'%K': 'Team_K%'}, inplace=True)
    pitcher_data = pitcher_data.merge(
        k_percentage_df.rename(columns={'%K': 'Opp_K%'}), 
        left_on='Opp', right_on='Team', how='left'
    )
    
    # Handle missing values
    pitcher_data.fillna({
        'SO_per_IP': pitcher_data['SO_per_IP'].mean(),
        'BB_per_IP': pitcher_data['BB_per_IP'].mean(),
        'Team_K%': pitcher_data['Team_K%'].mean(),
        'Opp_K%': pitcher_data['Opp_K%'].mean()
    }, inplace=True)
    
    return pitcher_data, k_percentage_df, betting_file

def main():
    """
    Main function to execute the data loading and preprocessing.
    """
    try:
        # Load the preprocessed data
        pitcher_data, k_percentage_df, betting_file = load_data()
        
        # Print some information about the loaded data
        print(f"Data loaded successfully from {betting_file}")
        print(f"Pitcher data shape: {pitcher_data.shape}")
        print(f"Team strikeout percentage data shape: {k_percentage_df.shape}")
        
        # Display the first few rows of each dataset
        #print("\nFirst 5 rows of pitcher data:")
        #print(pitcher_data.head())
        
        #print("\nTeam strikeout percentages:")
        #print(k_percentage_df.head())
        
        # You can add more code here to perform analysis, model training, etc.
        
    except Exception as e:
        print(f"Error occurred during data loading: {e}")

if __name__ == "__main__":
    main()