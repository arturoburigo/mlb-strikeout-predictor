import pandas as pd
import os
import time
import random
import glob
from datetime import datetime
import requests
from bs4 import BeautifulSoup

# %%
import pandas as pd
import numpy as np
import random
import time

def get_team_url_identifier(team, season):
    """
    Get the correct team identifier for the URL based on team and season.
    Handles special cases like OAK -> ATH for 2025 season.
    
    Args:
        team (str): Team abbreviation
        season (str): Season year
        
    Returns:
        str: Team identifier to use in URL
    """
    if team == 'OAK' and season == '2025':
        return 'ATH'
    return team

def get_pitcher_names_from_url(url):
    """
    Get pitcher names from a team's pitching page.
    
    Args:
        url (str): URL of the team's pitching page
        
    Returns:
        list: List of pitcher names found in data-stat="name_display"
    """
    try:
        # Enhanced headers to mimic a real browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the pitching table
        pitching_table = soup.find('table', {'id': 'players_standard_pitching'})
        
        if not pitching_table:
            print(f"No pitching table found at {url}")
            return []
        
        # Find all elements with data-append-csv attribute
        name_elements = pitching_table.find_all(attrs={'data-append-csv': True})
        
        pitcher_names = []
        for element in name_elements:
            # Get the value of data-append-csv attribute
            csv_value = element.get('data-append-csv', '')
            display_text = element.text.strip()
            
            if display_text:  # Only add non-empty names
                pitcher_names.append({
                    'name': display_text,
                    'csv_value': csv_value
                })
        
        # Also try to find elements with data-stat="name_display" as fallback
        if not pitcher_names:
            name_elements_fallback = pitching_table.find_all(attrs={'data-stat': 'name_display'})
            for element in name_elements_fallback:
                display_text = element.text.strip()
                if display_text:
                    pitcher_names.append({
                        'name': display_text,
                        'csv_value': element.get('data-append-csv', '')
                    })
        
        return pitcher_names
        
    except Exception as e:
        print(f"Error accessing {url}: {e}")
        return []

def get_pitcher_detailed_data(pitcher_csv_value, season):
    """
    Get detailed pitcher data using the CSV value from Baseball Reference.
    Uses the same method as get_pitcher_data_lastseason.py
    
    Args:
        pitcher_csv_value (str): The CSV value (ID) of the pitcher
        season (str): Season year
        
    Returns:
        pd.DataFrame: DataFrame with detailed pitcher data or None if not found
    """
    # Try the URL directly with the pitcher CSV value
    url = f"https://www.baseball-reference.com/players/gl.fcgi?id={pitcher_csv_value}&t=p&year={season}"
    print(f"  Trying URL: {url}")
    
    # Enhanced headers to mimic a real browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0'
    }
    
    try:
        pitcher_gl = pd.read_html(url, header=0, attrs={'id': 'players_standard_pitching'}, request_kwargs={'headers': headers})[0]
        
        pitcher_gl.insert(loc=0, column='Season', value=season)
        pitcher_gl.insert(loc=1, column='Pitcher_ID', value=pitcher_csv_value)
        
        print(f"  Found data for pitcher {pitcher_csv_value}")
        return pitcher_gl
        
    except (ValueError, IndexError):
        print(f"  Data not found for pitcher {pitcher_csv_value} on season {season}")
        time.sleep(7)
        return None

def collect_pitcher_names_and_data():
    """
    Collect pitcher names from 2025 and detailed data from 2023-2025.
    
    Returns:
        dict: Dictionary with team names as keys and lists of pitcher data as values
    """
    # Only collect pitcher names from 2025
    name_season = '2025'
    # Collect detailed data from 2023-2025
    data_seasons = ['2023', '2024', '2025']
    
    teams = ['OAK', 'TOR', 'KCR', 'SFG', 'MIA', 'TBR', 'BOS', 'CLE', 'CHC', 'NYY',
             'CHW', 'LAA', 'BAL', 'TEX', 'CIN', 'SEA', 'COL', 'HOU', 'ATL', 'DET',
             'ARI', 'MIN', 'PIT', 'MIL', 'PHI', 'LAD', 'SDP', 'NYM', 'WSN', 'STL']

    all_pitchers_data = {}
    all_detailed_data = []

    print(f"\n=== Collecting pitcher names from {name_season} ===")
    
    # Step 1: Collect pitcher names from 2025 only
    for team in teams:
        # Use the function to get the correct team identifier for URL
        team_url_id = get_team_url_identifier(team, name_season)
        url = f"https://www.baseball-reference.com/teams/{team_url_id}/{name_season}.shtml"
        print(f"\nCollecting pitcher names from: {url}")
        
        pitcher_names = get_pitcher_names_from_url(url)
        
        if pitcher_names:
            print(f"Found {len(pitcher_names)} pitchers for {team} in {name_season}:")
            
            # Store basic pitcher info
            key = f"{team}_{name_season}"
            all_pitchers_data[key] = pitcher_names
            
            # Step 2: Get detailed data for each pitcher from 2023-2025
            for pitcher in pitcher_names:
                print(f"  Getting detailed data for: {pitcher['name']} (ID: {pitcher['csv_value']})")
                
                for data_season in data_seasons:
                    print(f"    Trying season {data_season}...")
                    
                    detailed_data = get_pitcher_detailed_data(pitcher['csv_value'], data_season)
                    
                    if detailed_data is not None:
                        # Add team and pitcher name info (only if columns don't exist)
                        if 'Team' not in detailed_data.columns:
                            detailed_data.insert(loc=2, column='Team', value=team)
                        else:
                            detailed_data['Team'] = team
                            
                        if 'Pitcher_Name' not in detailed_data.columns:
                            detailed_data.insert(loc=3, column='Pitcher_Name', value=pitcher['name'])
                        else:
                            detailed_data['Pitcher_Name'] = pitcher['name']
                            
                        all_detailed_data.append(detailed_data)
                        
                        print(f"      Added {len(detailed_data)} game records from {data_season}")
                    else:
                        print(f"      No data found for {data_season}")
                    
                    # Delay of 7 seconds between season requests
                    time.sleep(7)
        else:
            print(f"No pitchers found for {team} in {name_season}")
        
        # Delay of 7 seconds between team requests
        time.sleep(7)

    return all_pitchers_data, all_detailed_data

if __name__ == "__main__":
    print("Starting pitcher name and data collection...")
    results, detailed_data = collect_pitcher_names_and_data()
    
    print(f"\n=== SUMMARY ===")
    print(f"Total teams/seasons processed: {len(results)}")
    total_pitchers = sum(len(pitchers) for pitchers in results.values())
    print(f"Total pitchers found: {total_pitchers}")
    
    # Save basic pitcher info to CSV
    all_data = []
    for key, pitchers in results.items():
        team, season = key.split('_')
        for pitcher in pitchers:
            all_data.append({
                'Team': team,
                'Season': season,
                'Pitcher_Name': pitcher['name'],
                'CSV_Value': pitcher['csv_value']
            })
    
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv('pitcher_names.csv', index=False)
        print(f"\nBasic pitcher info saved to 'pitcher_names.csv'")
        print(f"DataFrame shape: {df.shape}")
    
    # Save detailed pitcher data to CSV
    if detailed_data:
        detailed_df = pd.concat(detailed_data, ignore_index=True)
        
        # Clean up the data similar to the original script
        if 'Rk' in detailed_df.columns:
            detailed_df = detailed_df[detailed_df['Rk'].apply(lambda x: str(x).isdigit())]
            detailed_df.reset_index(drop=True, inplace=True)
            print("Lines with non-integer values were removed successfully.")
        
        # Remove unnecessary columns
        columns_to_drop = ['Rk', 'Gcar', 'Tm', 'Gtm', 'Rslt', 'Inngs', 'Dec', 'ER', 'SB', 'CS', 'PO', 'DFS(DK)', 'DFS(FD)']        
        # Rename and process Home column
        detailed_df.rename(columns={'Unnamed: 5': 'Home'}, inplace=True)
        detailed_df['Home'] = detailed_df['Home'].fillna(0)
        detailed_df['Home'] = detailed_df['Home'].apply(lambda x: 1 if x == '@' else 0)
        detailed_df['Home'] = detailed_df['Home'].astype(int)
        
        # Keep only relevant columns
        columns_to_keep = [
            'Season', 'Pitcher_ID', 'Team', 'Pitcher_Name', 'Date', 'Home', 'Opp', 'IP', 'H', 'BB', 'SO',
            'WPA', 'ERA', 'FIP', 'BF', 'Pit', 'Str', 'StS', 'StL', 'cçDR', 'aLI'
        ]
        
        # Only keep columns that exist
        existing_columns = [col for col in columns_to_keep if col in detailed_df.columns]
        detailed_df = detailed_df[existing_columns]
        
        detailed_df.to_csv('pitchers_detailed_data.csv', index=False)
        print(f"\nDetailed pitcher data saved to 'pitchers_detailed_data.csv'")
        print(f"Detailed DataFrame shape: {detailed_df.shape}")
        print(f"Columns: {list(detailed_df.columns)}")
    else:
        print("\nNo detailed data to save.")

