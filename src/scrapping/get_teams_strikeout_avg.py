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

def collect_team_strikeout_data(start_year, end_year):
    """
    Collect and analyze team strikeout data for the specified year range.
    
    Args:
        start_year (int): Starting year for data collection
        end_year (int): Ending year for data collection (inclusive)
        
    Returns:
        pd.DataFrame: DataFrame containing team strikeout percentages with teams as rows and years as columns
    """
    seasons = [str(season) for season in range(start_year, end_year + 1)]
    teams = ['OAK', 'TOR', 'KCR', 'SFG', 'MIA', 'TBR', 'BOS', 'CLE', 'CHC', 'NYY',
             'CHW', 'LAA', 'BAL', 'TEX', 'CIN', 'SEA', 'COL', 'HOU', 'ATL', 'DET',
             'ARI', 'MIN', 'PIT', 'MIL', 'PHI', 'LAD', 'SDP', 'NYM', 'WSN', 'STL']

    batting_team_df = pd.DataFrame()

    for season in seasons:
        for team in teams:
            # Use the function to get the correct team identifier for URL
            team_url_id = get_team_url_identifier(team, season)
            url = f"https://www.baseball-reference.com/teams/{team_url_id}/{season}.shtml"
            print(f"Collecting data from: {url}")
            bt_df = pd.read_html(url, header=0, attrs={'id': 'players_standard_batting'})[0]
            bt_df.insert(loc=0, column='Season', value=season)
            bt_df.insert(loc=1, column='Team', value=team)  # Always save as original team name (OAK)
            batting_team_df = pd.concat([batting_team_df, bt_df], ignore_index=True)
            time.sleep(random.randint(7, 8))

    # Convertendo colunas relevantes para numérico, tratando erros
    batting_team_df['SO'] = pd.to_numeric(batting_team_df['SO'], errors='coerce')
    batting_team_df['R'] = pd.to_numeric(batting_team_df['R'], errors='coerce')

    # Criar DataFrame com times como linhas e anos como colunas
    results_data = {}
    
    for team in teams:
        results_data[team] = {}
        for season in seasons:
            team_data = batting_team_df[batting_team_df['Team'] == team]
            season_data = team_data[team_data['Season'] == season]
            
            total_SO = season_data['SO'].sum()
            total_AB = season_data['R'].sum() + season_data['SO'].sum()
            
            k_percentage = (total_SO / total_AB * 100) if total_AB > 0 else 0
            results_data[team][season] = k_percentage

    # Criar DataFrame final
    results_df = pd.DataFrame(results_data).T  # Transpose para ter times como linhas
    results_df.index.name = 'Team'
    results_df.reset_index(inplace=True)
    
    # Salvar CSV
    results_df.to_csv('team_strikeout_percentage.csv', index=False)
    print("Team strikeout percentage matrix saved!")
    
    return results_df

if __name__ == "__main__":
    results = collect_team_strikeout_data(2023, 2025)
    print(results)


