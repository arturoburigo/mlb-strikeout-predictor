# %%
import pandas as pd
import numpy as np
import random
import time

def collect_team_strikeout_data(start_year, end_year):
    """
    Collect and analyze team strikeout data for the specified year range.
    
    Args:
        start_year (int): Starting year for data collection
        end_year (int): Ending year for data collection (inclusive)
        
    Returns:
        pd.DataFrame: DataFrame containing team strikeout percentages
    """
    # Definindo as temporadas e equipes
    seasons = [str(season) for season in range(start_year, end_year + 1)]
    teams = ['OAK', 'TOR', 'KCR', 'SFG', 'MIA', 'TBR', 'BOS', 'CLE', 'CHC', 'NYY',
             'CHW', 'LAA', 'BAL', 'TEX', 'CIN', 'SEA', 'COL', 'HOU', 'ATL', 'DET',
             'ARI', 'MIN', 'PIT', 'MIL', 'PHI', 'LAD', 'SDP', 'NYM', 'WSN', 'STL']

    # Inicializando um DataFrame vazio para coletar os dados
    batting_team_df = pd.DataFrame()

    # Coletando dados para todas as equipes
    for season in seasons:
        for team in teams:
            url = f"https://www.baseball-reference.com/teams/{team}/{season}.shtml#all_players_standard_batting"
            print(f"Coletando dados de: {url}")
            bt_df = pd.read_html(url, header=0, attrs={'id': 'players_standard_batting'})[0]
            bt_df.insert(loc=0, column='Season', value=season)
            bt_df.insert(loc=1, column='Team', value=team)
            batting_team_df = pd.concat([batting_team_df, bt_df], ignore_index=True)
            time.sleep(random.randint(7, 8))

    # Convertendo colunas relevantes para numérico, tratando erros
    batting_team_df['SO'] = pd.to_numeric(batting_team_df['SO'], errors='coerce')
    batting_team_df['R'] = pd.to_numeric(batting_team_df['R'], errors='coerce')

    results = []
    total_seasons = len(seasons)

    # Filtrando e processando os dados
    for team in teams:
            team_data = batting_team_df[batting_team_df['Team'] == team]
            weighted_k_sum = 0
            total_weight = 0
            
            # Calculate weights for each season
            weights = []
            for i in range(len(seasons)):
                if i == len(seasons) - 1:  # Most recent season
                    weights.append(0.85)  # 85% weight for most recent season
                else:
                    # Distribute remaining 15% among older seasons proportionally
                    remaining_weight = 0.15 / (len(seasons) - 1)
                    weights.append(remaining_weight)
            
            for idx, season in enumerate(seasons):
                season_data = team_data[team_data['Season'] == season]
                total_SO = season_data['SO'].sum()
                total_AB = season_data['R'].sum() + season_data['SO'].sum()
                
                if total_AB > 0:
                    k_percentage = total_SO / total_AB
                    weight = weights[idx]
                    weighted_k_sum += k_percentage * weight
                    total_weight += weight
            
            final_weighted_K = weighted_k_sum / total_weight if total_weight > 0 else 0
            results.append({'Team': team, '%K': final_weighted_K})

        # Convertendo a lista de resultados em um DataFrame
    results_df = pd.DataFrame(results)
    
    # Salvando os resultados em um novo arquivo CSV
    results_df.to_csv('team_strikeout_percentage.csv', index=False)
    print("Previsão da %K dos times foi salva com sucesso!")
    
    return results_df

# Example usage:
if __name__ == "__main__":
    results = collect_team_strikeout_data(2022, 2023)
    print(results)


