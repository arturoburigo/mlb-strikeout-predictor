import pandas as pd
import os
import time
import random
import glob
from datetime import datetime

def get_pitcher_last_season():
    # Get betting files
    betting_files = glob.glob('betting_data_*.csv')
    if not betting_files:
        raise ValueError("No betting files found. Please run betting_odds_today.py first")
    
    # Sort files by date (assuming format betting_data_YYYY-MM-DD.csv)
    betting_files.sort(key=lambda x: datetime.strptime(x.split('_')[2].split('.')[0], '%Y-%m-%d'), reverse=True)
    betting_file = betting_files[0]
    print(f"Using the most recent betting data: {betting_file}")
    betting_data_df = pd.read_csv(betting_file)
    
    seasons = [str(season) for season in range(2023, 2026)]
    print(f'number of seasons={len(seasons)}')

    pitchers = betting_data_df['Name_abbreviation'].tolist()
    print(f'number of pitchers={len(pitchers)}')

    dataframes = []

    if os.path.exists('pitchers_data.csv'):
        existing_df = pd.read_csv('pitchers_data.csv')  
    else:
        existing_df = pd.DataFrame(columns=['Season', 'Pitcher'])  

    def get_pitcher_data(pitcher, season):
        for id_suffix in ['01', '02', '03', '04']:
            url = f"https://www.baseball-reference.com/players/gl.fcgi?id={pitcher}{id_suffix}&t=p&year={season}"
            print(url)
            try:
                pitcher_gl = pd.read_html(url, header=0, attrs={'id': 'players_standard_pitching'})[0]

                pitcher_gl.insert(loc=0, column='Season', value=season)

                pitcher_gl.insert(loc=2, column='Pitcher', value=pitcher.lower())

                return pitcher_gl 
            except (ValueError, IndexError):
                print(f"Data not found for this pitcher {pitcher.lower()} with ID {id_suffix} on season {season}. trying next...")
                time.sleep(random.randint(7, 8))

        return None 

    for season in seasons:
        for pitcher in pitchers:
            pitcher_data = get_pitcher_data(pitcher, season)
            
            if pitcher_data is not None:
                if pitcher_data['Pitcher'].iloc[0] not in existing_df['Pitcher'].values:
                    dataframes.append(pitcher_data)
                else:
                    print(f"The pitcher {pitcher.lower()} already exists in the CSV. Ignoring...")
            else:
                print(f"Can't find {pitcher.lower()} on season {season}.")

            time.sleep(random.randint(7, 8))

    if dataframes:
        new_pitchers_df = pd.concat(dataframes, ignore_index=True)
        if 'Rk' in new_pitchers_df.columns:
            new_pitchers_df = new_pitchers_df[new_pitchers_df['Rk'].apply(lambda x: str(x).isdigit())]
            new_pitchers_df.reset_index(drop=True, inplace=True)
            print("Lines with non-integer values were removed successfully.")
        else:
            print("The column 'Rk' doesn't exist in the DataFrame.")

        # Colunas a serem removidas do dataframe
        new_pitchers_df.drop(columns=['Rk', 'Gcar', 'Tm', 'Gtm', 'Date', 'Rslt', 'Inngs', 'Dec', 'ER', 'SB', 'CS', 'PO', 'DFS(DK)', 'DFS(FD)'], inplace=True, errors='ignore')

        # Renomeando a coluna 'Unnamed: 5' para 'Home'
        new_pitchers_df.rename(columns={'Unnamed: 5': 'Home'}, inplace=True)

        # Preenchendo valores NaN com 0
        new_pitchers_df['Home'] = new_pitchers_df['Home'].fillna(0)

        # Aplicando a função lambda para mudar '@' para 1 e manter os outros valores
        new_pitchers_df['Home'] = new_pitchers_df['Home'].apply(lambda x: 1 if x == '@' else 0)

        # Convertendo a coluna 'Home' para tipo int (se ainda não estiver)
        new_pitchers_df['Home'] = new_pitchers_df['Home'].astype(int)

        # Exibindo as informações do DataFrame
        print(new_pitchers_df.info(verbose=True))

        # Colunas a manter, incluindo a coluna Season

        columns_to_keep = [
            'Season',   # Temporada
            'Pitcher',  # Nome do pitcher
            'Home',     # Se o jogo é em casa (1) ou fora (0)
            'Opp',      # Oponente
            'IP',       # Innings pitched
            'H',        # Hits permitidos
            'BB',       # Walks permitidos
            'SO',       # Strikeouts (número de strikeouts do pitcher)
            'WPA',       # Win Probability Added
            'ERA',      # Earned Run Average
            'FIP',      # Fielding Independent Pitching
            'BF',       # Batters faced
            'Pit',      # Pitches thrown
            'Str',      # Strikes
            'StS',      # Strike swinging
            'StL',      # Strike looking
            'DR',       # Days rest
            'aLI',       # Average leverage index
        ]

        # Dropar as colunas que não estão na lista
        new_pitchers_df = new_pitchers_df[columns_to_keep]

        # Exibir informações do DataFrame após o drop
        print(new_pitchers_df.info(verbose=True))

        # Concatenar com os dados existentes
        combined_df = pd.concat([existing_df, new_pitchers_df], ignore_index=True)

        # Salvando o DataFrame em um arquivo CSV
        combined_df.to_csv('pitchers_data.csv', index=False)

        print("CSV saved successfully!")
    else:
        print("No new data to save.")

if __name__ == "__main__":
    get_pitcher_last_season()

