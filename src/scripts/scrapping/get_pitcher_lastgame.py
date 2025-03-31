# %%
import pandas as pd
import os
import time
import random
import glob
from datetime import datetime

# Ler o arquivo betting_data mais recente
betting_files = glob.glob('betting_data_*.csv')
    
if not betting_files:
    # Fallback to the default name if no date-specific files found
    betting_file = 'betting_data.csv'
    print(f"Usando arquivo padrão: {betting_file}")
else:
    # Sort files by date (assuming format betting_data_YYYY-MM-DD.csv)
    betting_files.sort(key=lambda x: datetime.strptime(x.split('_')[2].split('.')[0], '%Y-%m-%d'), reverse=True)
    betting_file = betting_files[0]
    print(f"Usando arquivo mais recente: {betting_file}")

betting_data_df = pd.read_csv(betting_file)


season = 2025
print(f'Buscando dados para a temporada {season}')

# Criar a lista de pitchers a partir da coluna Name_abbreviation
pitchers = betting_data_df['Name_abbreviation'].tolist()
print(f'Número de pitchers: {len(pitchers)}')

# Lista para armazenar os dados do último jogo de cada pitcher
last_games = []

# Função para carregar o último jogo de um pitcher
def load_last_pitcher_game(pitcher, season):
    for id_suffix in ['01', '02', '03', '04']:
        url = f"https://www.baseball-reference.com/players/gl.fcgi?id={pitcher}{id_suffix}&t=p&year={season}"
        print(f"Tentando URL: {url}")
        try:
            # Carregar tabela de jogos do pitcher
            pitcher_gl = pd.read_html(url, header=0, attrs={'id': 'pitching_gamelogs'})[0]
            
            # Verificar se há dados válidos
            if pitcher_gl.empty:
                print(f"Tabela vazia para {pitcher} com ID {id_suffix}")
                continue
                
            # Filtrar apenas linhas com Rk numérico (dados válidos de jogos)
            pitcher_gl = pitcher_gl[pitcher_gl['Rk'].apply(lambda x: str(x).isdigit())]
            
            if pitcher_gl.empty:
                print(f"Nenhum jogo válido para {pitcher} com ID {id_suffix}")
                continue
                
            # Pegar o último jogo (maior valor de Rk)
            last_game = pitcher_gl.loc[pitcher_gl['Rk'].astype(int).idxmax()]
            
            # Adicionar informações do pitcher e temporada
            last_game['Season'] = season
            last_game['Pitcher'] = pitcher.lower()
            
            print(f"Último jogo encontrado para {pitcher} na temporada {season}")
            return last_game.to_dict()
            
        except (ValueError, IndexError) as e:
            print(f"Erro ao buscar dados para {pitcher} com ID {id_suffix}: {e}")
            time.sleep(random.randint(3, 5))  # Reduzir o tempo de espera
    
    print(f"Nenhum dado encontrado para {pitcher} na temporada {season}")
    return None

# Buscar o último jogo de cada pitcher
for pitcher in pitchers:
    last_game = load_last_pitcher_game(pitcher, season)
    if last_game:
        last_games.append(last_game)
    # Espera aleatória para não sobrecarregar o servidor
    time.sleep(random.randint(3, 5))  # Reduzir o tempo de espera

# Criar DataFrame com os últimos jogos
if last_games:
    last_games_df = pd.DataFrame(last_games)
    
    # Selecionar colunas relevantes, se presentes no DataFrame
    columns_to_keep = [
        'Season',   # Temporada
        'Pitcher',  # Nome do pitcher
        'Date',     # Data do jogo
        'Opp',      # Oponente
        'Home',     # Se o jogo é em casa (1) ou fora (0)
        'IP',       # Innings pitched
        'H',        # Hits permitidos
        'BB',       # Walks permitidos
        'SO',       # Strikeouts
        'ERA',      # Earned Run Average
        'FIP',      # Fielding Independent Pitching
        'GB',       # Ground Balls
        'FB',       # Fly Balls
        'LD',       # Line Drives
        'PU',       # Pop Ups
        'WPA'       # Win Probability Added
    ]
    
    # Manter apenas as colunas que existem no DataFrame
    available_columns = [col for col in columns_to_keep if col in last_games_df.columns]
    last_games_df = last_games_df[available_columns]
    
    # Processar a coluna Home se existir e tiver valor 'Unnamed: 5'
    if 'Unnamed: 5' in last_games_df.columns:
        last_games_df.rename(columns={'Unnamed: 5': 'Home'}, inplace=True)
        last_games_df['Home'] = last_games_df['Home'].fillna(0)
        last_games_df['Home'] = last_games_df['Home'].apply(lambda x: 1 if x == '@' else 0)
        last_games_df['Home'] = last_games_df['Home'].astype(int)
    
    # Imprimir o resultado
    print("\nÚltimos jogos de cada pitcher em 2024:")
    print(last_games_df)
    
    # Também imprimir informações estatísticas
    print("\nEstatísticas gerais:")
    print(last_games_df.describe())
else:
    print("Nenhum dado de último jogo foi encontrado para qualquer pitcher.")

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
