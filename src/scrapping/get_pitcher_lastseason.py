
import pandas as pd
import os
import time
import random
import glob
from datetime import datetime

# Ler o arquivo betting_data.csv
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
seasons = [str(season) for season in range(2023, 2025)]
print(f'number of seasons={len(seasons)}')

# Criar a lista de pitchers a partir da coluna Name_abbreviation
pitchers = betting_data_df['Name_abbreviation'].tolist()
print(f'number of pitchers={len(pitchers)}')

# Criar uma lista vazia para armazenar os DataFrames
dataframes = []


# Verifica se o arquivo CSV já existe
if os.path.exists('pitchers_data.csv'):
    existing_df = pd.read_csv('pitchers_data.csv')  # Lê o DataFrame existente
else:
    existing_df = pd.DataFrame(columns=['Season', 'Pitcher'])  # Cria um DataFrame vazio com as colunas necessárias

# Função para tentar carregar dados de um pitcher com diferentes IDs
def load_pitcher_data(pitcher, season):
    for id_suffix in ['01', '02', '03', '04']:
        url = f"https://www.baseball-reference.com/players/gl.fcgi?id={pitcher}{id_suffix}&t=p&year={season}"
        print(url)
        try:
            # Correção da chamada do pd.read_html
            pitcher_gl = pd.read_html(url, header=0, attrs={'id': 'pitching_gamelogs'})[0]

            # Adicionando a coluna 'Season'
            pitcher_gl.insert(loc=0, column='Season', value=season)

            # Adicionando a coluna 'Pitcher'
            pitcher_gl.insert(loc=2, column='Pitcher', value=pitcher.lower())

            return pitcher_gl  # Retorna o DataFrame se encontrado
        except (ValueError, IndexError):
            print(f"Dados não encontrados para o pitcher {pitcher.lower()} com ID {id_suffix} na temporada {season}. Tentando próximo...")
            time.sleep(random.randint(7, 8))

    return None  # Retorna None se nenhum ID funcionou

for season in seasons:
    for pitcher in pitchers:
        pitcher_data = load_pitcher_data(pitcher, season)
        
        if pitcher_data is not None:
            # Verifica se o pitcher já está no DataFrame existente
            if pitcher_data['Pitcher'].iloc[0] not in existing_df['Pitcher'].values:
                # Armazenando o DataFrame em uma lista
                dataframes.append(pitcher_data)
            else:
                print(f"O pitcher {pitcher.lower()} já existe no arquivo CSV. Ignorando...")
        else:
            print(f"Nenhum dado encontrado para o pitcher {pitcher.lower()} na temporada {season}.")

        # Espera aleatória para não sobrecarregar o servidor
        time.sleep(random.randint(7, 8))

# Concatenando todos os DataFrames da lista em um único DataFrame, se houver dados
if dataframes:
    new_pitchers_df = pd.concat(dataframes, ignore_index=True)

    # Realizando a limpeza de dados
    if 'Rk' in new_pitchers_df.columns:
        new_pitchers_df = new_pitchers_df[new_pitchers_df['Rk'].apply(lambda x: str(x).isdigit())]
        new_pitchers_df.reset_index(drop=True, inplace=True)
        print("Linhas não inteiras foram removidas com sucesso.")
    else:
        print("A coluna 'Rk' não existe no DataFrame.")

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
        'ERA',      # Earned Run Average
        'FIP',      # Fielding Independent Pitching
        'GB',       # Ground Balls
        'FB',       # Fly Balls
        'LD',       # Line Drives
        'PU',       # Pop Ups
        'WPA'       # Win Probability Added
    ]

    # Dropar as colunas que não estão na lista
    new_pitchers_df = new_pitchers_df[columns_to_keep]

    # Exibir informações do DataFrame após o drop
    print(new_pitchers_df.info(verbose=True))

    # Concatenar com os dados existentes
    combined_df = pd.concat([existing_df, new_pitchers_df], ignore_index=True)

    # Salvando o DataFrame em um arquivo CSV
    combined_df.to_csv('pitchers_data.csv', index=False)

    print("Arquivo CSV salvo com sucesso!")
else:
    print("Nenhum dado novo foi coletado para salvar.")


