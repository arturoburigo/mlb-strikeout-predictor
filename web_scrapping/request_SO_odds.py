import csv
import requests

def convert_odds_to_decimal(american_odds):
    if american_odds is None:  # Check for None value
        return None
    if american_odds > 0:
        return round((american_odds / 100) + 1, 2)  # Round to 2 decimal places
    else:
        return round((100 / abs(american_odds)) + 1, 2) 

# Definindo a URL e os parâmetros
url_props = "https://api.bettingpros.com/v3/props"
params_props = {
    "limit": 25,
    "page": 1,
    "sport": "MLB",
    "market_id": 285,
    "date": "2024-10-01",
    "location": "INT",
    "sort": "diff",
    "include_selections": "false",
    "include_markets": "true",
    "min_odds": -200,
    "max_odds": 400
}

url_offers = "https://api.bettingpros.com/v3/offers"
params_offers = {   
    "limit": 25,
    "page": 1,
    "picks": "true",
    "sport": "mlb",
    "market_id": "176",
    "location": "INT",
    "event_id": "94984:94983:94985:95011",
    "live": "true"
}

headers = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
    "Cache-Control": "no-store, no-cache, must-revalidate",
    "Expires": "0",
    "Origin": "https://www.bettingpros.com",
    "Pragma": "no-cache",
    "Priority": "u=3, i",
    "Referer": "https://www.bettingpros.com/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15",
    "x-api-key": "CHi8Hy5CEE4khd46XNYL23dCFX96oUdw6qOt1Dnh",
    "x-level": "YmFzaWM="
}

# Dicionário de mapeamento de abreviações
abbreviation_correction = {
    "KC": "KCR",  # Kansas City Royals
    "SD": "SDP",  # San Diego Padres
    "TB": "TBR",  # Tampa Bay Rays
    "WSH": "WSN",  # Washington Nationals
    "CWS": "CHW",  # Chicago White Sox
    "SF": "SFG",   # San Francisco Giants
}

# Requisição para obter os dados das props
response_props = requests.get(url_props, headers=headers, params=params_props)

# Inicializando um dicionário para armazenar os matchups
matchups = {}

# Requisição para obter os dados das ofertas (matchups)
response_offers = requests.get(url_offers, headers=headers, params=params_offers)

if response_offers.status_code == 200:
    data_offers = response_offers.json()

    if 'offers' in data_offers:  # Verifica se 'offers' está presente
        # Percorrer as ofertas e armazenar os matchups
        for offer in data_offers.get('offers', []):
            participants = offer.get('participants', [])
            if len(participants) == 2:
                team1 = participants[0]['team'].get('abbreviation')  # Obter a sigla do time 1
                team2 = participants[1]['team'].get('abbreviation')  # Obter a sigla do time 2

                # Corrigir abreviações, se necessário
                team1_corrected = abbreviation_correction.get(team1, team1)
                team2_corrected = abbreviation_correction.get(team2, team2)

                # Armazenar o matchup e determinar home e away teams
                matchups[team1_corrected] = {
                    'opponent': team2_corrected,
                    'home': team2_corrected,
                    'away': team1_corrected
                }
                matchups[team2_corrected] = {
                    'opponent': team1_corrected,
                    'home': team2_corrected,
                    'away': team1_corrected
                }
            else:
                print("Dados de participantes incompletos para esse evento.")
else:
    print(f"Erro: {response_offers.status_code}")

if response_props.status_code != 200:
    print(f"Error: Received status code {response_props.status_code}")
else:
    try:
        # Attempt to decode the response as JSON
        data_props = response_props.json()
    except requests.exceptions.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print("Response content:", response_props.text)  #


# Requisição para obter os dados das props
response_props = requests.get(url_props, headers=headers, params=params_props)

if response_props.status_code == 200:
    content_type = response_props.headers.get('Content-Type', '')
    if 'application/json' in content_type:
        try:
            data_props = response_props.json()
        except requests.exceptions.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print("Response content:", response_props.text)
    else:
        print("Expected JSON response but got:", content_type)
        print("Response content:", response_props.text)
else:
    print(f"Error: Received status code {response_props.status_code}")

if response_props.status_code == 200:
    data_props = response_props.json()

    # Open the CSV file for writing
    with open('betting_data.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write header row
        csv_writer.writerow([
            'Player', 'Name_abbreviation', 'Team', 'Opponent', 'Home Team', 'Away Team', 'Over Line', 
            'Over Odds', 'Under Line', 'Under Odds', 'API Projected Value', 
            'API Recommended Side', 'Streak', 'Streak Type', 'Probability', 'Bet Rating', 'Diff'
        ])

        props = data_props['props']  # Get the list of props

        for prop in props:
            player_name = prop['participant']['name']
            projection = prop['projection']
            team = prop['participant']['player']['team']
            over = prop.get('over', {})  # Access over data (avoid NameError)
            under = prop.get('under', {})  # Access under data (avoid NameError)
            performance = prop.get('performance', {})  # Access performance data (avoid NameError)

            # Extract data with checks for presence
            recommended_side = projection.get('recommended_side')
            value = projection.get('value')
            probability = projection.get('probability')
            bet_rating = projection.get('bet_rating')
            diff = projection.get('diff')
            over_line = over.get('line')
            over_odds = convert_odds_to_decimal(over.get('odds'))  # Convert to decimal
            under_line = under.get('line')
            under_odds = convert_odds_to_decimal(under.get('odds'))  # Convert to decimal
            streak = performance.get('streak')
            streak_type = performance.get('streak_type')

            # Corrigir a abreviação do time, se necessário
            team_corrected = abbreviation_correction.get(team, team)

            # Determinar o time oponente, home e away
            matchup = matchups.get(team_corrected, {})
            opponent = matchup.get('opponent', 'N/A')  # Se não encontrar, retorna 'N/A'
            home_team = matchup.get('home', 'N/A')
            away_team = matchup.get('away', 'N/A')

            # Create abbreviation for the player
            name_parts = player_name.lower().split()
            if len(name_parts) > 1:  # Ensure there's at least a first and last name
                last_name = name_parts[1][:5]  # First 5 letters of the last name
                first_name = name_parts[0][:2]  # First 2 letters of the first name
                abbreviation = last_name + first_name
            else:  # If there's only one name
                abbreviation = player_name[:5] + player_name[:2]

            # Write data to CSV row
            csv_writer.writerow([
                player_name, abbreviation, team_corrected, opponent, home_team, away_team, over_line, 
                over_odds, under_line, under_odds, value, 
                recommended_side, streak, streak_type, probability, bet_rating, diff
            ])
else:
    print(f"Error: {response_props.status_code}")
