# %%
import csv
import requests
from datetime import datetime
from typing import Dict, List, Optional, Union

class BettingDataScraper:
    BASE_URL = "https://api.bettingpros.com/v3"
    HEADERS = {
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

    TEAM_ABBREVIATIONS = {
        "KC": "KCR",   # Kansas City Royals
        "SD": "SDP",   # San Diego Padres
        "TB": "TBR",   # Tampa Bay Rays
        "WSH": "WSN",  # Washington Nationals
        "CWS": "CHW",  # Chicago White Sox
        "SF": "SFG",   # San Francisco Giants
    }

    def __init__(self):
        self.matchups = {}

    @staticmethod
    def convert_odds_to_decimal(american_odds: Optional[int]) -> Optional[float]:
        if american_odds is None:
            return None
        if american_odds > 0:
            return round((american_odds / 100) + 1, 2)
        return round((100 / abs(american_odds)) + 1, 2)

    @staticmethod
    def parse_date(date_str: str) -> str:
        """Convert various date formats to YYYY-MM-DD"""
        try:
            date_obj = datetime.strptime(date_str, "%B-%d-%Y")
            return date_obj.strftime("%Y-%m-%d")
        except ValueError:
            try:
                date_obj = datetime.strptime(date_str, "%b-%d-%Y")
                return date_obj.strftime("%Y-%m-%d")
            except ValueError:
                raise ValueError("Date must be in format 'month-dd-yyyy' (e.g., 'june-02-2024' or 'jun-02-2024')")

    @staticmethod
    def format_date_for_filename(date_str: str) -> str:
        """Format date for filename (YYYY-MM-DD)"""
        try:
            # Try different formats
            for fmt in ["%B-%d-%Y", "%b-%d-%Y", "%Y-%m-%d"]:
                try:
                    date_obj = datetime.strptime(date_str, fmt)
                    return date_obj.strftime("%Y-%m-%d")
                except ValueError:
                    continue
            raise ValueError("Invalid date format")
        except Exception:
            return datetime.now().strftime("%Y-%m-%d")

    def get_events(self, date: str, sport: str = "MLB") -> List[str]:
        """Fetch all event IDs for a given date and sport"""
        url = f"{self.BASE_URL}/events"
        params = {
            "date": self.parse_date(date),
            "sport": sport,
            "limit": 100,
            "page": 1
        }

        response = requests.get(url, headers=self.HEADERS, params=params)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch events: {response.status_code}")

        data = response.json()
        return [str(event['id']) for event in data.get('events', [])]

    def fetch_matchups(self, event_ids: List[str]) -> None:
        """Fetch and store matchup information"""
        url = f"{self.BASE_URL}/offers"
        params = {
            "limit": 25,
            "page": 1,
            "picks": "true",
            "sport": "mlb",
            "market_id": "285",
            "location": "INT",
            "event_id": ":".join(event_ids),
            "live": "true"
        }

        response = requests.get(url, headers=self.HEADERS, params=params)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch offers: {response.status_code}")

        data = response.json()
        self._process_matchups(data.get('offers', []))

    def _process_matchups(self, offers: List[Dict]) -> None:
        """Process and store matchup information"""
        for offer in offers:
            participants = offer.get('participants', [])
            if len(participants) != 2:
                continue

            team1 = participants[0]['team'].get('abbreviation')
            team2 = participants[1]['team'].get('abbreviation')

            team1_corrected = self.TEAM_ABBREVIATIONS.get(team1, team1)
            team2_corrected = self.TEAM_ABBREVIATIONS.get(team2, team2)

            self.matchups[team1_corrected] = {
                'opponent': team2_corrected,
                'home': team2_corrected,
                'away': team1_corrected
            }
            self.matchups[team2_corrected] = {
                'opponent': team1_corrected,
                'home': team2_corrected,
                'away': team1_corrected
            }

    def fetch_props(self, date: str) -> List[Dict]:
        """Fetch props data"""
        url = f"{self.BASE_URL}/props"
        params = {
            "limit": 25,
            "page": 1,
            "sport": "MLB",
            "market_id": 285,
            "date": self.parse_date(date),
            "location": "INT",
            "sort": "diff",
            "include_selections": "false",
            "include_markets": "true",
            "min_odds": -200,
            "max_odds": 400
        }

        response = requests.get(url, headers=self.HEADERS, params=params)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch props: {response.status_code}")

        return response.json().get('props', [])

    def create_player_abbreviation(self, player_name: str) -> str:
        """Create abbreviated name for a player"""
        name_parts = player_name.lower().split()
        if len(name_parts) > 1:
            return f"{name_parts[1][:5]}{name_parts[0][:2]}"
        return f"{player_name[:5]}{player_name[:2]}"

    def save_to_csv(self, props: List[Dict], output_file: str = 'betting_data.csv') -> None:
        """Save props data to CSV file"""
        with open(output_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([
                'Player', 'Name_abbreviation', 'Team', 'Opponent', 'Home Team', 'Away Team',
                'Over Line', 'Over Odds', 'Under Line', 'Under Odds', 'API Projected Value',
                'API Recommended Side', 'Streak', 'Streak Type', 'Probability', 'Bet Rating', 'Diff'
            ])

            for prop in props:
                self._write_prop_to_csv(prop, csv_writer)

    def _write_prop_to_csv(self, prop: Dict, csv_writer) -> None:
        """Write a single prop to CSV"""
        player_name = prop['participant']['name']
        team = prop['participant']['player']['team']
        team_corrected = self.TEAM_ABBREVIATIONS.get(team, team)
        matchup = self.matchups.get(team_corrected, {})

        csv_writer.writerow([
            player_name,
            self.create_player_abbreviation(player_name),
            team_corrected,
            matchup.get('opponent', 'N/A'),
            matchup.get('home', 'N/A'),
            matchup.get('away', 'N/A'),
            prop.get('over', {}).get('line'),
            self.convert_odds_to_decimal(prop.get('over', {}).get('odds')),
            prop.get('under', {}).get('line'),
            self.convert_odds_to_decimal(prop.get('under', {}).get('odds')),
            prop.get('projection', {}).get('value'),
            prop.get('projection', {}).get('recommended_side'),
            prop.get('performance', {}).get('streak'),
            prop.get('performance', {}).get('streak_type'),
            prop.get('projection', {}).get('probability'),
            prop.get('projection', {}).get('bet_rating'),
            prop.get('projection', {}).get('diff')
        ])

def get_date_string(date=None):
    """
    Formata uma data para o formato month-dd-yyyy.
    Se nenhuma data for fornecida, usa a data atual.
    """
    if date:
        return date
    
    # Usa a data atual
    today = datetime.now()
    month_names = ['january', 'february', 'march', 'april', 'may', 'june', 
                  'july', 'august', 'september', 'october', 'november', 'december']
    month_name = month_names[today.month - 1]
    return f"{month_name}-{today.day:02d}-{today.year}"

def main(date=None, output_file=None):
    try:
        # Se a data não for especificada, usa a data atual
        date_string = get_date_string(date)
        
        # Gera o nome do arquivo baseado na data
        if output_file is None:
            formatted_date = BettingDataScraper.format_date_for_filename(date_string)
            output_file = f'betting_data_{formatted_date}.csv'
        
        scraper = BettingDataScraper()
        
        print(f"Buscando dados para a data: {date_string}")
        
        # Get all events for the date
        event_ids = scraper.get_events(date_string)
        if not event_ids:
            print(f"Nenhum evento encontrado para a data: {date_string}")
            return

        # Fetch matchups using event IDs
        scraper.fetch_matchups(event_ids)

        # Fetch and save props data
        props = scraper.fetch_props(date_string)
        scraper.save_to_csv(props, output_file)
        print(f"Dados salvos com sucesso em {output_file}")

    except Exception as e:
        print(f"Erro: {str(e)}")


if __name__ == "__main__":
    # Por padrão, usa a data atual
    main()
    
    # Exemplo de como usar com uma data específica:
    # main("march-30-2025")
    
    # Exemplo com data específica e nome de arquivo personalizado:
    # main("march-30-2025", "dados_30_marco.csv")


