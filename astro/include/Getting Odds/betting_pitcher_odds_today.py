import csv
import requests
from datetime import datetime
from typing import Dict, List, Optional, Union
from dotenv import load_dotenv
import os
import logging
import pytz

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BettingDataScraper:
    BASE_URL = "https://api.bettingpros.com/v3"
    MLB_API_URL = "https://statsapi.mlb.com/api/v1/schedule"
    HEADERS = {
        "Origin": "https://www.bettingpros.com",
        "x-api-key": os.getenv("BETTINGPROS_API_KEY"),
        "sec-ch-ua-platform": '"macOS"',
        "Referer": "https://www.bettingpros.com/",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "sec-ch-ua": '"Google Chrome";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
        "sec-ch-ua-mobile": "?0"
    }

    TEAM_ABBREVIATIONS = {
        "KC": "KCR",   # Kansas City Royals
        "SD": "SDP",   # San Diego Padres
        "TB": "TBR",   # Tampa Bay Rays
        "WSH": "WSN",  # Washington Nationals
        "CWS": "CHW",  # Chicago White Sox
        "SF": "SFG",   # San Francisco Giants
        "NYM": "NYM",  # New York Mets
        "NY": "NYY",   # New York Yankees
        "BOS": "BOS",  # Boston Red Sox
        "TOR": "TOR",  # Toronto Blue Jays
        "BAL": "BAL",  # Baltimore Orioles
        "CLE": "CLE",  # Cleveland Guardians
        "DET": "DET",  # Detroit Tigers
        "MIN": "MIN",  # Minnesota Twins
        "CHW": "CHW",  # Chicago White Sox
        "HOU": "HOU",  # Houston Astros
        "LAA": "LAA",  # Los Angeles Angels
        "OAK": "OAK",  # Oakland Athletics
        "ATH": "OAK",  # Oakland Athletics
        "SEA": "SEA",  # Seattle Mariners
        "TEX": "TEX",  # Texas Rangers
        "ATL": "ATL",  # Atlanta Braves
        "MIA": "MIA",  # Miami Marlins
        "PHI": "PHI",  # Philadelphia Phillies
        "WSN": "WSN",  # Washington Nationals
        "CHC": "CHC",  # Chicago Cubs
        "CIN": "CIN",  # Cincinnati Reds
        "MIL": "MIL",  # Milwaukee Brewers
        "PIT": "PIT",  # Pittsburgh Pirates
        "STL": "STL",  # St. Louis Cardinals
        "ARI": "ARI",  # Arizona Diamondbacks
        "COL": "COL",  # Colorado Rockies
        "LAD": "LAD",  # Los Angeles Dodgers
        "SDP": "SDP",  # San Diego Padres
        "SFG": "SFG",  # San Francisco Giants
    }

    # Full team name to abbreviation mapping for MLB API
    FULL_TEAM_NAMES = {
        "Kansas City Royals": "KCR",
        "San Diego Padres": "SDP", 
        "Tampa Bay Rays": "TBR",
        "Washington Nationals": "WSN",
        "Chicago White Sox": "CHW",
        "San Francisco Giants": "SFG",
        "New York Mets": "NYM",
        "New York Yankees": "NYY",
        "Boston Red Sox": "BOS",
        "Toronto Blue Jays": "TOR",
        "Baltimore Orioles": "BAL",
        "Cleveland Guardians": "CLE",
        "Detroit Tigers": "DET",
        "Minnesota Twins": "MIN",
        "Houston Astros": "HOU",
        "Los Angeles Angels": "LAA",
        "Oakland Athletics": "OAK",
        "Athletics": "OAK",
        "Seattle Mariners": "SEA",
        "Texas Rangers": "TEX",
        "Atlanta Braves": "ATL",
        "Miami Marlins": "MIA",
        "Philadelphia Phillies": "PHI",
        "Chicago Cubs": "CHC",
        "Cincinnati Reds": "CIN",
        "Milwaukee Brewers": "MIL",
        "Pittsburgh Pirates": "PIT",
        "St. Louis Cardinals": "STL",
        "Arizona Diamondbacks": "ARI",
        "Colorado Rockies": "COL",
        "Los Angeles Dodgers": "LAD",
    }

    def __init__(self):
        self.matchups = {}
        self.events = {}
        self.mlb_schedule = {}

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

    def fetch_mlb_schedule(self, date: str) -> Dict:
        """Fetch MLB schedule data for a given date"""
        formatted_date = self.parse_date(date)
        
        # Build the URL with all parameters as they appear in the original URL
        url = f"{self.MLB_API_URL}?sportId=1&sportId=21&sportId=51&startDate={formatted_date}&endDate={formatted_date}&timeZone=America/New_York&gameType=E&&gameType=S&&gameType=R&&gameType=F&&gameType=D&&gameType=L&&gameType=W&&gameType=A&&gameType=C&language=en&leagueId=&&leagueId=&&leagueId=103&&leagueId=104&&leagueId=590&&leagueId=160&&leagueId=159&&leagueId=420&&leagueId=428&&leagueId=431&&leagueId=426&&leagueId=427&&leagueId=429&&leagueId=430&&leagueId=432&hydrate=team,linescore(matchup,runners),xrefId,story,flags,statusFlags,broadcasts(all),venue(location),decisions,person,probablePitcher,stats,game(content(media(epg),summary),tickets),seriesStatus(useOverride=true)&sortBy=gameDate,gameStatus,gameType"

        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch MLB schedule: {response.status_code}")

        data = response.json()
        games = data.get('dates', [{}])[0].get('games', [])
        
        # Process games and store matchup information
        for game in games:
            teams = game.get('teams', {})
            away_team = teams.get('away', {})
            home_team = teams.get('home', {})
            
            away_team_name = away_team.get('team', {}).get('name', '')
            home_team_name = home_team.get('team', {}).get('name', '')
            
            # Convert full names to abbreviations
            away_abbr = self.FULL_TEAM_NAMES.get(away_team_name, away_team_name)
            home_abbr = self.FULL_TEAM_NAMES.get(home_team_name, home_team_name)
            
            # Store matchup info with home/away designation
            self.mlb_schedule[away_abbr] = {
                'opponent': home_abbr,
                'home': 0,  # Away team
                'away_team': away_abbr,
                'home_team': home_abbr
            }
            
            self.mlb_schedule[home_abbr] = {
                'opponent': away_abbr,
                'home': 1,  # Home team
                'away_team': away_abbr,
                'home_team': home_abbr
            }
        
        return self.mlb_schedule

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
        events = data.get('events', [])
        
        # Store event information
        for event in events:
            event_id = str(event['id'])
            participants = event.get('participants', [])
            if len(participants) == 2:
                team1 = participants[0]['team'].get('abbreviation')
                team2 = participants[1]['team'].get('abbreviation')
                self.events[event_id] = {
                    'team1': self.TEAM_ABBREVIATIONS.get(team1, team1),
                    'team2': self.TEAM_ABBREVIATIONS.get(team2, team2)
                }
        
        return [str(event['id']) for event in events]

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

    def save_to_csv(self, props: List[Dict], output_file: str = 'betting_data.csv') -> None:
        """Save props data to CSV file"""
        with open(output_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([
                'Player', 'Team', 'Opponent', 'Home',
                'Over Line', 'Over Odds', 'Under Line', 'Under Odds', 'API Projected Value',
                'API Recommended Side', 'Streak', 'Streak Type', 'Diff'
            ])

            for prop in props:
                self._write_prop_to_csv(prop, csv_writer)

    def _write_prop_to_csv(self, prop: Dict, csv_writer) -> None:
        """Write a single prop to CSV"""
        player_name = prop['participant']['name']
        team = prop['participant']['player']['team']
        team_corrected = self.TEAM_ABBREVIATIONS.get(team, team)
        
        # Get matchup info from MLB schedule
        matchup_info = self.mlb_schedule.get(team_corrected, {})
        opponent = matchup_info.get('opponent', 'N/A')
        home_indicator = matchup_info.get('home', 0)

        csv_writer.writerow([
            player_name,
            team_corrected,
            opponent,
            home_indicator,
            prop.get('over', {}).get('line'),
            self.convert_odds_to_decimal(prop.get('over', {}).get('odds')),
            prop.get('under', {}).get('line'),
            self.convert_odds_to_decimal(prop.get('under', {}).get('odds')),
            prop.get('projection', {}).get('value'),
            prop.get('projection', {}).get('recommended_side'),
            prop.get('performance', {}).get('streak'),
            prop.get('performance', {}).get('streak_type'),
            prop.get('projection', {}).get('diff')
        ])

def get_date_string(date=None):
    """
    format date to month-dd-yyyy.
    if specif date is not define, it will use today's date.
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
            # Always write to include/Getting Odds directory
            import os
            from pathlib import Path
            include_dir = Path(__file__).parent
            output_file = os.path.join(include_dir, f'betting_data_{formatted_date}.csv')
        
        scraper = BettingDataScraper()
        
        logger.info(f"Searching data from date: {date_string}")
        
        # Debug prints
        import os
        print("[DEBUG] Current working directory:", os.getcwd())
        print("[DEBUG] Output file path:", output_file)
        print("[DEBUG] BETTINGPROS_API_KEY:", os.getenv("BETTINGPROS_API_KEY"))
        print(f"[DEBUG] Date string being used: {date_string}")
        print(f"[DEBUG] Parsed date for API: {BettingDataScraper.parse_date(date_string)}")
        
        # Get current NY time for comparison
        ny_tz = pytz.timezone('America/New_York')
        ny_now = datetime.now(ny_tz)
        print(f"[DEBUG] Current NY time: {ny_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        # Fetch MLB schedule data first
        mlb_schedule = scraper.fetch_mlb_schedule(date_string)
        logger.info(f"Fetched MLB schedule with {len(mlb_schedule)} team matchups")
        
        # Get all events for the date
        event_ids = scraper.get_events(date_string)
        if not event_ids:
            logger.warning(f"No games found for: {date_string}")
            return

        logger.info(f"Found {len(event_ids)} games for {date_string}")

        # Fetch and save props data
        props = scraper.fetch_props(date_string)
        if not props:
            logger.warning("No props data found")
            return
            
        scraper.save_to_csv(props, output_file)
        logger.info(f"Data saved to {output_file}")

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)


if __name__ == "__main__":
    # Por padrão, usa a data atual
    main()
    
    # Exemplo de como usar com uma data específica:
    # main("march-30-2025")
    
    # Exemplo com data específica e nome de arquivo personalizado:
    # main("march-30-2025", "dados_30_marco.csv")


