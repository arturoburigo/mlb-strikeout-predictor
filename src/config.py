import os
from dataclasses import dataclass, field
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

def _parse_emails(email_string: str) -> List[str]:
    """Parse comma-separated email addresses"""
    if not email_string:
        return []
    return [email.strip() for email in email_string.split(',') if email.strip()]

def _parse_chat_ids(chat_id_string: str) -> List[str]:
    """Parse comma-separated chat IDs"""
    if not chat_id_string:
        return []
    return [chat_id.strip() for chat_id in chat_id_string.split(',') if chat_id.strip()]

@dataclass
class APIConfig:
    """Configuration for external APIs"""
    bettingpros_api_key: str = field(default_factory=lambda: os.getenv('BETTINGPROS_API_KEY', ''))
    base_url: str = "https://api.bettingpros.com/v3"
    
    def __post_init__(self):
        if not self.bettingpros_api_key:
            raise ValueError("BETTINGPROS_API_KEY environment variable is required")

@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering pipeline"""
    # Weighting strategy
    recent_weight: float = 0.6
    season_weight: float = 0.4
    
    # Data quality thresholds
    min_innings_pitched: float = 1.0
    min_strikeouts: int = 0  # Allow 0 for more data
    
    # Rolling window sizes
    recent_games_window: int = 5
    momentum_window: int = 3
    
    # Rate calculation safety
    min_ip_for_rates: float = 0.1
    
    # Feature selection
    max_features: int = 10
    feature_importance_threshold: float = 0.01

@dataclass 
class ModelConfig:
    """Configuration for ML models"""
    # Models to try
    models_to_use: List[str] = field(default_factory=lambda: ['RandomForest', 'XGBoost'])
    
    # Cross-validation
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    
    # Hyperparameter tuning
    enable_hyperparameter_tuning: bool = True
    hp_search_iterations: int = 50
    
    # Model performance thresholds
    min_r2_score: float = 0.1
    
    # Feature selection
    feature_selection_method: str = "mutual_info"  # or "feature_importance"
    
@dataclass
class ScrapingConfig:
    """Configuration for web scraping"""
    # Selenium settings
    headless: bool = True
    wait_time: int = 7  # Base wait time between requests
    max_retries: int = 3
    
    # Request settings
    request_timeout: int = 30
    user_agent: str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    
    # URL settings
    betano_url: str = "https://www.betano.bet.br/sport/beisebol/eua/mlb/1662/?bt=strikeouts"
    baseball_reference_base: str = "https://www.baseball-reference.com"

@dataclass
class EmailConfig:
    """Configuration for email notifications"""
    sender_email: str = field(default_factory=lambda: os.getenv('EMAIL_SENDER', ''))
    sender_password: str = field(default_factory=lambda: os.getenv('EMAIL_PASSWORD', ''))
    sender_name: str = "MLB PREDICT"
    receiver_email: str = field(default_factory=lambda: os.getenv('RECEIVER_EMAIL', ''))
    cc_emails: List[str] = field(default_factory=lambda: _parse_emails(os.getenv('CC_EMAILS', '')))
    
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 465
    
    def __post_init__(self):
        if not self.sender_email or not self.sender_password:
            raise ValueError("EMAIL_SENDER and EMAIL_PASSWORD environment variables are required")

@dataclass
class TelegramConfig:
    """Configuration for Telegram notifications"""
    bot_token: str = field(default_factory=lambda: os.getenv('TELEGRAM_BOT_TOKEN', ''))
    chat_ids: List[str] = field(default_factory=lambda: _parse_chat_ids(os.getenv('TELEGRAM_CHAT_ID', '')))
    
    def __post_init__(self):
        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")

@dataclass
class SchedulingConfig:
    """Configuration for job scheduling"""
    timezone: str = "America/New_York"
    
    # Pipeline times (24-hour format)
    data_pipeline_hour: int = field(default_factory=lambda: int(os.getenv('DATA_PIPELINE_HOUR', 19)))
    data_pipeline_minute: int = field(default_factory=lambda: int(os.getenv('DATA_PIPELINE_MINUTE', 10)))
    
    email_hour: int = field(default_factory=lambda: int(os.getenv('EMAIL_HOUR', 22)))
    email_minute: int = field(default_factory=lambda: int(os.getenv('EMAIL_MINUTE', 40)))
    
    results_email_hour: int = field(default_factory=lambda: int(os.getenv('RESULTS_EMAIL_HOUR', 11)))
    results_email_minute: int = field(default_factory=lambda: int(os.getenv('RESULTS_EMAIL_MINUTE', 42)))
    
    aws_upload_hour: int = field(default_factory=lambda: int(os.getenv('AWS_UPLOAD_HOUR', 12)))
    aws_upload_minute: int = field(default_factory=lambda: int(os.getenv('AWS_UPLOAD_MINUTE', 0)))
    
    cleanup_hour: int = field(default_factory=lambda: int(os.getenv('CLEANUP_HOUR', 13)))
    cleanup_minute: int = field(default_factory=lambda: int(os.getenv('CLEANUP_MINUTE', 0)))
    
    telegram_hour: int = field(default_factory=lambda: int(os.getenv('TELEGRAM_HOUR', 22)))
    telegram_minute: int = field(default_factory=lambda: int(os.getenv('TELEGRAM_MINUTE', 45)))

@dataclass
class PathsConfig:
    """Configuration for file paths"""
    data_dir: str = "."
    models_dir: str = "models"
    logs_dir: str = "logs"
    cache_dir: str = "cache"
    
    # Generated file patterns
    betting_data_pattern: str = "betting_data_{date}.csv"
    predictions_pattern: str = "predicted_{date}.csv"
    results_pattern: str = "game_results_{date}.csv"
    engineered_features_file: str = "pitchers_data_engineered.csv"

@dataclass
class Config:
    """Main configuration class that combines all sub-configurations"""
    api: APIConfig = field(default_factory=APIConfig)
    features: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig) 
    model: ModelConfig = field(default_factory=ModelConfig)
    scraping: ScrapingConfig = field(default_factory=ScrapingConfig)
    email: EmailConfig = field(default_factory=EmailConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    scheduling: SchedulingConfig = field(default_factory=SchedulingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    
    # Global settings
    debug: bool = field(default_factory=lambda: os.getenv('DEBUG', 'False').lower() == 'true')
    log_level: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO'))
    
    @classmethod
    def load(cls) -> 'Config':
        """Load configuration with validation"""
        try:
            return cls()
        except ValueError as e:
            raise ValueError(f"Configuration error: {e}")
    
    def validate(self) -> None:
        """Validate configuration values"""
        # Validate feature engineering
        if not 0 < self.features.recent_weight < 1:
            raise ValueError("recent_weight must be between 0 and 1")
        
        if abs(self.features.recent_weight + self.features.season_weight - 1.0) > 0.001:
            raise ValueError("recent_weight + season_weight must equal 1.0")
        
        # Validate model config
        if self.model.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")
            
        if not 0 < self.model.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")

# Team abbreviation mappings
TEAM_ABBREVIATIONS = {
    "KC": "KCR",   # Kansas City Royals
    "SD": "SDP",   # San Diego Padres  
    "TB": "TBR",   # Tampa Bay Rays
    "WSH": "WSN",  # Washington Nationals
    "CWS": "CHW",  # Chicago White Sox
    "SF": "SFG",   # San Francisco Giants
    "NY": "NYY",   # New York Yankees
}

# Extend with all MLB teams
TEAM_ABBREVIATIONS.update({
    "NYM": "NYM", "BOS": "BOS", "TOR": "TOR", "BAL": "BAL", "CLE": "CLE",
    "DET": "DET", "MIN": "MIN", "CHW": "CHW", "HOU": "HOU", "LAA": "LAA", 
    "OAK": "OAK", "SEA": "SEA", "TEX": "TEX", "ATL": "ATL", "MIA": "MIA",
    "PHI": "PHI", "WSN": "WSN", "CHC": "CHC", "CIN": "CIN", "MIL": "MIL",
    "PIT": "PIT", "STL": "STL", "ARI": "ARI", "COL": "COL", "LAD": "LAD",
    "SDP": "SDP", "SFG": "SFG", "KCR": "KCR", "TBR": "TBR"
})

# Create a default global config instance
config = Config.load() 