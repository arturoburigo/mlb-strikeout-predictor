"""
Custom exceptions for the pitcher strikeout prediction system.
"""

class PitcherPredictionError(Exception):
    """Base exception for pitcher prediction system"""
    pass

class DataScrapingError(PitcherPredictionError):
    """Raised when data scraping fails"""
    pass

class DataLoadingError(PitcherPredictionError):
    """Raised when data loading fails"""
    pass

class FeatureEngineeringError(PitcherPredictionError):
    """Raised when feature engineering fails"""
    pass

class ModelTrainingError(PitcherPredictionError):
    """Raised when model training fails"""
    pass

class PredictionError(PitcherPredictionError):
    """Raised when prediction generation fails"""
    pass

class NotificationError(PitcherPredictionError):
    """Raised when notification sending fails"""
    pass

class ConfigurationError(PitcherPredictionError):
    """Raised when there's a configuration issue"""
    pass

class ValidationError(PitcherPredictionError):
    """Raised when data validation fails"""
    pass

class APIError(PitcherPredictionError):
    """Raised when API calls fail"""
    pass

class FileNotFoundError(PitcherPredictionError):
    """Raised when required files are not found"""
    pass 