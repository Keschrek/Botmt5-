class BotError(Exception):
    """Base exception for all custom bot errors."""
    pass

class ConfigurationError(BotError):
    """Exception raised for errors in configuration processing."""
    pass

class ConnectionError(BotError):
    """Exception raised for connection-related errors (e.g., MT5, Telegram)."""
    pass

class DataError(BotError):
    """Exception raised for data processing or retrieval errors."""
    pass

class TradingError(BotError):
    """Exception raised for errors during trading operations (e.g., placing orders)."""
    pass

class WatchdogError(BotError):
    """Exception raised for errors within the watchdog module."""
    pass

class TelegramError(ConnectionError):
    """Exception raised for specific Telegram API errors."""
    pass

class MT5Error(ConnectionError):
    """Exception raised for specific MT5 API errors."""
    pass

# Add other specific exceptions as needed 