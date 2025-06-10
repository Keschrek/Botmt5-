# tests/conftest.py

import pytest
from pytest_mock import MockerFixture
from unittest.mock import MagicMock

@pytest.fixture
def mock_mt5(mocker: MockerFixture):
    # Mock MetaTrader5 functions here
    mocker.patch('MetaTrader5.initialize', return_value=True)
    mocker.patch('MetaTrader5.copy_rates_from_pos', return_value=[]) # Placeholder
    mocker.patch('MetaTrader5.symbol_info_tick', return_value=type('obj', (object,), {'ask': 1.0, 'bid': 1.0})())
    # Add more MT5 mocks as needed
    pass

@pytest.fixture
def mock_telegram_bot(mocker: MockerFixture):
    # Mock python-telegram-bot functions here
    mock_bot = mocker.Mock()
    mocker.patch('telegram.Bot', return_value=mock_bot)
    # Add more Telegram mocks as needed
    return mock_bot

@pytest.fixture
def mock_mt5_client():
    """Provides a mock MT5Client instance."""
    mock_client = MagicMock()
    # Configure mock methods and attributes as needed for your tests
    # Example:
    # mock_client.initialize.return_value = True
    # mock_client.copy_rates_from_pos.return_value = [] # Mock return value for data
    # mock_client.symbol_info_tick.return_value = None # Mock return value for tick
    # mock_client.order_send.return_value = MagicMock(retcode=0) # Mock successful order
    
    # Simulate connection status
    mock_client.is_connected = True
    mock_client.symbols_mapping = {'XAUUSD': 'XAUUSD', 'XAUUSDm': None} # Example mapping
    
    return mock_client

@pytest.fixture
def mock_telegram_handler():
    """Provides a mock TelegramHandler instance."""
    mock_handler = MagicMock()
    # Configure mock methods and attributes
    # Example:
    # mock_handler.send_message = MagicMock() # Mock send_message method
    
    # Simulate last activity timestamp for watchdog test
    import time
    mock_handler.last_activity_timestamp = time.time()

    return mock_handler

# Add other fixtures (e.g., for config loading) here 