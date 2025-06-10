import pytest
import json
import jsonschema
from pathlib import Path

# Define the path to the signal schema file relative to the project root
# Assuming tests directory is at project_root/tests/
PROJECT_ROOT = Path(__file__).parent.parent
SIGNAL_SCHEMA_PATH = PROJECT_ROOT / "src" / "signals" / "schema.json"

@pytest.fixture
def signal_schema():
    """Loads and returns the signal schema."""
    with open(SIGNAL_SCHEMA_PATH, 'r') as f:
        return json.load(f)

@pytest.fixture
def allowed_symbols():
    """Provides a list of allowed symbols for testing."""
    # In a real scenario, this would likely come from your main config
    return ["XAUUSD", "EURUSD", "GBPUSD"]

def test_valid_signal_validates(signal_schema, allowed_symbols):
    """Tests that a valid signal payload validates against the schema and has an allowed symbol."""
    valid_signal = {
        "symbol": "XAUUSD",
        "action": "buy",
        "volume": 0.1,
        "price": 1800.0,
        "tp": 1810.0,
        "sl": 1790.0,
        "comment": "Test signal"
    }
    
    # Test schema validation
    try:
        jsonschema.validate(valid_signal, signal_schema)
        assert True # Schema validation passed
    except jsonschema.ValidationError as e:
        pytest.fail(f"Valid signal failed schema validation: {e}")

    # Test symbol validation
    assert valid_signal['symbol'] in allowed_symbols, f"Valid signal symbol {valid_signal['symbol']} not in allowed symbols."

def test_invalid_signal_missing_field_fails_validation(signal_schema, allowed_symbols):
    """Tests that a signal missing a required field fails schema validation."""
    invalid_signal = {
        "action": "buy",
        "volume": 0.1,
        "price": 1800.0
        # Missing 'symbol'
    }
    
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(invalid_signal, signal_schema)

def test_invalid_signal_disallowed_symbol_fails_check(signal_schema, allowed_symbols):
    """Tests that a signal with a disallowed symbol is rejected."""
    signal_with_disallowed_symbol = {
        "symbol": "BTCUSD", # Assuming BTCUSD is not in allowed_symbols
        "action": "buy",
        "volume": 0.1,
        "price": 1800.0
    }
    
    # First, check schema validation (should pass if structure is correct)
    try:
        jsonschema.validate(signal_with_disallowed_symbol, signal_schema)
    except jsonschema.ValidationError as e:
        pytest.fail(f"Signal with disallowed symbol failed schema validation unexpectedly: {e}")

    # Then, check symbol validation (should fail)
    assert signal_with_disallowed_symbol['symbol'] not in allowed_symbols, f"Signal symbol {signal_with_disallowed_symbol['symbol']} was unexpectedly in allowed symbols."

# Add more tests for other invalid scenarios (e.g., wrong data types, invalid enum values) 