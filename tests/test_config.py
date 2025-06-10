import pytest
import yaml
import json
from pathlib import Path
from typing import Dict, Any

# Import the ConfigLoader class from your source code
# Assuming the structure src/config/config_loader.py
import sys
# Temporarily add the src directory to the sys.path to import the module
# In a real test setup, you might configure PYTHONPATH differently or use a build tool
# Adjust the path based on your project structure if necessary
SRC_PATH = Path(__file__).parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:
    from config.config_loader import ConfigLoader, load_config
    # Remove the temporary path addition if it was added
    if str(SRC_PATH) in sys.path:
         sys.path.remove(str(SRC_PATH))
except ImportError as e:
    # Remove the temporary path addition even if import failed
    if str(SRC_PATH) in sys.path:
         sys.path.remove(str(SRC_PATH))
    pytest.fail(f"Could not import ConfigLoader. Make sure src/config/config_loader.py exists and is in your PYTHONPATH. Error: {e}")


# Define test data
# Use a minimal valid configuration based on schema.json
VALID_CONFIG_CONTENT = """
telegram:
  api_id: 123456
  api_hash: "abcdefg"
  chat_id: 789012
mt5:
  account: 987654
  server: "MetaQuotes-Demo"
  password: "password123"
  symbols:
    - "EURUSD"
    - "GBPUSD"
indicators:
  ema_periods:
    - 50
    - 200
  rsi_period: 14
  macd:
    fast: 12
    slow: 26
    signal: 9
  bollinger:
    period: 20
    sigma: 2.0
  atr_period: 14
order:
  tp_as_pips: true
  sl_as_pips: false
risk:
  risk_per_trade_pct: 1.0
backtest:
  commission_per_lot: 7.0
  slippage_pips: 2
  timezone: "Etc/UTC"
  fill_missing: true
ml:
  hyperparameters:
    learning_rate: 0.01
logs:
  path: "logs/bot.log"
  snapshots_dir: "snapshots"
  max_bytes: 10485760 # 10MB
  backup_count: 5
watchdog:
  check_interval_secs: 60
  telegram_max_lag_secs: 300
  mt5_max_lag_secs: 300
"""

INVALID_CONFIG_MISSING_FIELD = """
telegram:
  api_id: 123456
  api_hash: "abcdefg"
  # chat_id is missing
mt5:
  account: 987654
  server: "MetaQuotes-Demo"
  password: "password123"
  symbols:
    - "EURUSD"
    - "GBPUSD"
indicators:
  ema_periods:
    - 50
    - 200
  rsi_period: 14
  macd:
    fast: 12
    slow: 26
    signal: 9
  bollinger:
    period: 20
    sigma: 2.0
  atr_period: 14
order:
  tp_as_pips: true
  sl_as_pips: false
risk:
  risk_per_trade_pct: 1.0
backtest:
  commission_per_lot: 7.0
  slippage_pips: 2
  timezone: "Etc/UTC"
  fill_missing: true
ml:
  hyperparameters:
    learning_rate: 0.01
logs:
  path: "logs/bot.log"
  snapshots_dir: "snapshots"
  max_bytes: 10485760 # 10MB
  backup_count: 5
watchdog:
  check_interval_secs: 60
  telegram_max_lag_secs: 300
  mt5_max_lag_secs: 300
"""

INVALID_CONFIG_WRONG_TYPE = """
telegram:
  api_id: "not an integer"
  api_hash: "abcdefg"
  chat_id: 789012
mt5:
  account: 987654
  server: "MetaQuotes-Demo"
  password: "password123"
  symbols:
    - "EURUSD"
    - "GBPUSD"
indicators:
  ema_periods:
    - 50
    - 200
  rsi_period: 14
  macd:
    fast: 12
    slow: 26
    signal: 9
  bollinger:
    period: 20
    sigma: 2.0
  atr_period: 14
order:
  tp_as_pips: true
  sl_as_pips: false
risk:
  risk_per_trade_pct: 1.0
backtest:
  commission_per_lot: 7.0
  slippage_pips: 2
  timezone: "Etc/UTC"
  fill_missing: true
ml:
  hyperparameters:
    learning_rate: 0.01
logs:
  path: "logs/bot.log"
  snapshots_dir: "snapshots"
  max_bytes: 10485760 # 10MB
  backup_count: 5
watchdog:
  check_interval_secs: 60
  telegram_max_lag_secs: 300
  mt5_max_lag_secs: 300
"""

INVALID_YAML_CONTENT = """
telegram:
  api_id: 123456
  api_hash: "abcdefg"
  chat_id: 789012

this is not valid yaml
  indented_line: here
"""

# Use the schema content provided earlier
VALID_SCHEMA_CONTENT = """
{
  "type": "object",
  "properties": {
    "telegram": {
      "type": "object",
      "properties": {
        "api_id": {
          "type": "integer"
        },
        "api_hash": {
          "type": "string"
        },
        "chat_id": {
          "type": "integer"
        }
      },
      "required": [
        "api_id",
        "api_hash",
        "chat_id"
      ]
    },
    "mt5": {
      "type": "object",
      "properties": {
        "account": {
          "type": "integer"
        },
        "server": {
          "type": "string"
        },
        "password": {
          "type": "string"
        },
        "symbols": {
          "type": "array",
          "items": {
            "type": "string"
          }
        }
      },
      "required": [
        "account",
        "server",
        "password",
        "symbols"
      ]
    },
    "indicators": {
      "type": "object",
      "properties": {
        "ema_periods": {
          "type": "array",
          "items": {
            "type": "integer"
          }
        },
        "rsi_period": {
          "type": "integer"
        },
        "macd": {
          "type": "object",
          "properties": {
            "fast": {
              "type": "integer"
            },
            "slow": {
              "type": "integer"
            },
            "signal": {
              "type": "integer"
            }
          },
          "required": [
            "fast",
            "slow",
            "signal"
          ]
        },
        "bollinger": {
          "type": "object",
          "properties": {
            "period": {
              "type": "integer"
            },
            "sigma": {
              "type": "number"
            }
          },
          "required": [
            "period",
            "sigma"
          ]
        },
        "atr_period": {
          "type": "integer"
        }
      },
      "required": [
        "ema_periods",
        "rsi_period",
        "macd",
        "bollinger",
        "atr_period"
      ]
    },
    "order": {
      "type": "object",
      "properties": {
        "tp_as_pips": {
          "type": "boolean"
        },
        "sl_as_pips": {
          "type": "boolean"
        }
      },
      "required": [
        "tp_as_pips",
        "sl_as_pips"
      ]
    },
    "risk": {
      "type": "object",
      "properties": {
        "risk_per_trade_pct": {
          "type": "number"
        }
      },
      "required": [
        "risk_per_trade_pct"
      ]
    },
    "backtest": {
      "type": "object",
      "properties": {
        "commission_per_lot": {
          "type": "number"
        },
        "slippage_pips": {
          "type": "number"
        },
        "timezone": {
          "type": "string"
        },
        "fill_missing": {
          "type": "boolean"
        }
      },
      "required": [
        "commission_per_lot",
        "slippage_pips",
        "timezone",
        "fill_missing"
      ]
    },
    "ml": {
      "type": "object",
      "properties": {
        "hyperparameters": {
          "type": "object",
          "properties": {
            "learning_rate": {
              "type": "number"
            }
          },
          "required": [
            "learning_rate"
          ]
        }
      },
      "required": [
        "hyperparameters"
      ]
    },
    "logs": {
      "type": "object",
      "properties": {
        "path": {
          "type": "string"
        },
        "snapshots_dir": {
          "type": "string"
        },
        "max_bytes": {
          "type": "integer"
        },
        "backup_count": {
          "type": "integer"
        }
      },
      "required": [
        "path",
        "snapshots_dir",
        "max_bytes",
        "backup_count"
      ]
    },
    "watchdog": {
      "type": "object",
      "properties": {
        "check_interval_secs": {
          "type": "integer"
        },
        "telegram_max_lag_secs": {
          "type": "integer"
        },
        "mt5_max_lag_secs": {
          "type": "integer"
        }
      },
      "required": [
        "check_interval_secs",
        "telegram_max_lag_secs",
        "mt5_max_lag_secs"
      ]
    }
  },
  "required": [
    "telegram",
    "mt5",
    "indicators",
    "order",
    "risk",
    "backtest",
    "ml",
    "logs",
    "watchdog"
  ]
}
"""

@pytest.fixture
def config_dir(tmp_path):
    """Fixture to create a temporary config directory."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    return cfg_dir

@pytest.fixture
def valid_config_file(config_dir):
    """Fixture to create a valid config.yaml file."""
    config_path = config_dir / "config.yaml"
    config_path.write_text(VALID_CONFIG_CONTENT, encoding="utf-8")
    return config_path

@pytest.fixture
def valid_schema_file(config_dir):
    """Fixture to create a valid schema.json file."""
    schema_path = config_dir / "schema.json"
    schema_path.write_text(VALID_SCHEMA_CONTENT, encoding="utf-8")
    return schema_path

@pytest.fixture
def invalid_config_missing_field_file(config_dir):
    """Fixture to create a config.yaml with a missing required field."""
    config_path = config_dir / "config.yaml"
    config_path.write_text(INVALID_CONFIG_MISSING_FIELD, encoding="utf-8")
    return config_path

@pytest.fixture
def invalid_config_wrong_type_file(config_dir):
    """Fixture to create a config.yaml with a wrong data type."""
    config_path = config_dir / "config.yaml"
    config_path.write_text(INVALID_CONFIG_WRONG_TYPE, encoding="utf-8")
    return config_path

@pytest.fixture
def invalid_yaml_file(config_dir):
    """Fixture to create a config.yaml with invalid YAML syntax."""
    config_path = config_dir / "config.yaml"
    config_path.write_text(INVALID_YAML_CONTENT, encoding="utf-8")
    return config_path


def test_load_valid_config(valid_config_file, valid_schema_file):
    """Test successful loading and validation of a valid config."""
    config_data = load_config(str(valid_config_file))
    assert isinstance(config_data, dict)
    # Optionally assert some known values from the config
    assert config_data['telegram']['api_id'] == 123456
    assert config_data['mt5']['symbols'] == ["EURUSD", "GBPUSD"]

def test_load_missing_config_file(config_dir, valid_schema_file):
    """Test loading when config.yaml is missing."""
    missing_config_path = config_dir / "non_existent_config.yaml"
    with pytest.raises(FileNotFoundError) as excinfo:
        load_config(str(missing_config_path))
    assert f"Config file not found" in str(excinfo.value)

def test_load_missing_schema_file(valid_config_file, tmp_path):
    """Test loading when schema.json is missing."""
    # Create a config directory without a schema file
    alt_config_dir = tmp_path / "alt_config"
    alt_config_dir.mkdir()
    alt_config_path = alt_config_dir / "config.yaml"
    alt_config_path.write_text(VALID_CONFIG_CONTENT, encoding="utf-8")

    with pytest.raises(FileNotFoundError) as excinfo:
        load_config(str(alt_config_path))
    assert f"Config schema not found" in str(excinfo.value)

def test_load_invalid_yaml(invalid_yaml_file, valid_schema_file):
    """Test loading a config file with invalid YAML syntax."""
    with pytest.raises(yaml.YAMLError) as excinfo:
        load_config(str(invalid_yaml_file))
    # Check if the exception is a YAML parsing error
    assert isinstance(excinfo.value, yaml.YAMLError)

# Need to import jsonschema for this test
import jsonschema

def test_load_invalid_config_schema_missing_field(invalid_config_missing_field_file, valid_schema_file):
    """Test loading a config file with missing required fields according to the schema."""
    with pytest.raises(jsonschema.ValidationError) as excinfo:
        load_config(str(invalid_config_missing_field_file))
    # Check for a specific part of the expected error message
    assert "is a required property" in str(excinfo.value)

def test_load_invalid_config_schema_wrong_type(invalid_config_wrong_type_file, valid_schema_file):
    """Test loading a config file with incorrect data types according to the schema."""
    with pytest.raises(jsonschema.ValidationError) as excinfo:
        load_config(str(invalid_config_wrong_type_file))
    # Check for a specific part of the expected error message, like type mismatch
    assert "is not of type 'integer'" in str(excinfo.value) or "is not valid under any of the given schemas" in str(excinfo.value)

# Optionally, test the ConfigLoader class directly if its methods are public and intended for direct use
# def test_config_loader_instance(valid_config_file, valid_schema_file):
#     loader = ConfigLoader(str(valid_config_file))
#     config_data = loader.load()
#     assert isinstance(config_data, dict)
#     assert loader.config_path == Path(valid_config_file)
#     assert loader.schema_path == Path(valid_config_file).parent / "schema.json"
 