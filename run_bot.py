import argparse
import logging
import threading
import yaml
import os
import sys
import time
import pandas as pd
import pytz

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import modules
from mt5.mt5_client import MT5Client
from signals.telegram_handler import TelegramHandler
from live.live_trading import LiveTrading
from watchdog.watchdog import Watchdog
from ml.online_learner import OnlineLearner
from logging.snapshot_helper import SnapshotHelper # Uncommented
from logging.logger_setup import setup_logging
from backtests.backtester import Backtester # Import Backtester

# Configuration file path
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')

def load_config(config_path: str):
    """Loads the configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logging.critical(f"Configuration file not found at {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.critical(f"Error parsing configuration file: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='MT5 Trading Bot')
    parser.add_argument('mode', type=str, choices=['live', 'test'], help='Mode to run the bot in: live or test')
    # Add other arguments as needed (e.g., for backtesting parameters in test mode)
    parser.add_argument('--symbol', type=str, help='Symbol for backtesting or live trading (required in test mode)')
    parser.add_argument('--start-date', type=str, help='Start date for backtesting (YYYY-MM-DD) (required in test mode)')
    parser.add_argument('--end-date', type=str, help='End date for backtesting (YYYY-MM-DD) (required in test mode)')
    parser.add_argument('--timeframe', type=str, default='M30', help='Timeframe for backtesting (e.g., M30)')

    args = parser.parse_args()
    mode = args.mode

    # Load configuration
    config = load_config(CONFIG_PATH)

    # Setup logging
    setup_logging(config.get('logging', {}))
    logger = logging.getLogger(__name__)
    logger.info(f"Starting bot in {mode} mode...")

    mt5_client = None
    telegram_handler = None # Initialize as None
    live_trading = None
    watchdog = None
    online_learner = None # Initialize as None
    snapshot_helper = None # Initialize as None
    backtester = None # Initialize as None

    try:
        # Initialize MT5 Client (required for both modes)
        mt5_client = MT5Client(config.get('mt5', {}))
        if not mt5_client.is_connected():
            logger.critical("Failed to connect to MT5 terminal. Exiting.")
            sys.exit(1)

        # Initialize Optional Modules (OnlineLearner, SnapshotHelper) - can be done early
        if config.get('ml', {}).get('enabled', False):
             online_learner = OnlineLearner(config.get('ml', {}))
             logger.info("Online Learner initialized.")

        if config.get('snapshots', {}).get('enabled', False):
             snapshot_helper = SnapshotHelper(config.get('snapshots', {}))
             logger.info("Snapshot Helper initialized.")


        if mode == 'live':
            logger.info("Running in LIVE mode.")
            
            # Initialize Watchdog (only in live mode)
            watchdog = Watchdog(mt5_client, config.get('watchdog', {}))
            watchdog_thread = threading.Thread(target=watchdog.run)
            watchdog_thread.daemon = True
            watchdog_thread.start()
            logger.info("Watchdog started.")

            # Initialize Live Trading (only in live mode)
            # Pass dependencies: config, mt5_client, telegram_bot, online_learner, snapshot_helper
            # telegram_bot will be passed after telegram_handler is initialized
            live_trading = LiveTrading(config, mt5_client, telegram_bot=None, online_learner=online_learner, snapshot_helper=snapshot_helper)
            logger.info("Live Trading initialized.")

            # Start Live Trading loop for each symbol configured
            symbols = config.get('mt5', {}).get('symbols', [])
            if not symbols:
                 logger.critical("No symbols configured for live trading. Exiting.")
                 sys.exit(1)

            live_trading_threads = []
            for symbol in symbols:
                 thread = threading.Thread(target=live_trading.run, args=(symbol,))
                 thread.daemon = True
                 live_trading_threads.append(thread)
                 thread.start()
                 logger.info(f"Live trading started for {symbol}.")


        elif mode == 'test':
            logger.info("Test mode selected (Backtesting). Starting backtest...")
            
            # Check for required arguments in test mode
            if not args.symbol or not args.start_date or not args.end_date:
                 logger.critical("For test mode, --symbol, --start-date, and --end-date are required.")
                 sys.exit(1)

            try:
                 # Parse dates and localize to the backtest timezone from config
                 backtest_timezone_str = config.get('backtest', {}).get('timezone', 'UTC')
                 backtest_timezone = pytz.timezone(backtest_timezone_str)
                 
                 start_date = pd.to_datetime(args.start_date).tz_localize(backtest_timezone)
                 end_date = pd.to_datetime(args.end_date).tz_localize(backtest_timezone)
                 
                 # Ensure start date is before or equal to end date
                 if start_date > end_date:
                      logger.critical("Start date cannot be after end date.")
                      sys.exit(1)

            except Exception as e:
                 logger.critical(f"Invalid date format or timezone issue: {e}. Use YYYY-MM-DD.")
                 sys.exit(1)

            # Initialize Backtester (only in test mode)
            # Assuming Backtester needs config, mt5_client, and optionally snapshot_helper
            backtester = Backtester(config, mt5_client, snapshot_helper=snapshot_helper) # Pass snapshot_helper
            logger.info("Backtester initialized.")

            # Run the backtest (Blocking call in test mode)
            try:
                logger.info(f"Running backtest for symbol: {args.symbol}, timeframe: {args.timeframe}, period: {args.start_date} to {args.end_date}")

                # Assuming Backtester has a method to run the full backtest simulation or we call steps directly
                # Based on previous implementation in test mode, let's call steps directly.

                # 1. Load data
                data = backtester.load_data(args.symbol, args.timeframe, start_date, end_date)
                if data is None or data.empty:
                     logger.critical("Backtest aborted: Could not load data.")
                     sys.exit(1)

                # 2. Prepare data (e.g., handle missing values, basic cleaning)
                prepared_data = backtester.prepare_data(data.copy())

                # 3. Calculate indicators
                data_with_indicators = backtester.calculate_indicators(prepared_data.copy(), config.get('indicators', {}))

                # 4. Apply strategy rules to generate signals
                data_with_signals = backtester.apply_strategy(data_with_indicators.copy(), config.get('indicators', {}))

                # 5. Run backtest simulation
                metrics = backtester.run_backtest_simulation(data_with_signals.copy(), config)

                logger.info("Backtest run completed. Results:")
                print(metrics)

            except Exception as e:
                 logger.critical(f"An error occurred during backtesting: {e}", exc_info=True)
                 sys.exit(1)

            # Exit after backtest in test mode
            sys.exit(0)

        # Initialize and Start Telegram Handler (after other modules are potentially initialized)
        if config.get('telegram', {}).get('enabled', False):
            # Pass initialised instances of other modules
            telegram_handler = TelegramHandler(
                config.get('telegram', {}),
                config.get('mt5', {}).get('symbols', []),
                backtester=backtester,         # Pass backtester (will be None in live mode)
                mt5_client=mt5_client,         # Pass mt5_client
                watchdog_instance=watchdog,    # Pass watchdog (will be None in test mode)
                live_trading_instance=live_trading # Pass live_trading (will be None in test mode)
            )
            telegram_thread = threading.Thread(target=telegram_handler.run)
            telegram_thread.daemon = True # Allow main thread to exit even if telegram_thread is running
            telegram_thread.start()
            logger.info("Telegram handler started.")
            
            # Update live_trading with telegram_bot instance if it's in live mode
            if mode == 'live' and live_trading is not None:
                live_trading.telegram_bot = telegram_handler.application.bot # Pass the actual bot instance
                logger.info("Telegram bot instance passed to LiveTrading.")
        else:
            logger.info("Telegram is disabled in configuration.")


        # Keep the main thread alive only in live mode (test mode exits)
        if mode == 'live':
            # Keep the main thread alive while live trading threads are running
            # This simple loop will keep the main thread from exiting immediately.
            # In a more complex app, you might have a shutdown signal or mechanism.
            logger.info("Bot running in LIVE mode. Press Ctrl+C to stop.")
            while any(t.is_alive() for t in live_trading_threads):
                time.sleep(1)
            logger.info("All live trading threads stopped. Main thread exiting.")


    except Exception as e:
        logger.critical(f"An unexpected error occurred in the main process: {e}", exc_info=True)

    finally:
        # Clean up MT5 connection on exit
        if mt5_client:
            mt5_client.disconnect()
            logger.info("Disconnected from MT5 terminal.")
        logger.info("Bot stopped.")

if __name__ == "__main__":
    main()