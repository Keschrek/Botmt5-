import logging
import pandas as pd
import pytz
import ta # Import the TA library
import itertools
from typing import Dict, Any, List, Tuple
import numpy as np
import json
from pathlib import Path

# Assume MT5Client is in src.mt5.mt5_client
from src.mt5.mt5_client import MT5Client, get_mt5_timeframe
# Import SnapshotHelper
from src.logging.snapshot_helper import SnapshotHelper # Import the class

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, config: Dict[str, Any], mt5_client: MT5Client, snapshot_helper: SnapshotHelper = None):
        self.config = config
        self.mt5_client = mt5_client
        self.backtest_cfg = config['backtest']
        self.indicators_cfg = config['indicators']
        self.order_cfg = config['order']
        self.risk_cfg = config['risk']
        self.allowed_symbols = config['mt5']['symbols']
        self.snapshot_helper = snapshot_helper # Store the snapshot helper instance
        
        self.timezone = pytz.timezone(self.backtest_cfg['timezone'])
        
        logger.info("Backtester initialized.")

    def load_data(self, symbol: str, timeframe_str: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame | None:
        """Loads historical data from MT5 for a given period."""
        mt5_timeframe = get_mt5_timeframe(timeframe_str)
        if mt5_timeframe is None:
            logger.error(f"Invalid MT5 timeframe specified: {timeframe_str}")
            return None

        # MT5 loads data by position or by time range. Loading by time range is better for specific dates.
        # Need to convert pandas timestamps to datetime objects for MT5
        start_dt = start_date.to_pydatetime()
        end_dt = end_date.to_pydatetime()

        logger.info(f"Loading data for {symbol} ({timeframe_str}) from {start_dt} to {end_dt}")

        # MT5 copy_rates_from_time_range returns numpy array
        rates = self.mt5_client.get_rates_range(symbol, mt5_timeframe, start_dt, end_dt) # Assuming get_rates_range method exists or using copy_rates_from_time_range directly
        
        if rates is None:
            logger.warning(f"Could not load data for {symbol} ({timeframe_str}).")
            return None
            
        # Convert numpy array to pandas DataFrame
        rates_df = pd.DataFrame(rates)
        # Convert time in seconds to datetime and set as index
        rates_df['time'] = pd.to_datetime(rates_df['time'], unit='s')
        rates_df = rates_df.set_index('time')
        
        # Apply timezone info
        # MT5 times are typically broker time. Need to make timezone aware if not already and convert.
        # Assuming MT5 returns times in UTC or broker time that needs tz-info
        # This part might need adjustment based on actual MT5 behavior and your needs
        # Example: Assuming MT5 time is UTC, then convert to the configured timezone
        # rates_df.index = rates_df.index.tz_localize('UTC').tz_convert(self.timezone)
        # Or if MT5 time is already timezone-aware broker time:
        # rates_df.index = rates_df.index.tz_convert(self.timezone)
        # For simplicity here, we assume MT5 time is naive UTC and localize/convert
        try:
             rates_df.index = rates_df.index.tz_localize('UTC').tz_convert(self.timezone)
        except Exception as e:
             logger.warning(f"Could not localize/convert timezone for data: {e}. Proceeding with naive timestamps.")

        logger.info(f"Loaded {len(rates_df)} data points for {symbol} ({timeframe_str}).")
        return rates_df

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Performs data integrity checks and preparation (e.g., forward-fill)."""
        if self.backtest_cfg.get('fill_missing', False):
            logger.info("Applying forward-fill for missing data.")
            # Forward-fill missing values in OHLCV data
            data = data.fillna(method='ffill')
            # TODO: Add more sophisticated data cleaning if needed (e.g., outlier filtering)
            
        return data

    def calculate_indicators(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculates technical indicators and adds them to the DataFrame."""
        logger.info("Calculating indicators...")
        # Copy data to avoid modifying the original DataFrame
        data = data.copy()

        # Add indicators using the `ta` library
        # The `ta` library adds columns in place or returns a new DataFrame depending on version/method

        # EMA
        for period in params.get('ema_periods', []):
            data[f'EMA_{period}'] = ta.trend.ema_indicator(data['close'], window=period)

        # RSI
        rsi_period = params.get('rsi_period', 14)
        data['RSI'] = ta.momentum.rsi(data['close'], window=rsi_period)

        # MACD
        macd_params = params.get('macd', {})
        data['MACD_line'] = ta.trend.macd_line(data['close'], window_fast=macd_params.get('fast', 12), window_slow=macd_params.get('slow', 26))
        data['MACD_signal'] = ta.trend.macd_signal(data['close'], window_fast=macd_params.get('fast', 12), window_slow=macd_params.get('slow', 26), window=macd_params.get('signal', 9))
        data['MACD_hist'] = data['MACD_line'] - data['MACD_signal']

        # Bollinger Bands
        bollinger_params = params.get('bollinger', {})
        indicator_bb = ta.volatility.BollingerBands(
            close=data['close'], 
            window=bollinger_params.get('period', 20),
            window_dev=bollinger_params.get('sigma', 2.0)
        )
        data['BB_upper'] = indicator_bb.bollinger_hband()
        data['BB_lower'] = indicator_bb.bollinger_lband()

        # ATR
        atr_period = params.get('atr_period', 14)
        data['ATR'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'], window=atr_period)

        # Drop rows with NaN values created by indicator calculations
        data = data.dropna()
        logger.info(f"Indicators calculated. Data shape after dropping NaNs: {data.shape}")

        return data

    def apply_strategy(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Applies the trading strategy rules to the data."""
        logger.info("Applying trading strategy...")
        
        # Add a 'signal' column to the DataFrame (e.g., 1 for buy, -1 for sell, 0 for hold)
        data['signal'] = 0

        # Assuming EMA periods are in params['ema_periods'] and sorted (e.g., [8, 21])
        # Ensure ema_periods has at least two elements
        ema_periods = params.get('ema_periods', [])
        if len(ema_periods) < 2:
             logger.error("Insufficient EMA periods configured for strategy. Cannot apply rules.")
             return data

        ema_short_period = ema_periods[0] # Example: 8
        ema_long_period = ema_periods[1] # Example: 21

        # Define the conditions for Long and Short signals
        # Check if required indicator columns exist before applying rules
        required_cols = [
            f'EMA_{ema_short_period}',
            f'EMA_{ema_long_period}',
            'RSI',
            'MACD_hist',
            'BB_lower',
            'BB_upper',
            'close',
            'low', # Needed for 'touches lower Bollinger'
            'high' # Needed for 'touches upper Bollinger'
        ]

        if not all(col in data.columns for col in required_cols):
             missing_cols = [col for col in required_cols if col not in data.columns]
             logger.error(f"Missing required indicator columns for strategy application: {missing_cols}. Cannot apply rules.")
             return data

        # Rules:
        # Long: EMA8 > EMA21 ∧ RSI ∈ (30, 70) ∧ MACD-Hist > 0 ∧ Kurs berührt unteres Bollinger
        # Short: EMA8 < EMA21 ∧ RSI ∈ (30, 70) ∧ MACD-Hist < 0 ∧ Kurs berührt oberes Bollinger

        # Define boolean conditions using pandas for vectorized operations
        long_condition = (
            (data[f'EMA_{ema_short_period}'] > data[f'EMA_{ema_long_period}']) &
            (data['RSI'] > 30) & (data['RSI'] < 70) &
            (data['MACD_hist'] > 0) &
            (data['low'] <= data['BB_lower']) # Kurs berührt/kreuzt unteres Bollinger mit dem Tiefstkurs
        )

        short_condition = (
            (data[f'EMA_{ema_short_period}'] < data[f'EMA_{ema_long_period}']) &
            (data['RSI'] > 30) & (data['RSI'] < 70) &
            (data['MACD_hist'] < 0) &
            (data['high'] >= data['BB_upper']) # Kurs berührt/kreuzt oberes Bollinger mit dem Höchstkurs
        )

        # Apply the signals to the 'signal' column
        data.loc[long_condition, 'signal'] = 1 # Assign 1 for Long signals
        data.loc[short_condition, 'signal'] = -1 # Assign -1 for Short signals

        # Note: In a real strategy, you might need to add logic to avoid opening multiple positions,
        # handle position sizing based on risk, and potentially include exit conditions.
        # This method focuses purely on generating entry signals based on indicator conditions.

        logger.info("Trading strategy applied. Signals generated.")
        return data

    def run_backtest_simulation(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Runs a backtest simulation based on signals and calculates performance."""
        logger.info("Running backtest simulation...")
        # This is where you would simulate trades based on the 'signal' column
        # Account for commission_per_lot and slippage_pips from backtest_cfg

        # Initialize simulation variables
        initial_capital = 10000.0 # Example starting capital
        capital = initial_capital
        position = 0 # 0 for no position, >0 for long volume, <0 for short volume
        trades = []
        equity_curve = pd.Series(index=data.index, dtype=float)
        entry_price = None
        trade_comment = ""

        commission_per_lot = self.backtest_cfg.get('commission_per_lot', 0.0)
        # Slippage is often defined in points or pips. Need symbol info to convert pips to points.
        # For simplicity in simulation, let's assume slippage_pips affects the execution price directly
        # by adding/subtracting a value equivalent to slippage_pips * point value.
        # We need the point value for the symbol, which we can get from MT5Client or pass here.
        # For simulation, we might need to estimate it or fetch symbol info for the backtested symbol.
        # Let's assume we have a way to get the point value or calculate slippage in price units.

        # Placeholder for getting symbol point value - ideally fetch this once per symbol backtest
        # symbol_info = mt5.symbol_info(data['symbol'].iloc[0]) # This won't work as data doesn't have 'symbol' col
        # point = symbol_info.point if symbol_info else 0.00001 # Example default point
        # For simplicity, let's assume slippage_pips is a small price adjustment for now.
        slippage_adjust = self.backtest_cfg.get('slippage_pips', 0.0) # Simplified: treating slippage_pips as a direct price adjustment per unit
        logger.warning("Backtest simulation uses simplified slippage calculation.")


        for i, (timestamp, row) in enumerate(data.iterrows()):
            equity = capital + position * row['close'] # Current equity
            equity_curve[timestamp] = equity

            signal = row['signal']

            # Check for closing position (opposite signal or end of data)
            if position > 0 and (signal == -1 or i == len(data) - 1): # Close long position
                exit_price = row['close']
                # Apply slippage on exit price (opposite direction of entry slippage)
                # Simplified: subtract slippage for long exit
                adjusted_exit_price = exit_price - slippage_adjust

                trade_pl = (adjusted_exit_price - entry_price) * position
                commission_cost = commission_per_lot * abs(position) / 1.0 # Assuming volume in lots
                net_pl = trade_pl - commission_cost

                capital += net_pl
                trades.append({
                    'entry_time': entry_price_timestamp,
                    'exit_time': timestamp,
                    'symbol': "BACKTEST_SYMBOL", # Placeholder
                    'action': 'buy', # Was a long position
                    'volume': position,
                    'entry_price': entry_price,
                    'exit_price': adjusted_exit_price,
                    'gross_pl': trade_pl,
                    'commission': commission_cost,
                    'net_pl': net_pl,
                    'comment': trade_comment
                })
                logger.info(f"Closed Long @ {adjusted_exit_price:.5f} (P/L: {net_pl:.2f})")
                position = 0
                entry_price = None
                trade_comment = ""

            elif position < 0 and (signal == 1 or i == len(data) - 1): # Close short position
                exit_price = row['close']
                # Apply slippage on exit price (opposite direction of entry slippage)
                # Simplified: add slippage for short exit
                adjusted_exit_price = exit_price + slippage_adjust

                trade_pl = (entry_price - adjusted_exit_price) * abs(position)
                commission_cost = commission_per_lot * abs(position) / 1.0 # Assuming volume in lots
                net_pl = trade_pl - commission_cost

                capital += net_pl
                trades.append({
                    'entry_time': entry_price_timestamp,
                    'exit_time': timestamp,
                    'symbol': "BACKTEST_SYMBOL", # Placeholder
                    'action': 'sell', # Was a short position
                    'volume': abs(position),
                    'entry_price': entry_price,
                    'exit_price': adjusted_exit_price,
                    'gross_pl': trade_pl,
                    'commission': commission_cost,
                    'net_pl': net_pl,
                     'comment': trade_comment
                })
                logger.info(f"Closed Short @ {adjusted_exit_price:.5f} (P/L: {net_pl:.2f})")
                position = 0
                entry_price = None
                trade_comment = ""

            # Check for opening position
            if position == 0:
                if signal == 1: # Open long position
                    # Determine volume based on risk per trade
                    # This requires knowing account balance (current capital) and stop loss distance
                    # For simplicity, using a fixed volume for now.
                    # TODO: Implement risk-based volume calculation
                    volume_to_open = 0.1 # Placeholder
                    
                    entry_price = row['close']
                    # Apply slippage on entry price (adverse to trade direction)
                    # Simplified: add slippage for long entry
                    adjusted_entry_price = entry_price + slippage_adjust

                    position = volume_to_open # Volume is positive for long
                    entry_price_timestamp = timestamp
                    entry_price = adjusted_entry_price
                    trade_comment = f"Opened Long @ {adjusted_entry_price:.5f}"
                    logger.info(trade_comment)

                elif signal == -1: # Open short position
                     # Determine volume based on risk per trade
                     # TODO: Implement risk-based volume calculation
                     volume_to_open = 0.1 # Placeholder

                     entry_price = row['close']
                     # Apply slippage on entry price (adverse to trade direction)
                     # Simplified: subtract slippage for short entry
                     adjusted_entry_price = entry_price - slippage_adjust

                     position = -volume_to_open # Volume is negative for short
                     entry_price_timestamp = timestamp
                     entry_price = adjusted_entry_price
                     trade_comment = f"Opened Short @ {adjusted_entry_price:.5f}"
                     logger.info(trade_comment)

        # After loop, if a position is still open, close it at the last price
        if position != 0:
             logger.info("Closing remaining position at end of data.")
             # Simulate closing the position at the very last data point's close price
             last_row = data.iloc[-1]
             exit_price = last_row['close']
             # Apply slippage on exit price
             if position > 0: # Long position
                  adjusted_exit_price = exit_price - slippage_adjust
             else: # Short position
                  adjusted_exit_price = exit_price + slippage_adjust

             trade_pl = (adjusted_exit_price - entry_price) * position if position > 0 else (entry_price - adjusted_exit_price) * abs(position)
             commission_cost = commission_per_lot * abs(position) / 1.0
             net_pl = trade_pl - commission_cost

             capital += net_pl
             trades.append({
                 'entry_time': entry_price_timestamp,
                 'exit_time': data.index[-1],
                 'symbol': "BACKTEST_SYMBOL", # Placeholder
                 'action': 'buy' if position > 0 else 'sell',
                 'volume': abs(position),
                 'entry_price': entry_price,
                 'exit_price': adjusted_exit_price,
                 'gross_pl': trade_pl,
                 'commission': commission_cost,
                 'net_pl': net_pl,
                  'comment': trade_comment + " (Closed at end)"
             })
             logger.info(f"Closed remaining position @ {adjusted_exit_price:.5f} (P/L: {net_pl:.2f})")
             position = 0 # Ensure position is reset


        # Calculate performance metrics
        # Use the equity curve to calculate metrics like total return, Sharpe Ratio, Max Drawdown
        # Ensure equity_curve starts with initial capital at the very beginning of the data index
        # Even if the first signal occurs later, the equity starts from the first timestamp.
        # Re-index equity_curve to match data index and forward fill initial capital before first trade
        full_equity_curve = pd.Series(index=data.index, dtype=float)
        if not equity_curve.dropna().empty:
             first_trade_time = equity_curve.first_valid_index()
             # Fill equity curve before the first trade with initial capital
             full_equity_curve.loc[:first_trade_time] = initial_capital
             # Fill the rest with calculated equity values
             full_equity_curve.update(equity_curve)
        else:
             # No trades occurred, equity remains initial capital
             full_equity_curve[:] = initial_capital
             
        # Forward fill any NaNs that might still exist (shouldn't happen with correct logic)
        full_equity_curve = full_equity_curve.ffill()


        total_return = (full_equity_curve.iloc[-1] / full_equity_curve.iloc[0]) - 1.0 if full_equity_curve.iloc[0] != 0 else 0.0
        num_trades = len(trades)

        # Calculate Sharpe Ratio and Max Drawdown using a library or manually
        # Using `ffn` library is convenient for this
        try:
            import ffn # Import ffn library
            # ffn works well with a price series. Convert equity curve to a price series starting from 1.0
            price_series = full_equity_curve / initial_capital # Normalize equity curve
            
            # Calculate metrics using ffn
            # Ensure the price series has a frequency or is sorted by time
            # price_series = price_series.sort_index()
            
            # Handle cases with no trades where price_series might be flat
            if price_series.nunique() <= 1:
                 sharpe_ratio = 0.0
                 max_drawdown = 0.0
                 logger.warning("No price movement in equity curve, setting Sharpe and Drawdown to 0.")
            else:
                 # Calculate metrics. Risk free rate is assumed to be 0.
                 # Annualized Sharpe Ratio calculation assumes daily data by default. 
                 # For other frequencies (like M30 bars), you might need to adjust the annualization factor.
                 # ffn.PerformanceStats calculates daily Sharpe by default, then annualizes by sqrt(252).
                 # For M30 data, you might need sqrt(number of bars per year / number of bars per day * 252)? Complex.
                 # Let's calculate a simple Sharpe based on the *total* return over the backtest period.
                 # Annualized Sharpe = (Mean Daily Return - Risk Free Rate) / Std Dev of Daily Returns * sqrt(Annualization Factor)
                 # For simplicity, let's use ffn's default or calculate a simplified version if needed.

                 # Using ffn's performance stats for comprehensive metrics
                 perf_stats = price_series.calc_stats()
                 sharpe_ratio = perf_stats.sharpe
                 max_drawdown = perf_stats.max_drawdown
                 # ffn might provide more detailed trade stats too

                 # Example: Calculate win rate manually from trades list
                 winning_trades = [t for t in trades if t['net_pl'] > 0]
                 win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0.0


        except ImportError:
            logger.error("ffn library not found. Cannot calculate Sharpe Ratio and Max Drawdown. Please install it (`pip install ffn`).")
            sharpe_ratio = 0.0 # Default to 0 if library is missing
            max_drawdown = 0.0 # Default to 0
            # Calculate win rate manually
            winning_trades = [t for t in trades if t['net_pl'] > 0]
            win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0.0

        except Exception as e:
             logger.error(f"Error calculating performance metrics: {e}")
             sharpe_ratio = 0.0
             max_drawdown = 0.0
             winning_trades = [t for t in trades if t['net_pl'] > 0]
             win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0.0


        performance_metrics = {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": abs(max_drawdown), # Max Drawdown is typically reported as positive
            "num_trades": num_trades,
            "win_rate": win_rate,
            # Add other metrics as needed
        }

        logger.info(f"Backtest Simulation complete. Metrics: {performance_metrics}")
        # You might want to return the trades list and equity curve as well for detailed analysis/plotting
        return performance_metrics, trades, full_equity_curve

    def format_backtest_results(self, metrics: Dict[str, Any], trades: List[Dict[str, Any]], equity_curve: pd.Series, params: Dict[str, Any] = None) -> str:
        """Formats backtest results into a human-readable string."""
        formatted_output = "📊 **Backtest Ergebnisse** 📊\n"
        formatted_output += "------------------------------------\n"
        if params:
             formatted_output += f"Parameter: {params}\n"
             formatted_output += "------------------------------------\n"

        formatted_output += f"Gesamtrendite: {metrics.get('total_return', 0.0):.2%}\n"
        formatted_output += f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0.0):.2f}\n"
        formatted_output += f"Max Drawdown: {metrics.get('max_drawdown', 0.0):.2%}\n"
        formatted_output += f"Anzahl Trades: {metrics.get('num_trades', 0)}\n"
        formatted_output += f"Win Rate: {metrics.get('win_rate', 0.0):.2%}\n"
        formatted_output += "------------------------------------\n"

        if trades:
             formatted_output += f"Beispiel Trades ({min(5, len(trades))} von {len(trades)}):\n"
             # Display a few example trades
             for trade in trades[:5]: # Show first 5 trades
                 formatted_output += (
                     f"  - {trade.get('action','N/A').capitalize()} {trade.get('volume', 0):.2f} lots @ {trade.get('entry_price', 0.0):.5f} "
                     f"-> {trade.get('exit_price', 0.0):.5f} | P/L: {trade.get('net_pl', 0.0):.2f}\n"
                 )
             # Add summary of total P/L for all trades if needed
             total_net_pl = sum(t.get('net_pl', 0) for t in trades)
             formatted_output += f"Gesamter Netto-P/L aus Trades: {total_net_pl:.2f}\n"
        else:
            formatted_output += "Keine Trades ausgeführt.\n"

        # Note: Equity curve could be visualized separately, not easily included in a string.

        return formatted_output

    def save_backtest_results(self, metrics: Dict[str, Any], trades: List[Dict[str, Any]], equity_curve: pd.Series, params: Dict[str, Any] = None, filename_prefix: str = "backtest_results"):
        """Saves backtest results to a file."""
        # Define the save directory and filename
        results_dir = Path(self.config.get('logs', {}).get('backtest_results_dir', 'logs/backtests/'))
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp_str = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        sanitized_prefix = "".join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in filename_prefix)
        # Use .json extension for structured data, or .txt for formatted string
        # Saving as JSON allows easier programmatic access later
        results_filename = f"{sanitized_prefix}_{timestamp_str}.json"
        results_path = results_dir / results_filename

        # Prepare data for saving
        results_data = {
            'timestamp': timestamp_str,
            'parameters': params if params else {}, # Include parameters if available
            'metrics': metrics,
            'trades': trades,
            'equity_curve': equity_curve.astype(float).tolist(), # Convert Series to list for JSON, ensure float type
            'equity_curve_index': [str(ts) for ts in equity_curve.index.tolist()] # Save index as strings
        }

        try:
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=4)
            logger.info(f"Backtest results saved successfully to: {results_path}")
        except Exception as e:
            logger.error(f"Failed to save backtest results to {results_path}: {e}")

    def run_grid_search(self, symbol: str, timeframe_str: str, start_date: pd.Timestamp, end_date: pd.Timestamp, param_grid: Dict[str, List]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Runs a grid search over the given parameter grid."""
        logger.info(f"Starting grid search for {symbol} ({timeframe_str}) from {start_date.date()} to {end_date.date()} with parameter grid: {param_grid}")

        if symbol not in self.allowed_symbols:
            logger.error(f"Symbol {symbol} not allowed for backtesting.")
            return {}, {}

        # Load data once for the entire date range
        data = self.load_data(symbol, timeframe_str, start_date, end_date)
        if data is None or data.empty:
            logger.error("Failed to load data for grid search.")
            return {}, {}

        data = self.prepare_data(data)

        best_params = None
        best_performance = None
        best_trades = [] # Store trades for the best performance
        best_equity_curve = pd.Series(dtype=float) # Store equity curve for the best performance

        # Generate all combinations of parameters
        param_combinations = list(itertools.product(*param_grid.values()))
        keys = list(param_grid.keys())
        total_combinations = len(param_combinations)
        logger.info(f"Generated {total_combinations} parameter combinations.")

        for i, combo_values in enumerate(param_combinations):
            current_params = dict(zip(keys, combo_values))
            logger.info(f"Running backtest for combination {i+1}/{total_combinations}: {current_params}")
            
            try:
                # Calculate indicators and apply strategy with current parameters
                data_with_indicators = self.calculate_indicators(data.copy(), current_params) # Use a copy
                data_with_signals = self.apply_strategy(data_with_indicators, current_params) # Use a copy

                # Run simulation - now returns trades and equity curve
                performance, trades, equity_curve = self.run_backtest_simulation(data_with_signals, current_params)
                logger.info(f"Performance for {current_params}: {performance.get('total_return', 'N/A')}") # Log key metric

                # Evaluate performance (e.g., based on total_return, sharpe_ratio, etc.)
                # You can customize the evaluation metric here
                current_metric = performance.get('total_return', -float('inf')) # Use a metric for comparison

                if best_performance is None or current_metric > best_performance.get('total_return', -float('inf')):
                    best_performance = performance
                    best_params = current_params
                    best_trades = trades # Store trades for the best performance
                    best_equity_curve = equity_curve # Store equity curve for the best performance
                    logger.info(f"New best performance found: {current_metric} with parameters {best_params}")

            except Exception as e:
                logger.error(f"Error running backtest for parameters {current_params}: {e}")
                # Continue to the next combination even if one fails
                continue

        logger.info("Grid search finished.")
        if best_params:
            logger.info(f"Best parameters found: {best_params}")
            logger.info(f"Best performance: {best_performance}")
            
            # Format and save best results
            formatted_results = self.format_backtest_results(best_performance, best_trades, best_equity_curve, best_params)
            logger.info("Best backtest results formatted.\n" + formatted_results)
            
            try:
                 # Pass best_params to save_backtest_results
                 self.save_backtest_results(best_performance, best_trades, best_equity_curve, params=best_params, filename_prefix=f'grid_search_{symbol}_{timeframe_str}')
            except Exception as e:
                 logger.error(f"Failed to save grid search results: {e}")

            # Plot the best equity curve if snapshot_helper is available
            if self.snapshot_helper:
                 try:
                      self.snapshot_helper.save_equity_curve_snapshot(best_equity_curve, filename_prefix=f'grid_search_best_{symbol}_{timeframe_str}')
                 except Exception as e:
                      logger.error(f"Failed to plot equity curve for grid search: {e}")

        else:
            logger.warning("No successful backtests completed during grid search.")

        # Return best parameters and performance metrics
        return best_params if best_params else {}, best_performance if best_performance else {}

    def run_backtest_command(self, symbol: str, timeframe_str: str = "M30", reply_func=None):
        """Triggers a backtest run for a specific symbol and timeframe (e.g., via Telegram /test command)."""
        logger.info(f"Received backtest command for {symbol} ({timeframe_str}).")

        if symbol not in self.allowed_symbols:
            logger.warning(f"Backtest requested for disallowed symbol: {symbol}")
            if reply_func:
                 reply_func("Dieser Symbol ist für Backtesting nicht erlaubt.")
            return

        # Define the date range (last 6 months)
        end_date = pd.Timestamp.now(tz=self.timezone) # Use timezone-aware timestamp
        # Calculate start date by subtracting 6 months. Using DateOffset handles month ends correctly.
        start_date = end_date - pd.DateOffset(months=6)

        logger.info(f"Backtesting period: {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}") # Include time in log

        # For a single backtest command, we typically use the parameters from the main config
        current_params = self.indicators_cfg # Use indicator config as parameters

        # Load, prepare, calculate indicators, and apply strategy with default config params
        data = self.load_data(symbol, timeframe_str, start_date, end_date)
        if data is None or data.empty:
            logger.error("Backtest aborted due to data loading failure.")
            if reply_func:
                 reply_func("Fehler beim Laden der historischen Daten für Backtest.")
            return

        prepared_data = self.prepare_data(data.copy())
        data_with_indicators = self.calculate_indicators(prepared_data.copy(), current_params) # Use indicators_cfg as params
        data_with_signals = self.apply_strategy(data_with_indicators.copy(), current_params) # Use indicators_cfg as params

        # Run simulation - now returns trades and equity curve
        performance, trades, equity_curve = self.run_backtest_simulation(data_with_signals.copy(), current_params)

        # Format the results
        formatted_results = self.format_backtest_results(performance, trades, equity_curve, params=current_params) # Pass current_params

        # Save the results
        try:
             self.save_backtest_results(performance, trades, equity_curve, params=current_params, filename_prefix=f'backtest_{symbol}_{timeframe_str}')
        except Exception as e:
             logger.error(f"Failed to save backtest results: {e}")

        # Plot the equity curve if snapshot_helper is available
        if self.snapshot_helper:
             try:
                  self.snapshot_helper.save_equity_curve_snapshot(equity_curve, filename_prefix=f'backtest_{symbol}_{timeframe_str}')
             except Exception as e:
                  logger.error(f"Failed to plot equity curve for backtest command: {e}")

        # Send results via reply_func if available (e.g., Telegram)
        if reply_func:
            reply_func(formatted_results)
            
        logger.info("Backtest run completed.")

# Placeholder for MT5 data loading by time range (assuming it's not directly in MT5Client or needs wrapper)
# You would typically use mt5.copy_rates_from_time or mt5.copy_rates_range
# Added a placeholder method in MT5Client in the previous edit. 