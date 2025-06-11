import logging
import time
import pandas as pd
import ta
import threading
from typing import Dict, Any, Optional
import pytz
import MetaTrader5 as mt5

# Assume MT5Client is in src.mt5.mt5_client
from src.mt5.mt5_client import MT5Client, get_mt5_timeframe, get_mt5_order_type
# Assume TelegramHandler is in src.signals.telegram_handler
from src.signals.telegram_handler import TelegramHandler # Import for type hinting and sending messages
# Assume OnlineLearner is in src.ml.online_learner
from src.ml.online_learner import OnlineLearner # Import for type hinting and integration
# Assume SnapshotHelper is in src.logging.snapshot_helper
from src.logging.snapshot_helper import SnapshotHelper # Import for type hinting and plotting
from src.strategies.xauusd_strategy import XAUUSDStrategy

logger = logging.getLogger(__name__)

class LiveTrading(threading.Thread):
    def __init__(self, config: Dict[str, Any], mt5_client: MT5Client, telegram_handler: Optional[TelegramHandler] = None, online_learner: Optional[OnlineLearner] = None, snapshot_helper: Optional[SnapshotHelper] = None):
        super().__init__()
        self.config = config
        self.mt5_client = mt5_client
        self.telegram_handler = telegram_handler
        self.online_learner = online_learner
        self.snapshot_helper = snapshot_helper

        self.live_cfg = config.get('live', {}) # Use .get() with default empty dict
        self.indicators_cfg = config.get('indicators', {})
        self.order_cfg = config.get('order', {})
        self.risk_cfg = config.get('risk', {})
        self.mt5_cfg = config.get('mt5', {}) # Need MT5 config for symbols and potential timeframe

        self.symbols = self.mt5_cfg.get('symbols', [])
        self.timeframe_str = self.live_cfg.get('timeframe', "M30") # Get timeframe from live config
        self.mt5_timeframe = get_mt5_timeframe(self.timeframe_str)
        if self.mt5_timeframe is None:
             logger.critical(f"Invalid live trading timeframe specified in config: {self.timeframe_str}")
             # In a live bot, might want to exit or raise an error here.
             raise ValueError(f"Invalid MT5 timeframe: {self.timeframe_str}")

        self._stop_event = threading.Event()
        self.check_interval_seconds = self.live_cfg.get('check_interval_seconds', 60) # How often to check for new bars
        self.bars_to_fetch = self.live_cfg.get('bars_to_fetch', 200) # How many recent bars to fetch for analysis

        self.timezone = pytz.timezone(config.get('backtest', {}).get('timezone', 'UTC')) # Use timezone from backtest or default to UTC

        self._last_processed_time = {} # To keep track of the timestamp of the last processed bar per symbol
        for symbol in self.symbols:
            self._last_processed_time[symbol] = None # Initialize last processed time per symbol

        self.symbol = "XAUUSD"
        self.timeframe = "M30"
        self.strategy = XAUUSDStrategy()

        logger.info("LiveTrading initialized.")

    def run(self):
        logger.info("LiveTrading thread started.")
        # Minimal-Modus: Einfache Buy-Order für XAUUSD
        data = self.fetch_data(self.symbol)
        if data is not None and not data.empty:
            indicators = self.calculate_indicators(data)
            signal = self.evaluate_signals(indicators)
            if signal == "buy":
                self.execute_trade(self.symbol, signal, indicators.iloc[-1])
                logger.info("Live-Trading-Modul erfolgreich initialisiert und Buy-Order abgesetzt.")
        else:
            logger.warning("Keine Daten für XAUUSD erhalten.")
        self._stop_event.set()

    def stop(self):
        """Signals the live trading thread to stop."""
        self._stop_event.set()
        logger.info("LiveTrading stop signal received.")

    def fetch_data(self, symbol: str) -> pd.DataFrame:
        # Holt die letzten 100 Bars für das Symbol
        return self.mt5_client.get_rates(symbol, self.timeframe, 100)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        # Nutzt die neue Strategie-Klasse
        return self.strategy.calculate_all(data)

    def evaluate_signals(self, data: pd.DataFrame) -> Optional[str]:
        # Minimal: Gibt immer "buy" zurück, wenn Daten vorhanden
        if not data.empty:
            return "buy"
        return None

    def execute_trade(self, symbol: str, signal: str, latest_bar: pd.Series):
        # ATR-basiertes TP/SL
        atr = latest_bar.get('atr_14', None)
        if atr is None:
            logger.warning(f"ATR(14) nicht verfügbar für {symbol}. Trade nicht ausgeführt.")
            return
        entry = latest_bar['close']
        if signal == "buy":
            sl = entry - 1 * atr
            tp = entry + 2 * atr
        else:
            sl = entry + 1 * atr
            tp = entry - 2 * atr
        logger.info(f"Simulierte Order: {signal.upper()} {symbol} @ {entry} | TP={tp} (2xATR), SL={sl} (1xATR), Kategorie=GOOD, ATR={atr}")

    def fetch_latest_data(self, symbol: str, n_bars: int = 200) -> pd.DataFrame | None:
        """Fetches the latest N bars for a symbol and timeframe using MT5Client."""
        if self.mt5_timeframe is None:
             logger.error("Cannot fetch data: Invalid MT5 timeframe.")
             return None

        # Use get_rates from MT5Client. It should return a pandas DataFrame with datetime index.
        data = self.mt5_client.get_rates(symbol, self.mt5_timeframe, n_bars)

        if data is None or data.empty:
             logger.warning(f"Could not fetch latest data for {symbol}.")
             return None

        # MT5Client.get_rates should handle timezone conversion to the configured timezone
        # Verify that the returned data is timezone-aware and in the correct zone if needed
        # if data.index.tzinfo is None or str(data.index.tzinfo) != str(self.timezone):
        #      logger.warning(f"Data for {symbol} has unexpected timezone info: {data.index.tzinfo}. Expected: {self.timezone}")
        #      # Attempt conversion if needed, or log error
        #      try:
        #           # Assuming data is naive UTC if tzinfo is None
        #           data.index = data.index.tz_localize('UTC').tz_convert(self.timezone)
        #      except Exception as e:
        #           logger.error(f"Failed to convert data timezone for {symbol}: {e}")

        # Ensure data is sorted by index (time)
        data = data.sort_index()

        return data

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculates technical indicators for the latest data."""
        if data is None or data.empty:
            return data

        # Copy data to avoid modifying the original DataFrame
        data = data.copy()

        # Add indicators using the `ta` library based on self.indicators_cfg
        # This logic is similar to the backtester, but applied to the latest data chunk

        # EMA
        for period in self.indicators_cfg.get('ema_periods', []):
            # Check if required column 'close' exists before calculating
            if 'close' not in data.columns:
                logger.warning(f"Cannot calculate EMA_{period}: 'close' column missing.")
                continue
            data[f'EMA_{period}'] = ta.trend.ema_indicator(data['close'], window=period)

        # RSI
        if 'close' in data.columns:
             rsi_period = self.indicators_cfg.get('rsi_period', 14)
             data['RSI'] = ta.momentum.rsi(data['close'], window=rsi_period)
        else:
             logger.warning("Cannot calculate RSI: 'close' column missing.")

        # MACD
        if 'close' in data.columns:
             macd_params = self.indicators_cfg.get('macd', {})
             data['MACD_line'] = ta.trend.macd_line(data['close'], window_fast=macd_params.get('fast', 12), window_slow=macd_params.get('slow', 26))
             data['MACD_signal'] = ta.trend.macd_signal(data['close'], window_fast=macd_params.get('fast', 12), window_slow=macd_params.get('slow', 26), window=macd_params.get('signal', 9))
             data['MACD_hist'] = data['MACD_line'] - data['MACD_signal']
        else:
             logger.warning("Cannot calculate MACD: 'close' column missing.")

        # Bollinger Bands
        if 'close' in data.columns:
             bollinger_params = self.indicators_cfg.get('bollinger', {})
             indicator_bb = ta.volatility.BollingerBands(
                 close=data['close'],
                 window=bollinger_params.get('period', 20),
                 window_dev=bollinger_params.get('sigma', 2.0)
             )
             data['BB_upper'] = indicator_bb.bollinger_hband()
             data['BB_lower'] = indicator_bb.bollinger_lband()
        else:
             logger.warning("Cannot calculate Bollinger Bands: 'close' column missing.")

        # ATR
        if all(col in data.columns for col in ['high', 'low', 'close']):
             atr_period = self.indicators_cfg.get('atr_period', 14)
             data['ATR'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'], window=atr_period)
        else:
             logger.warning("Cannot calculate ATR: Missing high, low, or close columns.")

        # Drop rows with NaN values created by indicator calculations (usually the first few rows)
        initial_rows = len(data)
        data = data.dropna()
        if len(data) < initial_rows:
             logger.debug(f"Dropped {initial_rows - len(data)} rows with NaNs after indicator calculation.")

        return data

    def evaluate_trade_rules(self, data: pd.DataFrame) -> str | None:
        """Evaluates the trading strategy rules on the latest completed bar."""
        if data.empty or len(data) < 2: # Need at least 2 bars to ensure the last one is completed
            return None

        # Get the latest completed bar (last row in the DataFrame)
        latest_bar = data.iloc[-1]

        # Implement the trading rules based on EMA, RSI, MACD, Bollinger
        # Rules:
        # Long: EMA_short > EMA_long ∧ RSI ∈ (30, 70) ∧ MACD-Hist > 0 ∧ Price touches lower Bollinger (low <= BB_lower)
        # Short: EMA_short < EMA_long ∧ RSI ∈ (30, 70) ∧ MACD-Hist < 0 ∧ Price touches upper Bollinger (high >= BB_upper)

        # Assuming EMA periods are in self.indicators_cfg['ema_periods'] and sorted (e.g., [8, 21])
        ema_periods = self.indicators_cfg.get('ema_periods', [])
        if len(ema_periods) < 2:
             logger.error("Insufficient EMA periods configured for live strategy. Cannot evaluate rules.")
             return None

        ema_short_period = ema_periods[0] # Example: 8
        ema_long_period = ema_periods[1] # Example: 21

        # Check if required indicator columns exist in the latest bar
        required_cols = [
            f'EMA_{ema_short_period}',
            f'EMA_{ema_long_period}',
            'RSI',
            'MACD_hist',
            'BB_lower',
            'BB_upper',
            'close',
            'low',
            'high'
        ]

        if not all(col in latest_bar.index for col in required_cols):
             missing_cols = [col for col in required_cols if col not in latest_bar.index]
             logger.warning(f"Missing required indicator data in latest bar for signal evaluation: {missing_cols}. Skipping signal evaluation.")
             return None

        ema_short = latest_bar[f'EMA_{ema_short_period}']
        ema_long = latest_bar[f'EMA_{ema_long_period}']
        rsi = latest_bar['RSI']
        macd_hist = latest_bar['MACD_hist']
        close_price = latest_bar['close']
        low_price = latest_bar['low']
        high_price = latest_bar['high']
        bb_lower = latest_bar['BB_lower']
        bb_upper = latest_bar['BB_upper']

        # Check for Long signal
        long_condition_ema = ema_short > ema_long
        long_condition_rsi = 30 < rsi < 70
        long_condition_macd = macd_hist > 0
        long_condition_bb = low_price <= bb_lower # Price touches lower Bollinger with low

        if long_condition_ema and long_condition_rsi and long_condition_macd and long_condition_bb:
            logger.info(f"Long signal detected at {latest_bar.name} for {data.iloc[-1]['symbol']}.") # Use symbol from data if available
            return "buy"

        # Check for Short signal
        short_condition_ema = ema_short < ema_long
        short_condition_rsi = 30 < rsi < 70
        short_condition_macd = macd_hist < 0
        short_condition_bb = high_price >= bb_upper # Price touches upper Bollinger with high

        if short_condition_ema and short_condition_rsi and short_condition_macd and short_condition_bb:
            logger.info(f"Short signal detected at {latest_bar.name} for {data.iloc[-1]['symbol']}.")
            return "sell"

        # logger.debug(f"No trade signal at {latest_bar.name} for {data.iloc[-1]['symbol']}.") # Avoid excessive logging
        return None # No signal

    def execute_trade(self, symbol: str, signal: str, latest_bar: pd.Series):
        """Executes a trade based on the signal and risk management."""
        if signal is None:
            return

        # Check if there is already an open position for this symbol
        if self.mt5_client.get_open_positions(symbol):
            logger.info(f"Already have an open position for {symbol}. Skipping trade execution.")
            # TODO: Consider logic for adding to position or hedging if needed
            return

        order_type = get_mt5_order_type(signal)
        if order_type is None:
            logger.error(f"Invalid signal type for trade execution: {signal}")
            return

        # --- Risk Management & Position Sizing ---
        account_info = self.mt5_client.get_account_info()
        if account_info is None or account_info.balance is None:
             logger.error("Could not get account info for risk management. Cannot execute trade.")
             if self.telegram_handler:
                  self.telegram_handler.send_alert("🚨 **Handelsalarm:** Konnte Kontoinformationen für die Risikoanalyse nicht abrufen. Trade Execution abgebrochen.")
             return

        balance = account_info.balance
        if balance <= 0:
             logger.error("Account balance is zero or negative. Cannot execute trade.")
             if self.telegram_handler:
                  self.telegram_handler.send_alert("🚨 **Handelsalarm:** Kontostand ist Null oder negativ. Trade Execution abgebrochen.")
             return

        risk_per_trade_pct = self.risk_cfg.get('risk_per_trade_pct', 1.0) # Default to 1% risk
        risk_amount = balance * (risk_per_trade_pct / 100.0)
        if risk_amount <= 0:
             logger.warning("Calculated risk amount is zero or negative. Skipping trade execution.")
             return

        # Calculate Stop Loss price level based on ATR
        atr = latest_bar.get('ATR')
        if not atr:
             logger.warning(f"ATR not available in latest bar for {symbol}. Cannot calculate dynamic SL for risk management. Skipping trade.")
             if self.telegram_handler:
                  self.telegram_handler.send_alert(f"⚠️ **Handelswarnung:** ATR für {symbol} nicht verfügbar. Konnte dynamischen Stop-Loss für Risikoanalyse nicht berechnen. Trade Execution abgebrochen.")
             return

        sl_atr_multiplier = self.risk_cfg.get('sl_atr_multiplier', 1.0) # Default SL is 1 * ATR
        sl_distance_price_units = atr * sl_atr_multiplier # Distance in price units based on ATR

        # Need symbol info to calculate lot size and SL price
        symbol_info = self.mt5_client.get_symbol_info(symbol)
        if symbol_info is None:
             logger.error(f"Could not get symbol info for {symbol} for trade execution.")
             if self.telegram_handler:
                  self.telegram_handler.send_alert(f"🚨 **Handelsalarm:** Konnte Symbolinformationen für {symbol} nicht abrufen. Trade Execution abgebrochen.")
             return

        point = symbol_info.point
        trade_contract_size = symbol_info.trade_contract_size

        # Calculate potential loss per lot if SL is hit
        # Assumes entry price is close to the current price (latest_bar['close'])
        # In reality, execution price can differ due to slippage/spread.
        # Using latest_bar['close'] as a proxy for calculation.
        price = latest_bar['close']

        if signal == "buy":
            stop_loss_price = price - sl_distance_price_units
            # Ensure SL price is valid (e.g., not negative)
            if stop_loss_price <= 0:
                 logger.warning(f"Calculated Stop Loss price for BUY {symbol} is zero or negative ({stop_loss_price:.5f}). Adjusting to minimum.")
                 # Adjust SL to a minimal positive value or skip trade
                 # For simplicity, skipping trade if SL is invalid.
                 if self.telegram_handler:
                      self.telegram_handler.send_alert(f"⚠️ **Handelswarnung:** Ungültiger Stop-Loss für {symbol} BUY berechnet ({stop_loss_price:.5f}). Trade Execution abgebrochen.")
                 return
            # Loss per lot = (Entry Price - SL Price) * contract size
            # Using price as proxy for Entry Price
            loss_per_lot_at_sl = (price - stop_loss_price) * trade_contract_size

        elif signal == "sell":
            stop_loss_price = price + sl_distance_price_units
            # Ensure SL price is valid (e.g., not excessively high)
            # No simple lower bound check needed usually for sell SL above price
            # Loss per lot = (SL Price - Entry Price) * contract size
            # Using price as proxy for Entry Price
            loss_per_lot_at_sl = (stop_loss_price - price) * trade_contract_size

        else:
             logger.error(f"Unexpected signal ''{signal}'' in execute_trade.")
             return

        if loss_per_lot_at_sl <= 0:
             logger.warning(f"Calculated potential loss per lot is zero or negative for {symbol} {signal} ({loss_per_lot_at_sl:.2f}). This might indicate an issue with SL calculation or extremely low volatility. Skipping trade.")
             if self.telegram_handler:
                  self.telegram_handler.send_alert(f"⚠️ **Handelswarnung:** Berechneter Verlust pro Lot für {symbol} {signal} ist Null oder negativ. Trade Execution abgebrochen.")
             return

        # Calculate the maximum volume based on risk amount and loss per lot
        # Volume = Risk Amount / (Loss per lot / lot size)
        # lot size for XAUUSD is typically 100 (units) -> 1 standard lot = 100 units
        # Need to get lot size from symbol_info.volume_min, volume_max, volume_step
        # For simplicity, assuming standard lot calculation related to trade_contract_size
        # A standard lot is often 100,000 units for FX, 100 for XAUUSD
        # Assuming volume_step is the smallest tradable unit (e.g., 0.01 for 0.01 lots)
        min_volume = symbol_info.volume_min
        max_volume = symbol_info.volume_max
        volume_step = symbol_info.volume_step
        # Assumes loss_per_lot_at_sl is the loss per *unit* if trade_contract_size is 1
        # If trade_contract_size is, e.g., 100 for 1 standard lot, then loss_per_lot_at_sl is for 1 standard lot.
        # Let's assume loss_per_lot_at_sl is per unit of volume_step (e.g., per 0.01 lot)
        # Or more simply, calculate loss per *minimum tradable volume unit*
        # Loss per volume_step = abs(price - stop_loss_price) * (trade_contract_size / (1 / volume_step))
        # Let's stick to the formula: Volume = Risk Amount / Loss per unit. What is the unit?
        # Let's assume loss_per_lot_at_sl is the loss for 1 standard lot (volume = 1.0)
        # Correct calculation: Volume = Risk Amount / (Loss per 1 unit of volume)
        # Loss per 1 unit of volume = abs(price - stop_loss_price) * trade_contract_size
        loss_per_volume_unit = abs(price - stop_loss_price) * trade_contract_size

        if loss_per_volume_unit <= 0:
             # This check should ideally be redundant due to loss_per_lot_at_sl check, but for safety
             logger.warning(f"Calculated loss per volume unit is zero or negative for {symbol} {signal}. Skipping trade.")
             return

        calculated_volume = risk_amount / loss_per_volume_unit

        # Adjust calculated volume to be a multiple of volume_step and within min/max limits
        if calculated_volume < min_volume:
            logger.warning(f"Calculated volume ({calculated_volume:.2f}) is below minimum ({min_volume:.2f}). Setting to minimum.")
            volume_to_execute = min_volume
        elif calculated_volume > max_volume:
            logger.warning(f"Calculated volume ({calculated_volume:.2f}) is above maximum ({max_volume:.2f}). Setting to maximum.")
            volume_to_execute = max_volume
        else:
            # Round down to the nearest multiple of volume_step
            volume_to_execute = (calculated_volume // volume_step) * volume_step
            # Ensure it's at least min_volume after rounding
            volume_to_execute = max(volume_to_execute, min_volume)

        if volume_to_execute <= 0:
             logger.error(f"Calculated volume to execute is zero or negative after adjustments ({volume_to_execute:.2f}). Cannot execute trade.")
             if self.telegram_handler:
                  self.telegram_handler.send_alert(f"🚨 **Handelsalarm:** Berechnetes Handelsvolumen für {symbol} {signal} ist Null oder negativ. Trade Execution abgebrochen.")
             return

        # Calculate Take Profit price level (e.g., based on Risk-Reward ratio)
        rr_ratio = self.risk_cfg.get('risk_reward_ratio', 2.0) # Default RR ratio of 2.0
        if signal == "buy":
            # TP for a buy is above the entry price
            tp_distance_price_units = sl_distance_price_units * rr_ratio
            take_profit_price = price + tp_distance_price_units
        elif signal == "sell":
            # TP for a sell is below the entry price
            tp_distance_price_units = sl_distance_price_units * rr_ratio
            take_profit_price = price - tp_distance_price_units
        else:
             take_profit_price = None # Should not happen

        # --- Send the Order to MT5 ---
        request = {
            "action": mt5.TRADE_ACTION_DEAL, # Direct market execution
            "symbol": symbol,
            "volume": volume_to_execute,
            "type": order_type,
            "price": price, # Use current price for market order
            "deviation": self.order_cfg.get('deviation', 10), # Max price deviation in points
            "sl": stop_loss_price if stop_loss_price else 0.0, # Stop Loss level (0.0 if no SL)
            "tp": take_profit_price if take_profit_price else 0.0, # Take Profit level (0.0 if no TP)
            "magic": self.mt5_cfg.get('magic', 0), # Magic number to identify orders
            "comment": f"Bot {signal} trade",
            "type_time": mt5.ORDER_TIME_GTC, # Good Till Cancel
            "type_filling": mt5.ORDER_FILLING_IOC, # Immediate or Cancel
        }

        logger.info(f"Attempting to place {signal.upper()} order for {volume_to_execute:.2f} lots on {symbol} @ {price:.5f} (SL: {stop_loss_price:.5f}, TP: {take_profit_price:.5f})")

        result = self.mt5_client.send_order(request)

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Order successfully placed: {result}")
            if self.telegram_handler:
                 # Send Telegram confirmation with details
                 confirm_message = (
                     f"✅ **Handel Erfolgreich:** {signal.upper()} {volume_to_execute:.2f} lots {symbol}
"
                     f"Entry Price: {result.price:.5f}, SL: {result.request.sl:.5f}, TP: {result.request.tp:.5f}
"
                     f"Order Ticket: {result.order}, Deal Ticket: {result.deal}"
                 )
                 self.telegram_handler.send_message(confirm_message)

            # TODO: Store trade details for ML feedback later
            # self.trade_history = self.trade_history.append({...}, ignore_index=True)

        else:
            logger.error(f"Order placement failed: {result}")
            if self.telegram_handler:
                 # Send Telegram alert for failed order
                 error_message = (
                     f"❌ **Handel Fehlgeschlagen:** {signal.upper()} {volume_to_execute:.2f} lots {symbol}
"
                     f"Reason: {result.comment if result else 'Unknown error'}
"
                     f"Retcode: {result.retcode if result else 'N/A'}"
                 )
                 self.telegram_handler.send_alert(error_message)

        # TODO: Handle different filling types and partial fills

    # Helper method to get open positions (can be moved to MT5Client if needed)
    # def get_open_positions(self, symbol: str = None):
    #     """Retrieves open positions, optionally filtered by symbol."""
    #     positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
    #     if positions is None:
    #         logger.error(f"Error getting positions: {mt5.last_error()}")
    #         return []
    #     return list(positions) # Convert tuple to list


# TODO: Add methods for handling trade feedback for the OnlineLearner (e.g., on trade close)
# This is a significant piece of logic.

# TODO: Refine the run loop, potentially adding logic for:
# - Waiting for bar completion before processing (if check_interval is less than timeframe)
# - Handling disconnections and attempting to reconnect MT5
# - Synchronizing last processed time across bot restarts
# - More sophisticated logging and error reporting

# Note: The execute_trade method assumes mt5 is directly importable or accessible. 
# It should use the mt5_client instance to interact with the MT5 terminal.
# Corrected: Using self.mt5_client.send_order and self.mt5_client.get_symbol_info etc.
# Added mt5 import at the top for TRADE_ACTION_DEAL, ORDER_TIME_GTC, ORDER_FILLING_IOC, TRADE_RETCODE_DONE etc.