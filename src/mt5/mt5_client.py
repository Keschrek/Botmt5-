import logging
import time
import json
from typing import List, Dict, Any
import pandas as pd # Import pandas for data handling

# Import necessary components from MetaTrader5 library
try:
    import MetaTrader5 as mt5
except ImportError:
    logging.error("MetaTrader5 library not found. Please install it (`pip install MetaTrader5`).")
    # Depending on how critical MT5 is, you might want to exit or handle this.
    # For now, we'll just log and let subsequent calls fail.
    mt5 = None # Set to None if import fails

logger = logging.getLogger(__name__)

class MT5Client:
    def __init__(self, mt5_cfg: Dict[str, Any], telegram_bot=None):
        self.mt5_cfg = mt5_cfg
        self.telegram_bot = telegram_bot # Keep a reference to the Telegram bot instance
        self.is_connected = False
        self.symbols_mapping: Dict[str, str] = {}
        self.last_activity_timestamp = 0.0 # For watchdog

        if mt5:
            self._connect()
        else:
            logger.critical("MetaTrader5 library not available. Bot cannot connect to MT5.")


    def _connect(self):
        """Initializes and connects to MetaTrader 5."""
        if self.is_connected:
            logger.info("Already connected to MT5.")
            return

        if mt5 is None:
             logger.error("Cannot connect to MT5: Library not imported.")
             return

        logger.info(f"Attempting to connect to MT5 account {self.mt5_cfg['account']} on server {self.mt5_cfg['server']}...")

        # Set timeout for connection attempt (optional, default is usually sufficient)
        # mt5.set_timeout(10000) # Example: 10 seconds

        if mt5.initialize(
            login=self.mt5_cfg['account'],
            server=self.mt5_cfg['server'],
            password=self.mt5_cfg['password']
        ):
            self.is_connected = True
            self.last_activity_timestamp = time.time()
            logger.info("Successfully connected to MT5.")
            self._check_symbols()
        else:
            self.is_connected = False
            last_error = mt5.last_error()
            logger.error(f"Failed to connect to MT5. Error code: {last_error}")
            if self.telegram_bot:
                self.telegram_bot.send_message(self.mt5_cfg['chat_id'], f"MT5 Connection Failed: {last_error}")


    def _check_symbols(self):
        """Checks and maps symbols, especially for variations like 'm' suffixes."""
        if mt5 is None or not self.is_connected:
            logger.warning("Not connected to MT5. Cannot check symbols.")
            return

        logger.info("Checking configured symbols...")
        available_symbols = {s.name: s for s in mt5.symbols()} # Get all available symbols

        for symbol in self.mt5_cfg['symbols']:
            info = available_symbols.get(symbol)
            if info is None:
                logger.warning(f"Symbol '{symbol}' not found on MT5 server. Checking for alternatives...")
                # Simple check for 'm' suffix alternative
                if symbol.endswith('m'):
                    alt_symbol_name = symbol[:-1]
                else:
                    alt_symbol_name = symbol + 'm'

                alt_info = available_symbols.get(alt_symbol_name)
                if alt_info is not None:
                    logger.info(f"Alternative symbol '{alt_symbol_name}' found for '{symbol}'. Using alternative.")
                    self.symbols_mapping[symbol] = alt_symbol_name
                else:
                    logger.error(f"Neither '{symbol}' nor '{alt_symbol_name}' found. Cannot trade this symbol.")
                    self.symbols_mapping[symbol] = None # Mark as unavailable
                    if self.telegram_bot:
                        self.telegram_bot.send_message(self.mt5_cfg['chat_id'], f"MT5 Symbol Unavailable: Neither {symbol} nor {alt_symbol_name} found.")
            else:
                logger.info(f"Symbol '{symbol}' found.")
                self.symbols_mapping[symbol] = symbol # Use the original symbol if found

    def get_rates(self, symbol: str, timeframe: Any, n_bars: int) -> pd.DataFrame | None:
        """Retrieves historical data for a given symbol and timeframe."""
        if mt5 is None or not self.is_connected:
            logger.warning("Not connected to MT5. Cannot get rates.")
            return None

        mapped_symbol = self.symbols_mapping.get(symbol)
        if mapped_symbol is None:
            logger.warning(f"Cannot get rates: Symbol '{symbol}' not available or mapped.")
            return None

        # Convert timeframe string (e.g., "M30") to mt5.TIMEFRAME object
        # This requires mapping string inputs to the mt5.TIMEFRAME enum
        # Example mapping (needs to be comprehensive): {"M30": mt5.TIMEFRAME_M30, ...}
        # For now, assuming timeframe is already a valid mt5.TIMEFRAME object
        if not isinstance(timeframe, int): # Basic check if it looks like an mt5.TIMEFRAME enum value
             logger.error(f"Invalid timeframe format: {timeframe}. Expected mt5.TIMEFRAME object.")
             return None

        logger.info(f"Requesting {n_bars} bars of {mapped_symbol} on timeframe {timeframe}.")
        rates = mt5.copy_rates_from_pos(mapped_symbol, timeframe, 0, n_bars)

        if rates is None:
            last_error = mt5.last_error()
            logger.error(f"Failed to get rates for {mapped_symbol} on timeframe {timeframe}. Error code: {last_error}")
            # Consider adding reconnect logic here if needed or alerting
            if self.telegram_bot:
                self.telegram_bot.send_message(self.mt5_cfg['chat_id'], f"MT5 Get Rates Failed for {symbol}: {last_error}")
            return None
        else:
            logger.info(f"Successfully retrieved {len(rates)} bars for {mapped_symbol}.")
            self.last_activity_timestamp = time.time()
            # Convert rates to pandas DataFrame for easier handling
            rates_df = pd.DataFrame(rates)
            # Convert time in seconds to datetime
            rates_df['time'] = pd.to_datetime(rates_df['time'], unit='s')
            return rates_df

    def get_rates_range(self, symbol: str, timeframe: Any, start_time: pd.Timestamp, end_time: pd.Timestamp) -> pd.DataFrame | None:
        """Retrieves historical data for a given symbol and timeframe within a time range."""
        if mt5 is None or not self.is_connected:
            logger.warning("Not connected to MT5. Cannot get rates by range.")
            return None

        mapped_symbol = self.symbols_mapping.get(symbol)
        if mapped_symbol is None:
            logger.warning(f"Cannot get rates by range: Symbol '{symbol}' not available or mapped.")
            return None

        # Check if it looks like an mt5.TIMEFRAME enum value
        if not isinstance(timeframe, int):
             logger.error(f"Invalid timeframe format: {timeframe}. Expected mt5.TIMEFRAME object.")
             return None

        # Ensure start and end times are datetime objects (already handled by type hint, but good to be sure)
        # Convert pandas Timestamps to Python datetime objects for MT5
        start_dt = start_time.to_pydatetime()
        end_dt = end_time.to_pydatetime()

        logger.info(f"Requesting rates for {mapped_symbol} on timeframe {timeframe} from {start_dt} to {end_dt}.")

        # Use copy_rates_from_time_range
        rates = mt5.copy_rates_from_time_range(mapped_symbol, timeframe, start_dt, end_dt)

        if rates is None:
            last_error = mt5.last_error()
            logger.error(f"Failed to get rates by range for {mapped_symbol} on timeframe {timeframe}. Error code: {last_error}")
            # Consider adding reconnect logic here if needed or alerting
            if self.telegram_bot:
                self.telegram_bot.send_message(self.mt5_cfg['chat_id'], f"MT5 Get Rates Range Failed for {symbol}: {last_error}")
            return None
        elif len(rates) == 0:
             logger.warning(f"No rates retrieved for {mapped_symbol} on timeframe {timeframe} from {start_dt} to {end_dt}.")
             return pd.DataFrame() # Return empty DataFrame for no data
        else:
            logger.info(f"Successfully retrieved {len(rates)} bars for {mapped_symbol} by range.")
            self.last_activity_timestamp = time.time()
            # Convert rates to pandas DataFrame for easier handling
            rates_df = pd.DataFrame(rates)
            # Convert time in seconds to datetime and set as index
            rates_df['time'] = pd.to_datetime(rates_df['time'], unit='s')
            rates_df = rates_df.set_index('time')
            
            # Apply timezone info if MT5 returns naive timestamps
            # Assuming MT5 returns UTC naive timestamps, localize and convert to the configured timezone
            # You might need to adjust timezone handling based on your MT5 server's time.
            try:
                 # Assuming broker time is UTC and needs conversion to the timezone specified in config
                 # The Backtester and LiveTrading modules expect data in the configured timezone.
                 # The timezone is likely in self.config['backtest']['timezone'] or self.config['mt5']['timezone']
                 # Need to pass the config or timezone during MT5Client initialization if it's not just broker time (UTC)
                 # For now, assuming we convert from UTC naive to self.timezone (if available)
                 # self.timezone is not available in MT5Client init with current structure.
                 # Need to rethink timezone handling across modules or pass timezone to this method.

                 # Let's assume for now that MT5 returns UTC timestamps and we need to localize to UTC then convert.
                 # The target timezone should ideally come from config, but MT5Client currently only has mt5_cfg.
                 # Temporarily using UTC as target timezone for the DataFrame index.
                 rates_df.index = rates_df.index.tz_localize('UTC')
                 # If you need conversion to a specific timezone later, do it in the calling module (Backtester/LiveTrading)

            except Exception as e:
                 logger.warning(f"Could not localize timezone for data from range: {e}. Proceeding with naive timestamps.")

            return rates_df

    def get_tick(self, symbol: str) -> Dict[str, Any] | None:
         """Retrieves the latest tick data for a given symbol."""
         if mt5 is None or not self.is_connected:
             logger.warning("Not connected to MT5. Cannot get tick.")
             return None

         mapped_symbol = self.symbols_mapping.get(symbol)
         if mapped_symbol is None:
             logger.warning(f"Cannot get tick: Symbol '{symbol}' not available or mapped.")
             return None

         logger.info(f"Requesting latest tick for {mapped_symbol}.")
         tick = mt5.symbol_info_tick(mapped_symbol)

         if tick is None:
             last_error = mt5.last_error()
             logger.error(f"Failed to get tick for {mapped_symbol}. Error code: {last_error}")
             # Consider adding reconnect logic here or alerting
             if self.telegram_bot:
                 self.telegram_bot.send_message(self.mt5_cfg['chat_id'], f"MT5 Get Tick Failed for {symbol}: {last_error}")
             return None
         else:
             logger.info(f"Successfully retrieved tick for {mapped_symbol}.")
             self.last_activity_timestamp = time.time()
             # Convert tick object to a dictionary for easier handling
             tick_dict = {
                 'time': tick.time,
                 'bid': tick.bid,
                 'ask': tick.ask,
                 'last': tick.last,
                 'volume': tick.volume,
                 'time_msc': tick.time_msc,
                 'flags': tick.flags,
                 'volume_real': tick.volume_real
             }
             return tick_dict

    def send_order(self, symbol: str, order_type: int, volume: float, price: float, tp: float = 0.0, sl: float = 0.0, comment: str = ""):
        """Sends a trading order."""
        if mt5 is None or not self.is_connected:
            logger.warning("Not connected to MT5. Cannot send order.")
            if self.telegram_bot:
                 self.telegram_bot.send_message(self.mt5_cfg['chat_id'], f"MT5 not connected. Cannot send order for {symbol}.")
            return None

        mapped_symbol = self.symbols_mapping.get(symbol)
        if mapped_symbol is None:
            logger.warning(f"Cannot send order: Symbol '{symbol}' not available or mapped.")
            if self.telegram_bot:
                 self.telegram_bot.send_message(self.mt5_cfg['chat_id'], f"Cannot send order: Symbol {symbol} not available.")
            return None

        # Get symbol info to calculate TP/SL in price points if needed
        symbol_info = mt5.symbol_info(mapped_symbol)
        if symbol_info is None:
            last_error = mt5.last_error()
            logger.error(f"Failed to get symbol info for {mapped_symbol}. Error code: {last_error}. Cannot send order.")
            if self.telegram_bot:
                 self.telegram_bot.send_message(self.mt5_cfg['chat_id'], f"MT5 Error getting symbol info for {symbol}: {last_error}")
            return None

        point = symbol_info.point

        request = {
            "action": mt5.TRADE_ACTION_DEAL, # Or mt5.TRADE_ACTION_PENDING for limit/stop orders
            "symbol": mapped_symbol,
            "volume": float(volume),
            "type": order_type, # e.g., mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_SELL
            "price": float(price), # Entry price
            "deviation": self.mt5_cfg.get('slippage_pips', 0.5) * point, # Slippage in points
            "magic": 2023, # Arbitrary magic number
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC, # Good till cancel
            "type_filling": mt5.ORDER_FILLING_IOC, # Immediate or cancel
        }

        # Handle TP/SL in pips if configured
        # TP and SL are sent as price levels in MT5 order requests.
        # If config specifies TP/SL as pips, we need to calculate the price level.
        # The calculation depends on the order type (buy/sell) and the entry price.

        calculated_tp = 0.0
        calculated_sl = 0.0

        if tp > 0.0:
            if self.mt5_cfg.get('tp_as_pips', False):
                # Calculate TP price from pips
                tp_points = tp * mt5.symbol_info(mapped_symbol).point # Assuming tp is in pips
                if order_type == mt5.ORDER_TYPE_BUY:
                    calculated_tp = price + tp_points
                elif order_type == mt5.ORDER_TYPE_SELL:
                    calculated_tp = price - tp_points
                # Ensure TP is valid (e.g., above price for buy, below price for sell)
                if (order_type == mt5.ORDER_TYPE_BUY and calculated_tp <= price) or \
                   (order_type == mt5.ORDER_TYPE_SELL and calculated_tp >= price):
                    logger.warning(f"Calculated TP ({calculated_tp}) is not valid for order type {order_type}. Discarding TP.")
                    calculated_tp = 0.0 # Discard invalid TP
                else:
                     request["tp"] = calculated_tp
                     logger.debug(f"Calculated TP price from {tp} pips: {calculated_tp}")
            else:
                # TP is provided as a price level
                calculated_tp = tp
                request["tp"] = calculated_tp
                logger.debug(f"Using provided TP price level: {calculated_tp}")


        if sl > 0.0:
             if self.mt5_cfg.get('sl_as_pips', False):
                 # Calculate SL price from pips
                 sl_points = sl * mt5.symbol_info(mapped_symbol).point # Assuming sl is in pips
                 if order_type == mt5.ORDER_TYPE_BUY:
                     calculated_sl = price - sl_points
                 elif order_type == mt5.ORDER_TYPE_SELL:
                     calculated_sl = price + sl_points
                 # Ensure SL is valid (e.g., below price for buy, above price for sell)
                 if (order_type == mt5.ORDER_TYPE_BUY and calculated_sl >= price) or \
                    (order_type == mt5.ORDER_TYPE_SELL and calculated_sl <= price):
                     logger.warning(f"Calculated SL ({calculated_sl}) is not valid for order type {order_type}. Discarding SL.")
                     calculated_sl = 0.0 # Discard invalid SL
                 else:
                     request["sl"] = calculated_sl
                     logger.debug(f"Calculated SL price from {sl} pips: {calculated_sl}")
             else:
                 # SL is provided as a price level
                 calculated_sl = sl
                 request["sl"] = calculated_sl
                 logger.debug(f"Using provided SL price level: {calculated_sl}")

        logger.info(f"Sending order request: {request}")
        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            last_error = mt5.last_error()
            logger.error(f"Order failed. Return code: {result.retcode}. Description: {result.comment}. Error: {last_error}")
            # Log the full result for debugging
            logger.error(f"Order result: {result}")
            # Send Telegram notification on failure
            if self.telegram_bot:
                alert_message = f"‼️ MT5 Order Failed ({symbol} {volume} {order_type}): {result.comment} (Code: {result.retcode})"
                if last_error:
                    alert_message += f" Error: {last_error}"
                self.telegram_bot.send_message(self.mt5_cfg['chat_id'], alert_message)
            return None
        else:
            logger.info(f"Order successful. Ticket: {result.order}")
            self.last_activity_timestamp = time.time()
            # Optional: Send Telegram notification on success
            # if self.telegram_bot:
            #     success_message = f"✅ MT5 Order Successful ({symbol} {volume} {order_type}). Ticket: {result.order}"
            #     self.telegram_bot.send_message(self.mt5_cfg['chat_id'], success_message)
            return result

    def reconnect_loop(self):
        """Simple reconnect loop that blocks until connection is restored."""
        # This is a basic example. A real-world reconnect needs more sophistication.
        if mt5 is None:
            logger.warning("MT5 library not available. Reconnect loop aborted.")
            return

        while not self.is_connected:
            logger.warning("MT5 disconnected. Attempting to reconnect in 5 seconds...")
            time.sleep(5)
            self._connect()
            if self.is_connected:
                logger.info("MT5 reconnect successful.")
                if self.telegram_bot:
                    self.telegram_bot.send_message(self.mt5_cfg['chat_id'], "✅ MT5 Reconnect Successful.")
                break # Exit loop on success
            else:
                 # If still not connected, maybe send a Telegram alert (already in _connect failure)
                 pass # Continue trying to reconnect

    def shutdown(self):
        """Shuts down the MT5 connection."""
        if mt5 and self.is_connected:
            mt5.shutdown()
            self.is_connected = False
            logger.info("MT5 connection shut down.")
        elif mt5 and not self.is_connected:
             logger.info("MT5 connection already shut down.")
        else:
             logger.warning("MT5 library not available. Cannot shut down.")

# You might need to add functions here to map timeframe strings to mt5.TIMEFRAME enum values
def get_mt5_timeframe(timeframe_str: str):
    """Maps a timeframe string to the corresponding mt5.TIMEFRAME enum value."""
    # This is a partial mapping. Add all timeframes you need.
    timeframe_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1,
    }
    return timeframe_map.get(timeframe_str.upper())

# You might also need functions to map order type strings ("buy", "sell") to mt5.ORDER_TYPE enum values
def get_mt5_order_type(order_type_str: str):
    """Maps an order type string to the corresponding mt5.ORDER_TYPE enum value."""
    order_type_map = {
        "buy": mt5.ORDER_TYPE_BUY,
        "sell": mt5.ORDER_TYPE_SELL,
        # Add other order types like LIMIT, STOP if needed
    }
    return order_type_map.get(order_type_str.lower()) 