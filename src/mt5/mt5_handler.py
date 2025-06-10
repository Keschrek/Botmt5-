import logging
import time
import json # Using json for logging errors as per spec example
from datetime import datetime
import pytz

# Import MetaTrader5 library
import MetaTrader5 as mt5

logger = logging.getLogger(__name__)

class MT5Handler:
    def __init__(self, mt5_cfg, order_cfg, telegram_send_message_func=None):
        self.mt5_cfg = mt5_cfg
        self.order_cfg = order_cfg
        self.telegram_send_message = telegram_send_message_func
        self.initialized = False

    def _send_telegram_error(self, message):
        """Sends an error message via the provided Telegram send function."""
        if self.telegram_send_message:
            try:
                # This assumes telegram_send_message is an awaitable function
                # In a threaded context, this needs careful handling (e.g., using asyncio.run_coroutine_threadsafe)
                # For simplicity in this placeholder, we'll log a warning if it's not awaitable or fails.
                # A proper implementation in a multi-threaded/async app requires more complex coordination.
                logger.warning("Attempting to send Telegram message. Note: Async operation in sync thread requires careful handling.")
                # Example (requires a running asyncio loop in the main thread/app): asyncio.run_coroutine_threadsafe(self.telegram_send_message(message), loop)
                # For now, just call it and hope it works or is handled externally
                self.telegram_send_message(message)
            except Exception as e:
                logger.error(f"Failed to send Telegram error message: {e}")

    def initialize_mt5(self):
        """Initializes the MT5 connection with reconnect logic."""
        while True:
            logger.info("Attempting to initialize MetaTrader5...")
            try:
                # Pass timeout for connection
                if mt5.initialize(
                    login=self.mt5_cfg['account'],
                    server=self.mt5_cfg['server'],
                    password=self.mt5_cfg['password'],
                    timeout=10 # Connection timeout in seconds
                ):
                    logger.info("MetaTrader5 initialized successfully.")
                    self.initialized = True
                    return True
                else:
                    # Log MT5 error and last error code
                    last_error = mt5.last_error()
                    error_details = {"error_code": last_error[0], "error_description": last_error[1], "step": "mt5_initialize"}
                    logger.error(f"MT5 initialization failed: {last_error[0]} - {last_error[1]}")
                    self._send_telegram_error(f"MT5-Error: Initialization failed - {last_error[0]}")

            except Exception as e:
                error_details = {"error_description": str(e), "step": "mt5_initialize_exception"}
                logger.error(f"An unexpected error occurred during MT5 initialization attempt: {e}")
                self._send_telegram_error(f"MT5-Error: Initialization exception - {e}")

            logger.info("Retrying MT5 initialization in 5 seconds...")
            time.sleep(5)
            # Continue the while loop to retry

    def shutdown_mt5(self):
        """Shuts down the MT5 connection."""
        if self.initialized:
            logger.info("Shutting down MetaTrader5 connection...")
            mt5.shutdown()
            self.initialized = False
            logger.info("MetaTrader5 connection shut down.")

    def get_symbol_info(self, symbol):
        """Gets symbol information, handling potential alternative names."""
        # Ensure MT5 is initialized before calling any MT5 function
        if not self.initialized:
            if not self.initialize_mt5():
                logger.error(f"MT5 not initialized, cannot get info for {symbol}.")
                return None

        try:
            info = mt5.symbol_info(symbol)
            if info is None:
                logger.warning(f"Symbol info not found for {symbol}.")
                # TODO: Implement logic to check for alternative names if necessary
                # This might involve iterating through available symbols or having a predefined map
                return None

            logger.debug(f"Successfully retrieved info for symbol {symbol}.")
            return info

        except Exception as e:
            error_details = {"symbol": symbol, "error_description": str(e), "step": "get_symbol_info"}
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            self._send_telegram_error(f"MT5-Error: Failed to get symbol info for {symbol} - {e}")
            # Consider re-initializing MT5 if a call fails
            self.initialized = False
            self.initialize_mt5()
            return None

    def get_historical_data(self, symbol, timeframe, start_pos, count):
        """Retrieves historical data from MT5."""
         # Ensure MT5 is initialized
        if not self.initialized:
            if not self.initialize_mt5():
                logger.error(f"MT5 not initialized, cannot get historical data for {symbol}.")
                return None

        try:
            # Example for getting data (adjust timeframe and start_pos/count as needed)
            # Need to map timeframe string (e.g., 'M30') to mt5.TIMEFRAME_...
            # Assuming timeframe is an mt5.TIMEFRAME_* constant
            rates = mt5.copy_rates_from_pos(symbol, timeframe, start_pos, count)
            if rates is None:
                 last_error = mt5.last_error()
                 logger.warning(f"No historical data for {symbol} ({timeframe}) from {start_pos} count {count}. Error: {last_error[0]} - {last_error[1]}")
                 return None

            logger.debug(f"Successfully retrieved {len(rates)} bars for {symbol} ({timeframe}).")
            return rates

        except Exception as e:
            error_details = {"symbol": symbol, "timeframe": timeframe, "start_pos": start_pos, "count": count, "error_description": str(e), "step": "get_historical_data"}
            logger.error(f"Error getting historical data for {symbol}: {e}")
            self._send_telegram_error(f"MT5-Error: Failed to get historical data for {symbol} - {e}")
            # Consider re-initializing MT5
            self.initialized = False
            self.initialize_mt5()
            return None

    def place_order(self, symbol, order_type, volume, price, tp, sl, deviation, magic, comment):
        """Places a trade order, considering TP/SL in pips."""
         # Ensure MT5 is initialized
        if not self.initialized:
            if not self.initialize_mt5():
                logger.error(f"MT5 not initialized, cannot place order for {symbol}.")
                return None

        # TODO: Implement TP/SL calculation based on self.order_cfg['tp_as_pips'] and self.order_cfg['sl_as_pips']
        # This requires getting current price and symbol information (point size)
        # For now, assume TP/SL are absolute prices or calculated elsewhere

        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': symbol,
            'volume': float(volume), # Ensure volume is float
            'type': order_type, # e.g., mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_SELL
            'price': float(price), # Ensure price is float
            'deviation': int(deviation), # Ensure deviation is int
            'tp': float(tp) if tp is not None else 0.0,
            'sl': float(sl) if sl is not None else 0.0,
            'magic': int(magic), # Ensure magic is int
            'comment': str(comment),
            'type_time': mt5.ORDER_TIME_GTC, # Good till cancelled
            'type_filling': mt5.ORDER_FILLING_IOC, # Immediate or Cancel
        }

        try:
            result = mt5.order_send(request)

            if result is None:
                 last_error = mt5.last_error()
                 logger.error(f"Order send failed for {symbol}. Error: {last_error[0]} - {last_error[1]}")
                 self._send_telegram_error(f"MT5-Error: Order failed for {symbol} - {last_error[0]}")
                 # Consider re-initializing MT5
                 self.initialized = False
                 self.initialize_mt5()
                 return None

            # Log the result
            logger.info(f"Order sent: {symbol} {order_type} {volume}. Result: {result.retcode}")
            if result.retcode != mt5.TRADE_RETCODE_SUCCEEDED:
                logger.warning(f"Order was not fully successful. Result: {result}")
                self._send_telegram_error(f"MT5-Warning: Order not fully successful for {symbol}. Result: {result.retcode}")

            return result

        except Exception as e:
            error_details = {"symbol": symbol, "order_type": order_type, "volume": volume, "error_description": str(e), "step": "place_order"}
            logger.error(f"An unexpected error occurred placing order for {symbol}: {e}")
            self._send_telegram_error(f"MT5-Error: Exception placing order for {symbol} - {e}")
            # Consider re-initializing MT5
            self.initialized = False
            self.initialize_mt5()
            return None

    def run_polling(self):
        """Main polling loop for MT5 data or state checks."""
        # Ensure MT5 is initialized before starting the loop
        if not self.initialize_mt5():
            logger.critical("Failed to initialize MT5 connection. Cannot start polling loop.")
            return

        # This is a placeholder loop. 
        # In a real bot, this loop would periodically check for new data, 
        # process signals, update indicators, check open positions, etc.
        # It should also include the reconnect logic if the connection drops 
        # *after* initial successful initialization.

        logger.info("Starting MT5 polling loop (placeholder)...")
        while self.initialized: # Loop while MT5 is considered initialized
            try:
                # TODO: Implement actual MT5 polling logic here
                # Example: Get latest tick data, check connection status, etc.

                # Check connection status periodically
                if not mt5.terminal_info():
                    logger.warning("MT5 terminal not connected. Attempting to re-initialize...")
                    self.initialized = False # Mark as not initialized to trigger re-initialization
                    self.initialize_mt5() # This call has its own retry logic
                    # If initialize_mt5 fails after retries, it will return False
                    if not self.initialized:
                        logger.critical("Failed to re-initialize MT5 after connection loss. Stopping polling.")
                        break # Exit the polling loop

                # Simulate some work or data fetching
                # data = self.get_historical_data('XAUUSD', mt5.TIMEFRAME_M30, 0, 10)
                # if data:
                #     logger.debug(f"Fetched data count: {len(data)}")

                time.sleep(1) # Poll interval (adjust as needed)

            except Exception as e:
                error_details = {"error_description": str(e), "step": "mt5_polling_loop"}
                logger.error(f"An error occurred in MT5 polling loop: {e}")
                self._send_telegram_error(f"MT5-Error: Exception in polling loop - {e}")
                # Decide whether to attempt re-initialization or exit based on error type
                # For now, assume transient error and just log, loop continues
                time.sleep(5) # Wait before continuing the loop after an error

        logger.info("MT5 polling loop stopped.")

# Helper function to create and run the handler (can be called from run_bot.py)
def start_mt5_handler(mt5_cfg, order_cfg, telegram_send_message_func=None):
    handler = MT5Handler(mt5_cfg, order_cfg, telegram_send_message_func)
    # Note: The run_polling() method is blocking.
    # In run_bot.py, this should be run in a separate thread.
    handler.run_polling() 