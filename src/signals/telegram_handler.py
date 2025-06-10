import logging
import json
from pathlib import Path
import jsonschema
from typing import Dict, Any, List

# Import necessary classes from python-telegram-bot
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Assume Backtester is in src.backtests.backtester
# from src.backtests.backtester import Backtester # Uncomment and import when Backtester is passed

logger = logging.getLogger(__name__)

# Placeholder for the signal schema
SIGNAL_SCHEMA = {}

# Load signal schema on startup
try:
    schema_path = Path(__file__).parent / "schema.json"
    with open(schema_path, 'r') as f:
        SIGNAL_SCHEMA = json.load(f)
    logger.info("Signal schema loaded successfully.")
except FileNotFoundError:
    logger.error("Signal schema file not found!")
    # Handle error appropriately, maybe exit or disable signal processing
except json.JSONDecodeError:
    logger.error("Error decoding signal schema JSON.")
    # Handle error appropriately

class TelegramHandler:
    # Update __init__ to accept backtester instance
    def __init__(self, telegram_cfg: Dict[str, Any], allowed_symbols: List[str], backtester=None):
        self.telegram_cfg = telegram_cfg
        self.allowed_symbols = allowed_symbols
        self.backtester = backtester # Store backtester instance
        self.application = Application.builder().token(self.telegram_cfg['api_hash']).build()
        
        # Store chat_id for easy access and validation
        self.allowed_chat_id = self.telegram_cfg.get('chat_id')
        if self.allowed_chat_id is None:
             logger.critical("Telegram chat_id not configured. Bot will not respond to any messages.")
             # Decide how to handle this critical error: exit, log only, etc.

        # Add handlers
        # Process text messages that are not commands as potential signals
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_signal))
        
        # Add command handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("test", self.test_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
        # Add other command handlers here

        logger.info("TelegramHandler initialized with command handlers.")

    async def is_authorized(self, update: Update) -> bool:
        """Checks if the message is from the configured chat ID."""
        if self.allowed_chat_id is None:
             logger.warning("No authorized chat_id configured. Rejecting all messages.")
             # Maybe send a message to the update.effective_chat.id indicating configuration error?
             # await update.effective_chat.send_message("Bot configuration error: Authorized chat ID not set.")
             return False

        if update.effective_chat.id != self.allowed_chat_id:
            logger.warning(f"Received message from unauthorized chat ID: {update.effective_chat.id}")
            # Optionally send a message back to unauthorized users
            # await update.message.reply_text("You are not authorized to command this bot.")
            return False
        return True

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handles the /start command."""
        if not await self.is_authorized(update):
            return # Not authorized

        user = update.effective_user
        logger.info(f"Received /start command from user {user.id} in chat {update.effective_chat.id}")
        await update.message.reply_html(
            f"Hi {user.mention_html()}! I am your trading bot. Send me commands like /status or /test."
        )

    async def test_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handles the /test command to trigger a backtest."""
        if not await self.is_authorized(update):
            return # Not authorized

        logger.info(f"Received /test command from chat {update.effective_chat.id}")
        
        # Expected format: /test SYMBOL [timeframe]
        args = context.args
        if not args:
            await update.message.reply_text("Usage: /test SYMBOL [timeframe (default M30)]")
            return
            
        symbol = args[0].upper() # Get symbol from args
        timeframe_str = args[1] if len(args) > 1 else "M30" # Get timeframe or use default
        
        if symbol not in self.allowed_symbols:
             logger.warning(f"Backtest requested for disallowed symbol: {symbol}")
             await update.message.reply_text(f"Disallowed symbol for backtest: {symbol}.")
             return

        if self.backtester is None:
             logger.error("Backtester is not initialized. Cannot run backtest from Telegram.")
             await update.message.reply_text("Backtesting functionality is not available.")
             return
             
        logger.info(f"Triggering backtest for {symbol} ({timeframe_str}) via Telegram.")
        await update.message.reply_text(f"Starting backtest for {symbol} ({timeframe_str}). Please wait...")
        
        # TODO: Run the backtest in a non-blocking way (e.g., in a thread) 
        # and send the results back to the user when finished.
        # For now, calling the run_backtest_command method which is designed to be triggered.
        # This method itself might need to handle threading or report progress.
        # backtest_results = self.backtester.run_backtest_command(symbol, timeframe_str)
        # TODO: Format and send backtest_results via Telegram
        logger.warning("Backtest execution via Telegram is a placeholder.")
        await update.message.reply_text("Backtest initiated (placeholder).")
        # Example of how to send results (assuming backtest_results is a dict):
        # results_message = "Backtest Results:\n" + "\n".join([f"{k}: {v:.2f}" for k, v in backtest_results.items()])
        # await update.message.reply_text(results_message)


    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handles the /status command to report bot status."""
        if not await self.is_authorized(update):
            return # Not authorized
            
        logger.info(f"Received /status command from chat {update.effective_chat.id}")
        
        # TODO: Gather relevant status information (MT5 connection, active trades, errors, last activity, etc.)
        status_message = "🤖 Bot Status:\n"
        status_message += f"- MT5 Connection: {'Active' if self.mt5_client and self.mt5_client.is_connected else 'Inactive'}\n" # Assuming mt5_client instance is available
        # TODO: Add more status details
        status_message += "- Active Trades: N/A (TODO)\n"
        status_message += "- Last Activity: N/A (TODO)\n"
        status_message += "- Watchdog: Running (TODO)\n"

        await update.message.reply_text(status_message)


    async def handle_signal(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handles incoming text messages as potential trading signals."""
        # Ensure the message is from the configured chat ID and is not a command
        # The filter already checks for ~filters.COMMAND
        if not await self.is_authorized(update):
            return # Not authorized

        signal_payload = update.message.text
        logger.info(f"Received potential signal: {signal_payload}")

        # Attempt to parse as JSON
        try:
            signal_data = json.loads(signal_payload)
        except json.JSONDecodeError:
            logger.error(f"Invalid signal format: Not valid JSON - {signal_payload}")
            await update.message.reply_text("Invalid signal format: Not valid JSON.")
            return

        # Validate against schema
        # Ensure SIGNAL_SCHEMA was loaded
        if not SIGNAL_SCHEMA:
             logger.error("Signal schema not loaded. Cannot validate signal.")
             await update.message.reply_text("Bot error: Signal schema not loaded.")
             return

        try:
            jsonschema.validate(signal_data, SIGNAL_SCHEMA)
            logger.info("Signal payload validated against schema.")
        except jsonschema.ValidationError as e:
            logger.error(f"Invalid signal format: Schema validation failed - {e.message}")
            await update.message.reply_text(f"Invalid signal format: Schema validation failed - {e.message}")
            return

        # Filter by allowed symbols
        symbol = signal_data.get('symbol')
        if symbol and symbol not in self.allowed_symbols:
            logger.warning(f"Received signal for disallowed symbol: {symbol}")
            await update.message.reply_text(f"Signal for disallowed symbol: {symbol}.")
            return

        logger.info(f"Valid signal received for {symbol}: {signal_data}")

        # TODO: Process the valid signal (e.g., send to LiveTrading module to execute a trade)
        # This would involve calling a method on the LiveTrading instance, passing the signal_data
        # Assuming LiveTrading instance is available (needs to be passed to TelegramHandler)
        # For now, just reply received.
        await update.message.reply_text(f"Signal received and validated for {symbol}. (Processing TODO)")

    def run(self):
        logger.info("Starting Telegram handler polling...")
        # This method is intended to be run in a separate thread
        # It will block until polling is stopped
        self.application.run_polling()

    def stop(self):
        logger.info("Stopping Telegram handler polling...")
        # This method should be called from the main thread during shutdown
        # It signals the polling loop to stop gracefully
        self.application.stop_polling()

# This function might be called from run_bot.py
# Update start_telegram_handler to accept backtester (and potentially live_trading)
def start_telegram_handler(telegram_cfg: Dict[str, Any], allowed_symbols: List[str], backtester=None):
    # Pass backtester instance to the handler
    handler = TelegramHandler(telegram_cfg, allowed_symbols, backtester=backtester)
    # The run() method is blocking, so the caller (run_bot.py) must run it in a thread.
    # No need to explicitly create a thread here, run_bot.py already does this.
    # handler.run() # Don't call run() here if the caller is threading it.
    return handler # Return the handler instance for use in run_bot.py (e.g., for shutdown)

# Example of how to use in run_bot.py (assuming config and symbols are loaded):
# telegram_handler_instance = start_telegram_handler(cfg['telegram'], cfg['mt5']['symbols'], backtester_instance)
# telegram_thread = threading.Thread(target=telegram_handler_instance.run, daemon=True)
# telegram_thread.start() 