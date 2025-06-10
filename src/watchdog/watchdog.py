import logging
import time
import threading
from typing import Dict, Any

# Assume MT5Client is in src.mt5.mt5_client
from src.mt5.mt5_client import MT5Client
# Assume TelegramHandler is in src.signals.telegram_handler
from src.signals.telegram_handler import TelegramHandler

logger = logging.getLogger(__name__)

class Watchdog(threading.Thread):
    def __init__(self, watchdog_cfg: Dict[str, Any], telegram_handler: TelegramHandler, mt5_client: MT5Client):
        super().__init__()
        self.watchdog_cfg = watchdog_cfg
        self.telegram_handler = telegram_handler
        self.mt5_client = mt5_client

        self._stop_event = threading.Event()
        self.check_interval_seconds = self.watchdog_cfg.get('check_interval_seconds', 600) # Default 10 minutes
        self.balance_alert_threshold = self.watchdog_cfg.get('balance_alert_threshold', None) # Optional threshold
        self.last_activity_threshold_seconds = self.watchdog_cfg.get('last_activity_threshold_seconds', 3600) # Default 1 hour

        logger.info(f"Watchdog initialized with check interval: {self.check_interval_seconds} seconds.")

    def run(self):
        """Main watchdog loop running in a separate thread."""
        logger.info("Watchdog thread started.")
        while not self._stop_event.is_set():
            try:
                self.perform_checks()
                # Use wait() with the interval to allow stopping during sleep
                self._stop_event.wait(self.check_interval_seconds)
            except Exception as e:
                logger.error(f"An error occurred in watchdog loop: {e}")
                # Continue loop even if a check fails

        logger.info("Watchdog thread stopped.")

    def stop(self):
        """Signals the watchdog thread to stop."""
        self._stop_event.set()
        logger.info("Watchdog stop signal received.")

    def perform_checks(self):
        """Executes all configured checks."""
        logger.debug("Performing watchdog checks...")

        # Check MT5 Connection
        if self.watchdog_cfg.get('check_mt5_connection', True):
            self.check_mt5_connection()
            logger.info("MT5 connection check performed.")

        # Check Bot Process (Less relevant as a self-check, more for external monitoring)
        # if self.watchdog_cfg.get('check_bot_process', False):
        #     self.check_bot_process()

        # Check Last Activity Timestamp
        if self.watchdog_cfg.get('check_last_activity', True):
            self.check_last_activity()
            logger.info("Last activity check performed.")

        # Check Account Balance
        if self.watchdog_cfg.get('check_balance', True):
             self.check_account_balance()
             logger.info("Account balance check performed.")

        logger.debug("Watchdog checks finished.")


    def check_mt5_connection(self):
        """Checks the MT5 connection status."""
        try:
            if not self.mt5_client.is_connected():
                alert_message = "🚨 **Watchdog-Alarm:** MT5-Verbindung verloren!"
                logger.critical(alert_message)
                self.telegram_handler.send_alert(alert_message) # Assuming send_alert exists and uses configured chat_id
                # TODO: Implement reconnection logic here or trigger it in MT5Client
                # self.mt5_client.reconnect()
                logger.warning("MT5 connection lost. Reconnection logic needs implementation.")
            else:
                logger.debug("MT5 connection is active.")
        except Exception as e:
            logger.error(f"Error checking MT5 connection: {e}")
            alert_message = f"⚠️ **Watchdog-Fehler:** Fehler bei der MT5-Verbindungsprüfung: {e}"
            self.telegram_handler.send_alert(alert_message)

    # def check_bot_process(self):
    #     """Placeholder for checking if the bot process is running (e.g., via PID file)."""
    #     # This is better handled by an external process monitor (e.g., systemd, forever, pm2)
    #     pass # Not implementing self-check for simplicity


    def check_last_activity(self):
        """Checks if there has been recent activity (e.g., data updates, trades)."""
        # Assumes MT5Client has a get_last_activity_timestamp method returning a Unix timestamp or None
        try:
            last_activity_timestamp = self.mt5_client.get_last_activity_timestamp()
            
            if last_activity_timestamp:
                 time_since_last_activity = time.time() - last_activity_timestamp
                 if time_since_last_activity > self.last_activity_threshold_seconds:
                     alert_message = f"⏳ **Watchdog-Alarm:** Keine Aktivität seit {time_since_last_activity:.0f} Sekunden erkannt (Schwellenwert: {self.last_activity_threshold_seconds}s). Daten-Feed oder Bot-Logik könnte stecken geblieben sein."
                     logger.warning(alert_message)
                     self.telegram_handler.send_alert(alert_message)
                 else:
                     logger.debug("Recent activity detected.")
            else:
                 # This case might also indicate an issue, depending on expected behavior of get_last_activity_timestamp
                 logger.warning("Letzter Aktivitäts-Zeitstempel ist nicht verfügbar. Kann die Inaktivität nicht überprüfen.")

        except Exception as e:
             logger.error(f"Error checking last activity: {e}")
             alert_message = f"⚠️ **Watchdog-Fehler:** Fehler bei der Prüfung der letzten Aktivität: {e}"
             self.telegram_handler.send_alert(alert_message)

    def check_account_balance(self):
        """Checks the current account balance and alerts if below threshold."""
        if self.balance_alert_threshold is None:
            logger.debug("Balance alert threshold not set. Skipping balance check.")
            return
            
        try:
            # Assumes MT5Client.get_account_info() returns an object with a 'balance' attribute
            account_info = self.mt5_client.get_account_info()
            if account_info and hasattr(account_info, 'balance') and account_info.balance is not None:
                current_balance = account_info.balance
                if current_balance < self.balance_alert_threshold:
                    alert_message = f"📉 **Watchdog-Alarm:** Kontostand ({current_balance:.2f}) ist unter dem Warnschwellenwert ({self.balance_alert_threshold:.2f})."
                    logger.warning(alert_message)
                    self.telegram_handler.send_alert(alert_message)
                else:
                     logger.debug(f"Kontostand ({current_balance:.2f}) ist über dem Schwellenwert ({self.balance_alert_threshold:.2f}).")
            else:
                logger.warning("Konnte Kontoinformationen für die Saldenprüfung nicht abrufen.")
                # An alert for MT5 connection loss might cover this, but logging helps debug
                logger.debug("Skipping balance check due to inability to get account info or balance.")
                
        except Exception as e:
            logger.error(f"Error checking account balance: {e}")
            alert_message = f"⚠️ **Watchdog-Fehler:** Fehler bei der Kontostandsprüfung: {e}"
            self.telegram_handler.send_alert(alert_message)

# TODO: Integrate this Watchdog into run_bot.py and pass necessary instances. 