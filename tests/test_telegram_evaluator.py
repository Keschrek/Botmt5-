import logging
from src.signals.telegram_evaluator import TelegramEvaluator
from src.live.live_trading import LiveTrading
import pandas as pd

def test_medium_signal_as_good():
    # Beispiel-Daten
    atr = 10.0
    entry = 2000.0
    tp = entry + 2 * atr
    sl = entry - 1 * atr
    raw_message = f"BUY XAUUSD @ {entry} TP {tp} SL {sl}"
    indicator_data = {'ema': True, 'rsi': True, 'macd': True, 'bollinger': True, 'atr': True}
    knowledge_checks = {'market_structure': False, 'candlestick_patterns': True, 'support_resistance': True, 'psychological_levels': True}
    evaluator = TelegramEvaluator()
    rating = evaluator.evaluate(raw_message, indicator_data, knowledge_checks, atr=atr)
    assert rating.rating == "GOOD"
    # Simuliere execute_trade
    latest_bar = pd.Series({'close': entry, 'atr_14': atr})
    logging.basicConfig(level=logging.INFO)
    class DummyTrading:
        def execute_trade(self, symbol, signal, latest_bar):
            logger = logging.getLogger("test")
            atr = latest_bar['atr_14']
            entry = latest_bar['close']
            sl = entry - 1 * atr
            tp = entry + 2 * atr
            logger.info(f"Simulierte Order: {signal.upper()} {symbol} @ {entry} | TP={tp} (2xATR), SL={sl} (1xATR), Kategorie=GOOD, ATR={atr}")
    DummyTrading().execute_trade("XAUUSD", "buy", latest_bar)

if __name__ == "__main__":
    test_medium_signal_as_good() 