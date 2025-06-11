import pandas as pd
import ta

class XAUUSDStrategy:
    """
    Strategie-Modul für XAUUSD. Berechnet EMA(8,21), RSI(14), MACD(12,26,9), Bollinger(20,2), ATR(14).
    Input: DataFrame mit Spalten ['open','high','low','close','volume']
    Output: DataFrame mit Indikator-Spalten
    """
    def calculate_ema(self, data: pd.DataFrame, period: int) -> pd.Series:
        return ta.trend.ema_indicator(data['close'], window=period)

    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        return ta.momentum.rsi(data['close'], window=period)

    def calculate_macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        macd_line = ta.trend.macd_line(data['close'], window_fast=fast, window_slow=slow)
        macd_signal = ta.trend.macd_signal(data['close'], window_fast=fast, window_slow=slow, window=signal)
        macd_hist = macd_line - macd_signal
        return pd.DataFrame({'macd_line': macd_line, 'macd_signal': macd_signal, 'macd_hist': macd_hist})

    def calculate_bollinger(self, data: pd.DataFrame, period: int = 20, sigma: float = 2.0) -> pd.DataFrame:
        bb = ta.volatility.BollingerBands(data['close'], window=period, window_dev=sigma)
        return pd.DataFrame({'bb_upper': bb.bollinger_hband(), 'bb_lower': bb.bollinger_lband()})

    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        return ta.volatility.average_true_range(data['high'], data['low'], data['close'], window=period)

    def calculate_all(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data['ema_8'] = self.calculate_ema(data, 8)
        data['ema_21'] = self.calculate_ema(data, 21)
        data['rsi_14'] = self.calculate_rsi(data)
        macd = self.calculate_macd(data)
        data = pd.concat([data, macd], axis=1)
        boll = self.calculate_bollinger(data)
        data = pd.concat([data, boll], axis=1)
        data['atr_14'] = self.calculate_atr(data)
        return data.dropna() 