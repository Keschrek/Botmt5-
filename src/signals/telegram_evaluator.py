import re
from typing import Literal, Dict, Any

class SignalRating:
    def __init__(self, rating: Literal["GOOD", "MEDIUM", "BAD"], reason: str):
        self.rating = rating
        self.reason = reason

class TelegramEvaluator:
    """
    Bewertet Telegram-Signale.
    Input: raw_message (z.B. "BUY XAUUSD @ 2300.00 TP 2310.00 SL 2295.00")
    Output: SignalRating("GOOD"|"MEDIUM"|"BAD", reason)
    Beispiel:
        >>> TelegramEvaluator().evaluate("BUY XAUUSD @ 2300.00 TP 2310.00 SL 2295.00")
        SignalRating("GOOD", "Alle Checks bestanden.")
    """
    def evaluate(self, raw_message: str, indicator_data: Dict[str, Any] = None, knowledge_checks: Dict[str, bool] = None, atr: float = None) -> SignalRating:
        pattern = r"(BUY|SELL)\s+(\w+)\s+@\s*([\d.]+)\s+TP\s*([\d.]+)\s+SL\s*([\d.]+)"
        m = re.search(pattern, raw_message.upper())
        if not m:
            return SignalRating("BAD", "Parsing fehlgeschlagen.")
        direction, symbol, entry, tp, sl = m.groups()
        entry, tp, sl = float(entry), float(tp), float(sl)
        # ATR-basierte TP/SL-Prüfung (optional)
        if atr is not None:
            if direction == "BUY":
                expected_tp = entry + 2 * atr
                expected_sl = entry - 1 * atr
            else:
                expected_tp = entry - 2 * atr
                expected_sl = entry + 1 * atr
            # Toleranz für Rundungsfehler
            tol = 0.01
            if abs(tp - expected_tp) > tol or abs(sl - expected_sl) > tol:
                return SignalRating("BAD", f"TP/SL nicht ATR-basiert: erwartet TP={expected_tp}, SL={expected_sl}, erhalten TP={tp}, SL={sl}")
        # Dummy-Checks: In Realität hier Indikator- und Knowledge-Checks einbinden
        if indicator_data and not all(indicator_data.values()):
            return SignalRating("BAD", "Indikator-Check nicht bestanden.")
        if knowledge_checks and not all(knowledge_checks.values()):
            # Früher: return SignalRating("MEDIUM", ...)
            # Jetzt: MEDIUM wird zu GOOD
            return SignalRating("GOOD", "Mind. ein Wissens-Check nicht bestanden (als GOOD behandelt).")
        return SignalRating("GOOD", "Alle Checks bestanden.") 