# Trading-Grundlagen für den XAUUSD-Bot

Dieses Dokument bietet einen Überblick über grundlegende Konzepte des Tradings, die für das Verständnis und die Weiterentwicklung des MT5/Telegram Trading Bots wichtig sind.

## 1. Marktmechanik
- Bid/Ask, Spread und Liquidität  
- Hebelwirkung, Margin-Anforderungen  
- Lot-Größen und Kontraktwerte

## 2. Order-Typen  
- Market-, Limit-, Stop-Orders  
- Take-Profit (TP) & Stop-Loss (SL)  
- Einfügen von TP/SL als Pips vs. absolute Kurse

## 3. XAUUSD-Spezifika  
- Handelszeiten für Gold (COMEX, MT5-Server-Zeiten)  
- Typische Volatilität & Marktstruktur  
- Kontraktgröße und Währungseinheiten

## 4. Technische Indikatoren  
### 4.1 Exponentielle Gleitende Durchschnitte (EMA)  
- Formel & Glättungsfaktor  
- Typische Perioden (z. B. 8, 21)  
### 4.2 Relative Strength Index (RSI)  
- Berechnung & Interpretation  
- Overbought/Oversold-Level  
### 4.3 MACD  
- Linien & Histogramm  
- Signalgenerierung  
### 4.4 Bollinger-Bänder  
- Standardabweichung & Mittellinie  
- Bandberührung als Trade-Signal  
### 4.5 Average True Range (ATR)  
- Volatilitätsmaß  
- Nutzung für Positionssizing & SL-Placement

## 5. Strategie-Design  
- Entry-/Exit-Regeln (EMA-Kreuzungen, RSI-Range, Bandberührung)  
- Risikomanagement:  
  - Risk-Reward-Verhältnis  
  - Positionsgrößenberechnung anhand `risk_per_trade_pct`  
- Lifecycle eines Trades: Signal → Order → Monitoring → Exit

## 6. Backtesting-Best-Practices  
- Datenqualität & Zeitzonen  
- Forward Fill & Ausreißer-Filter  
- Slippage, Kommission  
- Kennzahlen: Sharpe-Ratio, Max-Drawdown, Trefferquote

## 7. Online-Learning-Konzepte  
- Inkrem­entelles Lernen mit River  
- Feature-Engineering aus Indikatoren & Trade-Metadaten  
- Metriken: Accuracy, AUC, Precision, Recall

## 8. Weiterführende Ressourcen  
- Investopedia-Artikel  
- TA-Lib-Dokumentation  
- River-Pipeline-Guide  
- MetaQuotes-API-Referenz 