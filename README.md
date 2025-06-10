# Trading Bot

## Voraussetzungen
- Python 3.10+

## Installation
```bash
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows PowerShell
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Konfiguration

    Lege deine API-Keys in config/config.yaml an.

    Verifiziere gegen das Schema:

    ```bash
    python run_bot.py --mode test --config config/config.yaml
    ```

Usage

```bash
python run_bot.py --mode live --config config/config.yaml
```

Windows Scheduled Tasks

    Öffne Task Scheduler.

    Erstelle Task für scripts/rotate-logs.ps1 (täglich).

    Erstelle Task für scripts/update-docs.ps1 (wöchentlich).

    Erstelle Task für scripts/build-docs.ps1 (optional nach Docs-Änderung).

Security

⚠️ Wichtig: config/config.yaml enthält sensible Daten.

    Niemals ins öffentliche Repo pushen!

    .gitignore enthält:

    ```
    venv/
    logs/
    logs/snapshots/
    __pycache__/
    ```

Linting & Typisierung

    Lint mit flake8 .

    Typen checken mit mypy src/

Version Pinning

In requirements.txt sind kritische Pakete auf Major-Version fixiert, um Breaking-Changes zu vermeiden.

## Dokumentation

Eine umfassende Dokumentation des Bots und der zugrunde liegenden Konzepte finden Sie in den [MkDocs-Seiten](site/index.html). Dies beinhaltet:

- **Trading-Grundlagen:** Erklärungen zu Marktmechanismen, Order-Typen, Indikatoren, Strategie-Design und Risikomanagement.
- Details zur Bot-Architektur und den Modulen.
- Anleitungen zur Konfiguration und zum Betrieb.

Stellen Sie sicher, dass Sie die Dokumentation nach Änderungen neu bauen, indem Sie das Skript `scripts/build-docs.ps1` ausführen.

## Trading-Domain-Wissen

In `docs/trading_basics.md` findest du ein umfangreiches Glossar und Praxiswissen zu:  
- Marktmechanik und Order-Typen  
- Spezifika von XAUUSD  
- Theorie & Anwendung der Indikatoren  
- Strategie-Design und Risikomanagement  
- Backtesting-Methodik  
- Online-Learning-Konzepte

Diese Referenz stellt sicher, dass jeder Entwickler und der Bot selbst auf fundiertem Trading-Wissen aufbaut.

## Sicherheit

⚠️ Wichtig: config/config.yaml enthält sensible Daten.

    Niemals ins öffentliche Repo pushen!

    .gitignore enthält:

    ```
    venv/
    logs/
    logs/snapshots/
    __pycache__/
    ```