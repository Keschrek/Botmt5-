import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import os
import pandas as pd
from typing import Dict, Any, List # Import List for type hinting signals

logger = logging.getLogger(__name__)

class SnapshotHelper:
    def __init__(self, config: Dict[str, Any], project_root: Path):
        """Initialisiert den SnapshotHelper."""
        self.config = config # Store the full config
        self.log_cfg = config.get('logs', {})
        self.indicators_cfg = config.get('indicators', {})
        self.project_root = project_root
        self.snapshots_dir = self.project_root / Path(self.log_cfg.get('snapshots_dir', 'logs/snapshots/'))
        
        # Ensure snapshots directory exists
        os.makedirs(self.snapshots_dir, exist_ok=True)
        logger.info(f"SnapshotHelper initialized. Snapshots will be saved to: {self.snapshots_dir}")

    def save_snapshot(self, data: pd.DataFrame, filename_prefix: str, signals: pd.Series = None):
        """Generates and saves a snapshot plot of the data.\n\n        Args:\n            data: DataFrame containing price and indicator data.\n            filename_prefix: Prefix for the saved filename (e.g., symbol).\n            signals: Optional pandas Series with index matching data.index, indicating signal points (1 for buy, -1 for sell).\n        """
        if data.empty:
            logger.warning("Cannot save snapshot: Data is empty.")
            return

        # Ensure data index is datetime and potentially timezone-aware for proper plotting
        if not isinstance(data.index, pd.DatetimeIndex):
             logger.error("Data index is not DatetimeIndex. Cannot save snapshot.")
             return

        # Create figure and primary axes for price and overlays
        fig, axes1 = plt.subplots(figsize=(15, 8))
        axes2 = axes1.twinx() # Create a secondary axes for indicators with different scale
        
        # Plot the close price on primary axis
        axes1.plot(data.index, data['close'], label='Close Price', color='blue')
        axes1.set_ylabel('Price', color='blue', fontsize=10)
        axes1.tick_params(axis='y', labelcolor='blue')
        
        # Plot indicators that are in the DataFrame on primary axis (overlays on price)
        # EMA
        for period in self.indicators_cfg.get('ema_periods', []):
            ema_col = f'EMA_{period}'
            if ema_col in data.columns:
                axes1.plot(data.index, data[ema_col], label=f'EMA {period}') # Matplotlib auto-assigns colors
        
        # Bollinger Bands
        if 'BB_upper' in data.columns and 'BB_lower' in data.columns:
             axes1.plot(data.index, data['BB_upper'], label='BB Upper', color='green', linestyle='--', alpha=0.7)
             axes1.plot(data.index, data['BB_lower'], label='BB Lower', color='red', linestyle='--', alpha=0.7)
             axes1.fill_between(data.index, data['BB_lower'], data['BB_upper'], color='gray', alpha=0.1)

        # Plot indicators on secondary axis (different scale)
        # RSI
        if 'RSI' in data.columns:
             axes2.plot(data.index, data['RSI'], label='RSI', color='purple', linestyle='-', alpha=0.7)
             axes2.set_ylabel('RSI', color='purple', fontsize=10)
             axes2.tick_params(axis='y', labelcolor='purple')
             # Add horizontal lines for RSI levels if desired
             # axes2.axhline(70, color='purple', linestyle=':', alpha=0.5)
             # axes2.axhline(30, color='purple', linestyle=':', alpha=0.5)

        # MACD
        if 'MACD_line' in data.columns and 'MACD_signal' in data.columns:
            # MACD Histogram can also be plotted as bars
            if 'MACD_hist' in data.columns:
                 axes2.bar(data.index, data['MACD_hist'], label='MACD Hist', color='gray', alpha=0.3)
            axes2.plot(data.index, data['MACD_line'], label='MACD Line', color='darkorange', linestyle='-', alpha=0.7)
            axes2.plot(data.index, data['MACD_signal'], label='MACD Signal', color='teal', linestyle='-', alpha=0.7)
            # MACD lines are typically centered around 0, so maybe they also need a separate axis or scale adjustment
            # For simplicity, plotting on axes2 for now.

        # Plot signals if provided
        if signals is not None and not signals.empty:
             buy_signals = signals[signals == 1]
             sell_signals = signals[signals == -1]

             # Plot buy signals (upward arrows) at the low price level + a margin
             # Need to ensure alignment with data index
             if not buy_signals.empty:
                 buy_indices = data.index.intersection(buy_signals.index)
                 if not buy_indices.empty:
                      # Adjust arrow vertical position slightly above the low of the bar
                      arrow_pos_y = data.loc[buy_indices, 'low'] * 0.99 # Or a fixed offset
                      axes1.plot(buy_indices, arrow_pos_y, '^', markersize=10, color='green', lw=0, label='Buy Signal', alpha=0.8)

             # Plot sell signals (downward arrows) at the high price level - a margin
             if not sell_signals.empty:
                 sell_indices = data.index.intersection(sell_signals.index)
                 if not sell_indices.empty:
                      # Adjust arrow vertical position slightly below the high of the bar
                      arrow_pos_y = data.loc[sell_indices, 'high'] * 1.01 # Or a fixed offset
                      axes1.plot(sell_indices, arrow_pos_y, 'v', markersize=10, color='red', lw=0, label='Sell Signal', alpha=0.8)

        # Combine legends from both axes
        lines1, labels1 = axes1.get_legend_handles_labels()
        lines2, labels2 = axes2.get_legend_handles_labels()
        # Avoid duplicate labels if an indicator is plotted on both (shouldn't be the case here)
        all_lines = lines1 + lines2
        all_labels = labels1 + labels2
        
        # Formatting the plot
        axes1.set_title(f'Trading Signal Snapshot: {filename_prefix}', fontsize=14)
        axes1.set_xlabel('Time', fontsize=10)
        # Only show grid on the primary axis for clarity
        axes1.grid(True)

        # Format the x-axis to show dates and times nicely
        fig.autofmt_xdate()
        axes1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        # axes1.xaxis.set_major_locator(mdates.AutoDateLocator())

        # Add a single legend for all plotted elements
        axes1.legend(all_lines, all_labels, loc='best')

        # Set DPI and save the figure
        timestamp_str = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        # Sanitize filename_prefix to remove potentially invalid characters
        sanitized_prefix = "".join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in filename_prefix)
        snapshot_filename = f"{sanitized_prefix}_{timestamp_str}.png"
        snapshot_path = self.snapshots_dir / snapshot_filename

        try:
            plt.savefig(snapshot_path, dpi=self.log_cfg.get('snapshot_dpi', 150), bbox_inches='tight')
            logger.info(f"Snapshot saved successfully to: {snapshot_path}")
        except Exception as e:
            logger.error(f"Failed to save snapshot to {snapshot_path}: {e}")
        finally:
            plt.close(fig) # Close the figure to free memory

    def save_equity_curve_snapshot(self, equity_curve: pd.Series, filename_prefix: str):
        """Generates and saves a snapshot plot of the equity curve.\n\n        Args:\n            equity_curve: Pandas Series representing the equity curve (index is datetime).\n            filename_prefix: Prefix for the saved filename.\n        """
        if equity_curve.empty:
            logger.warning("Cannot save equity curve snapshot: Data is empty.")
            return

        if not isinstance(equity_curve.index, pd.DatetimeIndex):
             logger.error("Equity curve index is not DatetimeIndex. Cannot save equity curve snapshot.")
             return

        fig, axes = plt.subplots(figsize=(15, 8))

        # Plot the equity curve
        axes.plot(equity_curve.index, equity_curve.values, label='Equity Curve', color='blue')

        # Formatting the plot
        axes.set_title(f'Equity Curve Snapshot: {filename_prefix}', fontsize=14)
        axes.set_xlabel('Time', fontsize=10)
        axes.set_ylabel('Equity', fontsize=10)
        axes.legend(loc='best')
        axes.grid(True)

        # Format the x-axis to show dates and times nicely
        fig.autofmt_xdate()
        axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

        # Set DPI and save the figure
        timestamp_str = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        sanitized_prefix = "".join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in filename_prefix)
        snapshot_filename = f"{sanitized_prefix}_equity_curve_{timestamp_str}.png"
        snapshot_path = self.snapshots_dir / snapshot_filename

        try:
            plt.savefig(snapshot_path, dpi=self.log_cfg.get('snapshot_dpi', 150), bbox_inches='tight')
            logger.info(f"Equity curve snapshot saved successfully to: {snapshot_path}")
        except Exception as e:
            logger.error(f"Failed to save equity curve snapshot to {snapshot_path}: {e}")
        finally:
            plt.close(fig) # Close the figure to free memory

    # Note: This helper needs access to the indicators configuration to know which EMAs etc. to plot.
    # It also needs the data itself (a pandas DataFrame with price and indicator columns).
    # The `filename_prefix` could include symbol and signal type.

# This helper can be initialized in run_bot.py and passed to LiveTrading/Backtester. 