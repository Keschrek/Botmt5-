import logging
import logging.handlers
import os
from pathlib import Path

def setup_logging(log_cfg, project_root):
    """Sets up logging based on the provided configuration."""

    # Ensure the log directory exists
    log_dir = project_root / Path(log_cfg['path']).parent
    os.makedirs(log_dir, exist_ok=True)

    # Get the absolute path for the log file
    log_file_path = project_root / Path(log_cfg['path'])

    # Basic configuration for console output (optional, for debugging)
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a logger
    # Get the root logger as we want to handle all messages
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # Set the minimum logging level

    # Prevent adding duplicate handlers if setup is called multiple times
    if not logger.handlers:
        # Create a rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=log_cfg['max_bytes'],
            backupCount=log_cfg['backup_count']
        )

        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Add formatter to handler
        file_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(file_handler)

        # Add a console handler as well for immediate feedback
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        logger.info("Logging setup complete.")
    else:
        logger.info("Logging already set up.") 