import logging
import colorlog
from logging.handlers import RotatingFileHandler
from typing import Optional, Union
import os

DEFAULT_LOG_COLORS = {
    "DEBUG": "purple",
    "INFO": "cyan",
    "WARNING": "yellow",
    "ERROR": "red,bold",
    "CRITICAL": "red,bg_white",
}

PRETTY_FORMAT = "%(log_color)s[%(asctime)s] " "%(message)s"


def create_logger(
    name: str,
    save_dir: Optional[str],
    level: Union[int, str] = logging.DEBUG,
    *,
    log_to_file: bool = True,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 3,
    log_colors: dict = DEFAULT_LOG_COLORS,
    datefmt: Optional[str] = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """
    Create a logger with the given name, attach a colored stream handler,
    and optionally add a rotating file handler.

    Parameters:
        name: Name of the logger.
        level: Logging level (numeric or string).
        log_to_file: If True, also log to a file with rotation.
        save_dir: File path to write logs if log_to_file is True.
        max_bytes: Maximum size in bytes before rotation.
        backup_count: Number of backup files to keep.
        use_utc: If True, timestamps are in UTC.
        log_colors: Mapping of level names to color definitions.
        datefmt: Date format string for timestamps.
    """
    logger = colorlog.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    filename = os.path.join(save_dir, "logger.log")

    if not logger.handlers:
        formatter = colorlog.ColoredFormatter(
            fmt=PRETTY_FORMAT,
            datefmt=datefmt,
            log_colors=log_colors,
            style="%",
            reset=True,
        )
        stream_handler = colorlog.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        if log_to_file and filename:
            file_formatter = logging.Formatter(
                "[%(asctime)s]%(message)s",
                datefmt=datefmt,
            )
            file_handler = RotatingFileHandler(
                filename, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

    return logger
