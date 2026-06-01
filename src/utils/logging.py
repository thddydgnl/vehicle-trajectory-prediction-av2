import logging
from pathlib import Path

from src.utils.paths import ensure_dir


def get_logger(name: str, log_path: str | Path | None = None) -> logging.Logger:
    """Create a simple project logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_path is not None:
        output_path = Path(log_path)
        ensure_dir(output_path.parent)
        existing_files = [
            handler
            for handler in logger.handlers
            if isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename) == output_path
        ]
        if not existing_files:
            file_handler = logging.FileHandler(output_path, encoding="utf-8")
            file_handler.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(file_handler)

    return logger
