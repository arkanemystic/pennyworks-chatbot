import logging
import sys
from typing import Optional

def setup_logger(log_file: Optional[str] = None, level: int = logging.DEBUG):
    logger = logging.getLogger("pennyworks")
    logger.setLevel(level)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')

    # Console handler
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # Optional file handler
    if log_file and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
