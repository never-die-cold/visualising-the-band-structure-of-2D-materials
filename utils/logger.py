"""
Application logging setup.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from .paths import app_data_dir


def setup_logger(name: str = "bandviz", level: int = logging.INFO) -> logging.Logger:
    """
    配置应用日志：同时输出到控制台和文件 logs/bandviz_YYYY-MM-DD.log。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler
    log_dir = app_data_dir() / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"bandviz_{datetime.now().strftime('%Y-%m-%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
