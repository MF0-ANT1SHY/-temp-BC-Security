import logging
import os


# Configure logging
def setup_logging(name, level="critical"):
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Map string level to logging level
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    if level.lower() not in level_map:
        raise ValueError(
            f"Invalid logging level: {level}. Must be one of {list(level_map.keys())}"
        )

    log_level = level_map[level.lower()]

    # Configure basic logging
    logging.basicConfig(level=log_level)

    # Main logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    fh.setLevel(log_level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
