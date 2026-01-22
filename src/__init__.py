__version__ = "1.0.12"
import logging
import sys

PKG_LOGGER_NAME = __name__
logger = logging.getLogger(PKG_LOGGER_NAME)

if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(name)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S"
        )
    )
    logger.addHandler(handler)
    logger.propagate = False


def set_verbose_mode(verbose: bool = True):
    """
    Toggle INFO/DEBUG messages for the whole package.

    Parameters
    ----------
    verbose : bool
        True  → log level INFO (default)
        False → log level WARNING
    """
    level = logging.INFO if verbose else logging.WARNING
    logger.setLevel(level)
    for h in logger.handlers:
        h.setLevel(level)

# Removed lazy_import as it was likely used for optional dependencies

from .deepheal import DeepHeal
