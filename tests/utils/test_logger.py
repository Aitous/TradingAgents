
import logging
from io import StringIO

from tradingagents.utils.logger import get_logger


def test_logger_formatting():
    # Capture stdout
    capture = StringIO()
    handler = logging.StreamHandler(capture)
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

    logger = get_logger("test_logger_unit")
    logger.setLevel(logging.INFO)
    # Remove existing handlers to avoid cluttering output or double logging
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    logger.addHandler(handler)

    logger.info("Test Info")
    logger.error("Test Error")

    output = capture.getvalue()
    print(f"Captured: {output}") # For debugging
    assert "INFO: Test Info" in output
    assert "ERROR: Test Error" in output
