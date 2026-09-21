"""Module that implements the writing of logs.

 `setup_logger` writes D-CAF logs to `dcaf.log` and returns a logger that can
 optionally record timing diagnostics.

Example:
    ```python
    from dcaf.utilities.logger import setup_logger

    logger = setup_logger("dcaf_output", level="debug")
    logger.info("Starting simulation")
    with logger.timing("Initializing solver"):
        initialize_solver()
    ```
"""
import logging
import os
import time
from contextlib import contextmanager


class FlushFileHandler(logging.FileHandler):
    """FileHandler that flushes on every log record."""
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()

class TimedLogger:
    """Wrapper that logs labels always, and timings only in DEBUG mode.

    Args:
        logger: Python logger receiving D-CAF log records.
    """
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    @contextmanager
    def timing(self, label: str, first_label: bool = True):
        """Log a labeled operation and, in debug mode, its elapsed time.

        Args:
            label: Description written to the log.
            first_label: Whether to write the label at info level
                before timing begins.
        """
        # Always log the label
        if first_label:
            self.logger.info(f"{label}")
        t0 = time.perf_counter()
        yield
        dt = time.perf_counter() - t0
        # Extra timing line only in DEBUG
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"[TIMING] {label} {dt:.6f}")

    # Pass-through methods
    def info(self, *args, **kwargs):    return self.logger.info(*args, **kwargs)
    def warning(self, *args, **kwargs): return self.logger.warning(*args, **kwargs)
    def error(self, *args, **kwargs):   return self.logger.error(*args, **kwargs)
    def debug(self, *args, **kwargs):   return self.logger.debug(*args, **kwargs)
    def critical(self, *args, **kwargs):return self.logger.critical(*args, **kwargs)
    def isEnabledFor(self, level):      return self.logger.isEnabledFor(level)

def setup_logger(
    output_folder: str = "./dcaf_output/",
    level: int | str = None,
) -> TimedLogger:
    """Configure the D-CAF file logger and return its timing wrapper.

    Existing handlers on the root logger are removed. The new logger writes to
    `dcaf.log` in `output_folder`, replacing any existing file. If `level` is
    omitted, `DCAF_LOGLEVEL` is used, with `INFO` as the fallback.

    Args:
        output_folder: Directory where `dcaf.log` is written.
        level: Logging level name or Python logging level.

    Returns:
        (TimedLogger): Configured logger with standard log methods and the
            `timing` context manager.
    """
    os.makedirs(output_folder, exist_ok=True)
    log_file = os.path.join(output_folder, "dcaf.log")

    # Reset any existing handlers
    root = logging.getLogger()
    if root.handlers:
        for h in root.handlers[:]:
            root.removeHandler(h)

    handler = FlushFileHandler(log_file, mode="w")
    # drop lines that contain "petar: transition from state"
    handler.addFilter(lambda record: "petar: transition from state" not in record.getMessage().lower())

    # Control verbosity via level
    if level is None:
        level_name = os.environ.get("DCAF_LOGLEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
    elif isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[handler]
    )

    base_logger = logging.getLogger("dcaf")
    return TimedLogger(base_logger)
