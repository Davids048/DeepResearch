import logging
import sys
import os
import datetime

class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def format(self, record):
        original = record.levelname
        record.levelname = f"{self.COLORS.get(original, self.RESET)}{original}{self.RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original



def setup_logging(name=None, logger=None, log_file=None, level=logging.INFO):
    """Setup a logger with consistent formatting"""
    # print("setting up logging...")
    if name:
        logger = logging.getLogger(name)
        print(f"using named logger: {name}.")   
    if logger is None:
        print('using default logger...')
        logger = logging.getLogger(__name__)

    logger.propagate = False
    if not logger.handlers:
        fmt_str = "[%(asctime)s] [%(levelname)-8s] [%(name)s:%(lineno)d:%(funcName)s]  %(message)s"
        datefmt = "%m-%d %H:%M:%S"

        # Console handler with color
        console_formatter = ColoredFormatter(
            fmt_str,
            datefmt=datefmt
        )
        console_handler = logging.StreamHandler(sys.stdout) 
        console_handler.setFormatter(console_formatter) 
        logger.addHandler(console_handler)
        
        # File handler without color (plain text)
        if log_file:
            # create unique log files
            if os.path.exists(log_file):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = log_file.replace(".log", f"_{timestamp}.log")
            
            plain_formatter = logging.Formatter(
                fmt_str,
                datefmt=datefmt
            )
            file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            file_handler.setFormatter(plain_formatter) 
            logger.addHandler(file_handler)
            
        logger.setLevel(level)
    return logger
