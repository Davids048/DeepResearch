import logging
import sys
import os
import datetime

class ColoredFormatter(logging.Formatter):
    """Formatter that adds color to log levels"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Get the color for this log level
        color = self.COLORS.get(record.levelname, self.RESET)
        
        # Format the message
        formatted = super().format(record)
        
        # Find where the actual message starts (after the levelname)
        # Split at " - " after levelname
        parts = formatted.split(' - ', 3)  # Split into at most 4 parts
        if len(parts) >= 4:
            # Color everything except the actual message
            colored = f">>>>> {color}{parts[0]} - {parts[1]} - {parts[2]}{self.RESET} - {parts[3]}"
            return colored
        return formatted


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
        fmt_str = '[%(asctime)s] - %(levelname)s - [%(name)s:%(lineno)d] - \n %(message)s'
        datefmt = '%m-%d %H:%M:%S'

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
