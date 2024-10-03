import logging
import time

def configure_logger(debug=False):
    fh = logging.FileHandler(f'{time.strftime("%Y-%m-%d_%H%M%S")}-experiments.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO if not debug else logging.DEBUG)
    logging.basicConfig(format='(%(asctime)s) %(levelname)s - %(name)s:%(message)s', level=logging.DEBUG, handlers=[fh,ch])

def info(name, message):
    logger = logging.getLogger(name)
    logger.info(message)

def debug(name, message):
    logger = logging.getLogger(name)
    logger.debug(message)

def warn(name, message):
    logger = logging.getLogger(name)
    logger.warn(message)

