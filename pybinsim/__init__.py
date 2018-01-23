import logging

__version__ = "1.2.1"

from pybinsim.application import BinSim


def init_logging(loglevel):
    console_handler = logging.StreamHandler()
    console_handler.setLevel(loglevel)

    formatter = logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    logger = logging.getLogger("pybinsim")
    logger.addHandler(console_handler)
    logger.setLevel(loglevel)

    return logger


logger = init_logging(logging.INFO)
logger.info("Starting pybinsim v{}".format(__version__))
