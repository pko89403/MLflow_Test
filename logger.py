import logging
import logging.handlers
from config import config

class logger():

    def __init__(self):
        #logger 생성
        self.set_logger()

    def set_logger(self):
        self.logger = logging.getLogger(config().name())
        # set baseline
        self.logger.setLevel(logging.CRITICAL)

        # set log format
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s | %(filename)s:%(lineno)s | %(process)s] >> %(message)s')

        # generate file handler to save log data as file
        filehandler = logging.FileHandler(config().save_dir())
        filehandler.setFormatter(formatter)
        self.logger.addHandler(filehandler)

        self.set_Level(config().json_data["Logger"]["level"])

    def set_Level(self, level):
        self.logger.setLevel(str(config().json_data["Logger"]["level"]))

    """
    def debug(self, data):
        self.logger.debug(data)

    def info(self,data):
        self.logger.info(data)

    def warning(self,data):

        self.logger.warning(data)

    def error(self,data):

        self.logger.error(data)

    def critical(self,data):
        self.logger.critical(data)
    """