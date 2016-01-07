import logging
import logging.config


class Logger(object):

    def __init__(self, log_level=logging.DEBUG, stdout_level=logging.INFO, filename='pyAl.log'):
        """
        Init actlearn.logger class
        :type log_level: int
        :param log_level: Log level for Log File
        :type stdout_level: int
        :param stdout_level: Log Level for Stdout
        :type filename: str
        :param filename: Name of log file
        :return:
        """
        self.log_level = log_level
        self.stdout_level = stdout_level
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setLevel(self.stdout_level)
        self.stream_handler.setFormatter(self.formatter)
        self.file_handler = logging.FileHandler(filename)
        self.file_handler.setLevel(self.log_level)
        self.file_handler.setFormatter(self.formatter)

    def get_logger(self, log_name, log_level=logging.DEBUG):
        """
        Get Logger with name configured
        :type log_name: str
        :param log_name: Name of this Logger Structure
        :type log_level: int
        :param log_level: Log level for current logger
        :return: logging.Logger
        """
        new_logger = logging.getLogger(log_name)
        if self.file_handler not in new_logger.handlers:
            new_logger.addHandler(self.file_handler)
        if self.stream_handler not in new_logger.handlers:
            new_logger.addHandler(self.stream_handler)
        new_logger.setLevel(log_level)
        return new_logger

    def set_log_filename(self, filename):
        """
        Change the log filename
        :type filename: str
        :param filename: Name of logfile
        :return: None
        """
        self.file_handler = logging.FileHandler(filename)
        self.file_handler.setLevel(self.log_level)
        self.file_handler.setFormatter(self.formatter)

    def set_log_level(self, log_level):
        """
        Change File Log Level
        :type log_level: int
        :param log_level: file log level
        :return:
        """
        self.log_level = log_level
        self.file_handler.setLevel(self.log_level)

    def set_stdout_level(self, log_level):
        """
        Change stdout Log Level
        :type log_level: int
        :param log_level: stream log level
        :return:
        """
        self.stdout_level = log_level
        self.stream_handler.setLevel(self.log_level)

actlearn_logger = Logger()
