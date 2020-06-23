import os
import functools
import logging.config

import yaml
from tqdm import tqdm


class TqdmStreamHandler(logging.Handler):
    def __init__(self):
        super().__init__()

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


@functools.lru_cache()  # so that calling setup multiple times won't add many handlers
def setup_logger(config_file="logger.yml", name=None, default_level=logging.INFO):
    """Setup logger with yaml configuration.

    :param str | None name: name of the logger
    :param str config_file: yaml configuration file

    :returns: logger

    If configuration file does not exist use :func:`logging.basicConfig`"""

    # Configure logger
    config_file = os.environ.get("LOG_CFG", config_file)
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)
    else:
        level = logging.getLevelName(os.environ.get("LOG_LEVEL", default_level))
        logging.basicConfig(level=level)

    logger = logging.getLogger(name)

    return logger
