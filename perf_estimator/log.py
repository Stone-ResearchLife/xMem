import logging.config
import datetime
from .config import Config, default_setting

__Basic_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'tmp.log',
            'maxBytes': 10*1024*1024,  # 10 MB
            'backupCount': 3,
            'encoding': 'utf8',
        },
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
            'formatter': 'standard',
        },
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['default', "console"],
            'level': 'INFO',
            'propagate': True
        },
    }
}


def init_logging(level: str = "INFO", conf: Config = default_setting) -> None:
    log_dir = conf.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir.joinpath(f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log")
    _basic = __Basic_CONFIG
    _basic["handlers"]["default"]["filename"] = str(log_file)
    _basic["loggers"][""]["level"] = level
    logging.config.dictConfig(__Basic_CONFIG)