import logging
from pathlib import Path

#get logger for a given name and log file
def get_logger(name, log_file):

    models_dir = Path(__file__).resolve().parents[1]
    logs_dir = models_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter("%(levelname)s - %(message)s")

        #keep all logs under `Licenta/models/logs/` regardless of current working directory.
        log_path = Path(log_file)
        if not log_path.is_absolute():
            log_path = logs_dir / log_path.name

        file_handler = logging.FileHandler(str(log_path))
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger