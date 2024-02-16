import sys
from src.playstoreappdownload.logger import logging
from src.playstoreappdownload.exception import CustomException


if __name__ == "__main__":
    logging.info("Execution has started")

    try:
        pass

    except Exception as e:
        logging.info("Exception ocurred")
        raise CustomException(e,sys)

