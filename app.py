import sys
from src.playstoreappdownload.logger import logging
from src.playstoreappdownload.exception import CustomException
from src.playstoreappdownload.components.data_ingestion import DataIngestion


if __name__ == "__main__":
    logging.info("Execution has started")

    try:
        data_ingestion = DataIngestion()
        data_ingestion.initiate_data_ingestion()

    except Exception as e:
        logging.info("Exception ocurred")
        raise CustomException(e,sys)

