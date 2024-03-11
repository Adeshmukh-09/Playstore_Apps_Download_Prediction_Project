import sys
from src.playstoreappdownload.logger import logging
from src.playstoreappdownload.exception import CustomException
from src.playstoreappdownload.components.data_ingestion import DataIngestion
from src.playstoreappdownload.components.data_transformation import DataTransformation
from src.playstoreappdownload.components.model_trainer import ModelTrainer


if __name__ == "__main__":
    logging.info("Execution has started")

    try:
        data_ingestion = DataIngestion()
        train_data_path,test_data_path = data_ingestion.initiate_data_ingestion()
        data_transformation = DataTransformation()
        train_arr,test_arr = data_transformation.initiate_data_transformation(train_data_path,test_data_path)
        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))


    except Exception as e:
        logging.info("Exception ocurred")
        raise CustomException(e,sys)

