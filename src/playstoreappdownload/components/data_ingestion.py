import os 
import sys 
from src.playstoreappdownload.exception import CustomException
from src.playstoreappdownload.logger import logging
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass

class DataIngestionConfig:
    train_data_path:str = os.path.join('Artifact','train.csv')
    test_data_path:str = os.path.join('Artifact','test.csv')
    raw_data_path:str = os.path.join('Artifact','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info('Reading the data from the .csv file')

            data = pd.read_csv('notebook\\Data\\downloads_new.csv')

            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info('created the raw data file')

            logging.info('splitting the raw data  into train and test data')
            train_data, test_data = train_test_split(data, test_size=0.20, random_state=42)

            train_data.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('train and test data has been splitted from raw data and different files has been created for both.')
            logging.info('Data ingestion completed.')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
          


    
