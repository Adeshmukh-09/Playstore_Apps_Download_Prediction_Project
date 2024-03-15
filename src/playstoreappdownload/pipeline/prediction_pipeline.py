import sys
import os
import pandas as pd
from src.playstoreappdownload.exception import CustomException
from src.playstoreappdownload.logger import logging 
from src.playstoreappdownload.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = os.path.join('Artifact','model.pkl')
            preprocessor_path = os.path.join('Artifact','preprocessor.pkl')
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self, playstore_app_ID:int, Category:str, Rating:float, Reviews:int, Price:float, Content_Rating:str, OS_Version_Required:str):
        self.playstore_app_ID = playstore_app_ID
        self.Category = Category
        self.Rating = Rating
        self.Reviews = Reviews
        self.Price = Price
        self.Content_Rating = Content_Rating
        self.OS_Version_Required = OS_Version_Required

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict={
                'playstore_app_ID': [self.playstore_app_ID],
                'Category': [self.Category],
                'Rating': [self.Rating],
                'Reviews': [self.Reviews],
                'Price': [self.Price],
                'Content_Rating': [self.Content_Rating],
                'OS_Version_Required': [self.OS_Version_Required]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)


    
