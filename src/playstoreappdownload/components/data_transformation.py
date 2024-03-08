import os
import sys
from src.playstoreappdownload.exception import CustomException
from src.playstoreappdownload.logger import logging
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.playstoreappdownload.utils import save_object
import pandas as pd
import numpy as np

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('Artifact','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            logging.info('Data transformation initiated')

            numerical_columns = ['playstore app ID','Rating','Reviews','Price']
            categorical_columns = ['Category','Content_Rating','OS_Version_Required']

            Category_Cat = [
                'Education', 'Tools', 'Entertainment', 'Books And Reference', 'Family', 'Finance',
                'Lifestyle', 'Productivity', 'Music And Audio', 'Business', 'Personalization',
                'Health And Fitness', 'Game', 'News And Magazines', 'Photography', 'Game Puzzle',
                'Sports', 'Shopping', 'Travel And Local', 'Communication', 'Medical', 'Game Casual',
                'Game Arcade', 'Social', 'Game Action', 'Food And Drink', 'Video Players', 'Game Card',
                'Game Simulation', 'Game Educational', 'Maps And Navigation', 'Game Role Playing',
                'Game Strategy', 'Game Adventure', 'Auto And Vehicles', 'Game Word', 'Game Sports',
                'Game Racing', 'Dating', 'Weather', 'Game Board', 'Art And Design', 'House And Home',
                'Game Trivia', 'Game Casino', 'Parenting', 'Events', 'Beauty', 'Libraries And Demo',
                'Comics', 'Game Music']
            
            Content_Rating_cat = ['Everyone','Teen','Everyone 10+','Mature 17+','Adults only 18+','Unrated']
            
            OS_Version_Required_cat = ['4.1 and up', '4.0.3 and up', '4.0 and up', '4.4 and up','Varies with device', '5.0 and up', '2.3 and up', '4.2 and up', '4.3 and up', '2.3.3 and up', '2.2 and up', '3.0 and up', '6.0 and up', '2.1 and up', '5.1 and up', '1.6 and up','7.0 and up', '1.5 and up', '3.2 and up', '2.0 and up', '4.4w and up', '3.1 and up', '8.0 and up', '2.0.1 and up', '7.1 and up', '1.1 and up', '1.0 and up']

            numerical_pipeline = Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
            ])

            categorical_pipeline = Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder',OrdinalEncoder(categories=[Category_Cat,Content_Rating_cat,OS_Version_Required_cat])),
                ('scaler',StandardScaler(with_mean=False))
            ])

            logging.info(f'numerical columns are:{numerical_columns}')
            logging.info(f'Categorical columns are:{categorical_columns}')

            preprocessor = ColumnTransformer([
                ('numerical_pipeline',numerical_pipeline,numerical_columns),
                ('categorical_pipeline',categorical_pipeline,categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('reading teh test and train data')
            preprocessing_obj = self.get_data_transformer_object()

            target_column = 'Downloads'

            input_feature_train_df = train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df = train_df.loc[:,[target_column]]

            input_feature_test_df = test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df = test_df.loc[:,[target_column]]

            logging.info('Applying preprocessor on train and test dataset')
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info('Savinf preprocessing object')

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr
            )

        except Exception as e:
            raise CustomException(e,sys)




