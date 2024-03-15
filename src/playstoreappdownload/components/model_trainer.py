import os 
import sys 
from dataclasses import dataclass
from src.playstoreappdownload.exception import CustomException
from src.playstoreappdownload.logger import logging 
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
from src.playstoreappdownload.utils import model_evaluation
from src.playstoreappdownload.utils import save_object

@dataclass

class ModelTrainerConfig:
    trained_model_file_path = os.path.join('Artifact','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluation_metric(self,actual,predict):
        mae = mean_absolute_error(actual,predict)
        rmse = np.sqrt(mean_absolute_error(actual,predict))
        r2_score_result = r2_score(actual,predict)
        return mae, rmse, r2_score_result


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('splitting the input and the test data')
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                'Linear Regression': LinearRegression(),
                'Ridge': Ridge(),
                'Lasso': Lasso(),
                'Elasticnet': ElasticNet(),
                'SVR': SVR(),
                'DecisionTreeRegression': DecisionTreeRegressor(),
                'RandomForestRegression': RandomForestRegressor(),
                'AdaboostRegression': AdaBoostRegressor(),
                'GradientboostRegression': GradientBoostingRegressor()
            }

            parameters = {
                'Linear Regression':{},
                'Ridge': {'alpha':[1,2,3,4,5,6,7,8,9,10]
                          },
                'Lasso': {'alpha':[1,2,3,4,5,6,7,8,9,10]
                          },
                'Elasticnet': {'alpha':[1,2,3,4,5,6,7,8,9,10]
                    },
                'SVR': {
                    'kernel':['linear','poly','rbf','sigmoid','precomputed'],
                    'gamma':['scale','auto']
                },
                'DecisionTreeRegression': {
                    'criterion': ['squared_error','friedman_mse','absolute_error','poisson'],
                    #'splitter': ['best','random'],
                    #'max_feature': ['sqrt','log2'],
                    #'max_depth': [1,2,3,4,5,6,7]
                },

                'RandomForestRegression':{
                   # 'criterion': ['squared_error','friedman_mse','absolute_error','poisson'],
                    #'max_feature': ['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,]
                },
                'AdaboostRegression': {
                    'learning_rate': [0.1,0.01,0.5,0.001],
                    #'loss': ['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,]
                },

                'GradientboostRegression':{
                    'learning_rate': [0.1,0.01,0.5,0.001],
                   # 'loss': ['linear','square','exponential'],
                    'subsample': [0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,],
                    #'criterion': ['friedman_mse','squared_error'],
                    #'max_feature': ['sqrt','log2']
                }

            }

            model_report:dict = model_evaluation(X_train,y_train,X_test,y_test,models,parameters)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            print('This is the best model')
            print(best_model_name)

            model_name = list(parameters.keys())

            actual_model=''
            for model in model_name:
                if best_model_name==model:
                    actual_model = actual_model + model

            best_parameters = parameters[actual_model]

            mlflow.set_registry_uri('https://dagshub.com/Adeshmukh-09/Playstore_Apps_Download_Prediction_Project.mlflow')
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            ## Mlflow pipeline initiated
            with mlflow.start_run():
                predicted_qualities = best_model.predict(X_test)
                (rmse, mae, r2_score_result) = self.evaluation_metric(y_test,predicted_qualities)
                mlflow.log_params(best_parameters)
                mlflow.log_metric('rmse',rmse)
                mlflow.log_metric('mae',mae)
                mlflow.log_metric('r2_score_result', r2_score_result)

                if tracking_url_type_store != 'file':
                    mlflow.sklearn.log_model(best_model, 'model', registered_model_name = actual_model)
                else:
                    mlflow.sklearn.log_model(best_model, 'model')


            if best_model_score < 0.4:
                raise CustomException('No best model found')
            logging.info(f'Best model found on both training and test data')

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )   

            predict = best_model.predict(X_test)
            r2_score_result = r2_score(y_test,predict) 

            return r2_score_result




        except Exception as e:
            raise CustomException(e,sys)
