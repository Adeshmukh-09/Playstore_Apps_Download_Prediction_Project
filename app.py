import sys
from src.playstoreappdownload.logger import logging
from src.playstoreappdownload.exception import CustomException
from src.playstoreappdownload.components.data_ingestion import DataIngestion
from src.playstoreappdownload.components.data_transformation import DataTransformation
from src.playstoreappdownload.components.model_trainer import ModelTrainer
from flask import Flask, request, render_template
from src.playstoreappdownload.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

#creating route for the home page 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method=='GET':
       return render_template('home.html')
    else:
        data = CustomData(
            playstore_app_ID = request.form.get('playstore_app_ID'),
            Category = request.form.get('Category'),
            Rating = request.form.get('Rating'),
            Reviews = request.form.get('Reviews'),
            Price = request.form.get('Price'),
            Content_Rating = request.form.get('Content_Rating'),
            OS_Version_Required = request.form.get('OS_Version_Required')
        )
        pred_df = data.get_data_as_dataframe()
        print(pred_df)
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(pred_df)
        results = round(pred[0],0)
        return render_template('home.html',final_result = results)



if __name__ == "__main__":
    logging.info("Execution has started")

    try:
        data_ingestion = DataIngestion()
        train_data_path,test_data_path = data_ingestion.initiate_data_ingestion()
        data_transformation = DataTransformation()
        train_arr,test_arr = data_transformation.initiate_data_transformation(train_data_path,test_data_path)
        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))
        app.run(host = '0.0.0.0', port = 9090, debug = True)


    except Exception as e:
        logging.info("Exception ocurred")
        raise CustomException(e,sys)

