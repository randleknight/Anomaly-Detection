from pathlib import Path
import pandas as pd
import numpy as np
from feature_engine.encoding import CountFrequencyEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import shap
import mlflow.sklearn
import mlflow

import config

# define Mlflow env. settings
mlflow.set_tracking_uri(uri=config.mlflow_uri)
%env MLFLOW_TRACKING_URI= config.mlflow_db
%env mlflow_tracking_username = config.mlflow_un
%env mlflow_tracking_password = config.mlflow_pw
#mlflow.enable_system_metrics_logging()
#mlflow.autolog(log_models=True, log_datasets=True)

mlflow.set_experiment("T0083 - Audit Review")
with mlflow.start_run(run_name=f"Sklearn-IF Model", nested=True) as greatgrandparent_run:
    for customer in config.customerList:
        for auditLog in config.auditList:
            with mlflow.start_run(run_name=f"Base context-{auditLog}-{customer}", nested=True) as grandparent_run:
                filepath = Path(f'{config.data_location}FS_base_data/{auditLog}/FS_base_data_{auditLog}_{customer}.parquet')
                if filepath.is_file()==True:
                    df = pd.read_parquet(filepath, engine='pyarrow')
                    #df = df.drop(['T0083','T0083_SEVERITY','T_MLAR','T_MLAR_SEVERITY','MLAR_TYPE'],axis=1) #RK should be already done in FS scripts
                    #df = df.astype(str).replace('nan',np.nan).dropna(how='all',axis=1) #RK should be already done in FS scripts
                    df = df.astype(str)
                    features = df.columns.tolist()
                    categoricalColumns = df.select_dtypes(include=['object']).columns.tolist()
                    encoder = CountFrequencyEncoder(encoding_method='frequency', variables=categoricalColumns, ignore_format=True)
                    dataFrame = encoder.fit_transform(df) 
                    train, test = train_test_split(dataFrame,test_size=0.30)


                    # Define model parameters
                    numTrees = 100
                    #sampleSize = 256
                    random = np.random.RandomState(1234)

                    # Initialize the model
                    if_model = IsolationForest(random_state=random)
                    
                    # Start an MLflow run
                    with mlflow.start_run(run_name=f"Model Evaluation", nested=True) as parent_run:
                        # Log model parameters and train
                        #mlflow.log_param("ntrees", numTrees)
                        #mlflow.log_param("sample_size", sampleSize)
                        if_model.fit(dataFrame) 
                        
                        # Predict anomalies and generate histogram
                        predictedTrain = if_model.predict(train) 
                        predictedTest = if_model.predict(test) 
                        
                        # Combine the scores with the original data
                        finalTrainDataframe = pd.concat([pd.DataFrame(data=predictedTrain,columns=['anomaly']),train],axis=1)
                        finalTestDataframe = pd.concat([pd.DataFrame(data=predictedTest,columns=['anomaly']),test],axis=1)
                        predictedDF = finalTrainDataframe[finalTrainDataframe['anomaly'] > 0]

                        #Export Dataset
                        mlflow.log_table(predictedDF,"predictedDF.json")

                        with mlflow.start_run(run_name=f"Model Explaination", nested=True) as child_run:
                            #Train explaination model
                            #custom_masker = shap.maskers.Partition(dataFrame, max_samples=30, clustering='correlation')
                            #explainer = shap.Explainer(if_model.predict,masker=custom_masker) #, dataFrame) max_eval=500,nsamples=100
                            explainer = shap.TreeExplainer(if_model) 
                            shap_values = explainer.shap_values(dataFrame)
                            shapBarplot = plt.figure(figsize=(9, 3))
                            shap.summary_plot(shap_values, features=explainer.feature_names, plot_type="bar", max_display=15)
                            mlflow.log_figure(shapBarplot, f"{customer}_GlobalBarPlot.png")  
                            logModel = mlflow.sklearn.log_model(sk_model=if_model,artifact_path=f"Sklearn-IF_{auditLog}_{customer}_base_model",registered_model_name=f"Sklearn-IF_{auditLog}_{customer}_base_model")

                            #explainOutput = DRF_model.explain(predictedTest)
                            #predictContribution = DRF_model.predict_contributions(predictedTest)
                else:
                    print("Error - Feature store file not found. Check directory.")