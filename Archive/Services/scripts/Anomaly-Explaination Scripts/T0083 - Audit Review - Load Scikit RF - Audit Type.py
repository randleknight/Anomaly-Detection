from pathlib import Path
import pandas as pd
import numpy as np
from feature_engine.encoding import CountFrequencyEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
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

mlflow.set_experiment("T0083 - Audit Review")

for customer in config.customerList: 
    for auditLog in config.auditList:  
        for FilterType in config.FilterTypeList:
            with mlflow.start_run(run_name=f"Sklearn-RF Model-{auditLog}-Audit Type context-{customer}", nested=True) as grandparent_run:
                filepath = Path(f'{config.data_location}FS_audit_type_table_name/{auditLog}/{FilterType}/FS_audit_type_table_name_{auditLog}_{FilterType}_{customer}.parquet')
                if filepath.is_file()==True:
                    ### Replace code - Add encoding script and load data for inference.
                    df = pd.read_parquet(filepath, engine='pyarrow')
                    #df = df.drop(['T0083','T0083_SEVERITY','T_MLAR','T_MLAR_SEVERITY','MLAR_TYPE'],axis=1) #RK should be already done in FS scripts
                    #df = df.astype(str).replace('nan',np.nan).dropna(how='all',axis=1) #RK should be already done in FS scripts
                    df = df.astype(str)
                    features = df.columns.tolist()
                    categoricalColumns = df.select_dtypes(include=['object']).columns.tolist()
                    encoder = CountFrequencyEncoder(encoding_method='frequency', variables=categoricalColumns, ignore_format=True)
                    dataFrame = encoder.fit_transform(df) 
                    #Update to Anomaly field onced changed
                    X_train, X_test, y_train, y_test = train_test_split(
                        dataFrame[dataFrame.columns.drop(list(dataFrame.filter(like='T0083_SEVERITY')))], 
                        dataFrame.filter(like='T0083_SEVERITY').sum(axis=1), 
                        test_size=0.30, 
                        random_state=1234)
                    ###   
                      
                    # Define model parameters
                    model_name = f"Sklearn-RF_{auditLog}_{FilterType}_{customer}_audit_type_model"
                    model_version = "latest"
                    model_uri = f"models:/{model_name}/{model_version}"
                    rf_model = mlflow.sklearn.load_model(model_uri)
                
                    # Start an MLflow run
                    with mlflow.start_run(run_name=f"Model Prediction", nested=True) as parent_run:
                        # Predict anomalies and generate histogram
                        predictedTest = pd.DataFrame(rf_model.predict(X_test.values)) 
                        
                        # Combine the scores with the original data
                        finalTestDataframe = pd.concat([pd.DataFrame(data=predictedTest,columns=['anomaly']),y_train],axis=1)
                        predictedDF = finalTestDataframe[finalTestDataframe['anomaly'] > 0]

                        #Export Dataset
                        mlflow.log_table(predictedDF,"predictedDF.json")

                        with mlflow.start_run(run_name=f"Row Explaination", nested=True) as child_run:
                            #Train explaination model
                            #Replace below with load 'saved' explainer
                            explainer = shap.TreeExplainer(rf_model) #, feature_perturbation='interventional'
                            #shap_values = explainer.shap_values(predictedTest, check_additivity=False) #temp fix
                            shap_values = explainer(predictedTest)
                            shapForcePlot = plt.figure(figsize=(9, 3))
                            shap.force_plot(explainer.expected_value, shap_values.values[1], features=explainer.feature_names, matplotlib=True, show=True) #shap_values.values
                            mlflow.log_figure(shapForcePlot, f"_{customer}_LocalForcePlot.png")  
                else:
                    print("Error - Feature store file not found. Check directory.")

                    