from pathlib import Path
import pandas as pd
import numpy as np
from feature_engine.encoding import CountFrequencyEncoder
import shap
import joblib
import datetime
import json
import sys
import contextlib

if len(sys.argv) > 1:
    configAuditSelect = sys.argv[1]
    auditList = configAuditSelect.split(",")  
    dbType = sys.argv[2]
    configCredName = sys.argv[3] 
    customerList = configCredName.split(",")  
    dataLocation = sys.argv[4]
    modelLocation = sys.argv[5]
    useMlflow = sys.argv[6]
    mlflowUri = sys.argv[7]
    mlflowExperiment = sys.argv[8]
else:
    print("No variable received.")
    sys.exit(1)

# define Mlflow env. settings
if useMlflow == 'True':
    import mlflow
    import mlflow.sklearn
    print(f"MLflow connected.")
    mlflow.set_tracking_uri(mlflowUri)
    mlflow.set_experiment(mlflowExperiment)
    # remove below later
    # mlflow.set_tracking_uri(uri=config.mlflow_uri)
    # %env MLFLOW_TRACKING_URI= config.mlflow_db
    # %env mlflow_tracking_username = config.mlflow_un
    # %env mlflow_tracking_password = config.mlflow_pw
    # mlflow.enable_system_metrics_logging()
    # mlflow.autolog(log_models=True, log_datasets=True)

def describe_top_shap_values(shap_values, X, feature_names, top_n=3):
    descriptions = []

    for i in range(len(X)):
        row_shap_values = shap_values[i]
        top_indices = sorted(range(len(row_shap_values)), key=lambda k: abs(row_shap_values[k]), reverse=True)[:top_n]

        explanation = f""
        for idx in top_indices:
            feature_name = feature_names[idx]
            shap_value = row_shap_values[idx]
            feature_value = X.iloc[i, idx]

            if shap_value > 0:
                effect = "increased"
            elif shap_value < 0:
                effect = "decreased"
            else:
                effect = "no effect"
            
            explanation += f"-{feature_name},{effect},{abs(shap_value):.2f}; "
        descriptions.append(explanation)

    return descriptions

# with mlflow.start_run(run_name=f"Sklearn-IF Inference", nested=True) as greatgrandparent_run:
run_context = mlflow.start_run(run_name=f"Sklearn-IF Inference", nested=True) if useMlflow == 'True' else contextlib.nullcontext()
with run_context:
    for customerName in customerList:
        for auditLog in auditList:
            # with mlflow.start_run(run_name=f"Base context-{auditLog}-{customerName}", nested=True) as grandparent_run:
            run_context = mlflow.start_run(run_name=f"Base context-{auditLog}_{customerName}") if useMlflow == 'True' else contextlib.nullcontext()
            with run_context:    
                filepath = Path(f'{dataLocation}FS_base_data/{auditLog}/FS_base_data_{auditLog}_{customerName}.parquet')
                if filepath.is_file()==True:
                    ### Replace code - Add encoding script and load data for inference.
                    dataFrame = pd.read_parquet(filepath, engine='pyarrow')
                    dataFrame = dataFrame.astype(str)
                    features = dataFrame.columns.tolist()
                    categoricalColumns = dataFrame.select_dtypes(include=['object']).columns.tolist()
                    encoder = CountFrequencyEncoder(encoding_method='frequency', variables=categoricalColumns, ignore_format=True)
                    #train, test = train_test_split(dataFrame,test_size=0.30)
                    #enc_train = encoder.fit_transform(train) #Troubleshooting
                    #enc_test = encoder.fit_transform(test)
                    enc_dataframe = encoder.fit_transform(dataFrame)
                    ###

                    # Load model
                    print("loading starting")
                    if useMlflow == 'True':
                        model_name = f"Sklearn-IF_{auditLog}_{customerName}_base_model"
                        model_version = "latest"
                        model_uri = f"models:/{model_name}/{model_version}"
                        if_model = mlflow.sklearn.load_model(model_uri)
                    else:
                        dirPath = f"{modelLocation}{auditLog}"
                        if_model = joblib.load(f"{dirPath}/Sklearn-IF_{auditLog}_{customerName}_base_model.pkl")
                    
                    # Start an MLflow run
                    # with mlflow.start_run(run_name=f"Model Prediction", nested=True) as parent_run:                   
                    run_context = mlflow.start_run(run_name=f"Model Prediction", nested=True) if useMlflow == 'True' else contextlib.nullcontext()
                    with run_context: 
                        # Predict anomalies and generate histogram
                        predicted = if_model.predict(enc_dataframe) 
                        predictedScore = if_model.score_samples(enc_dataframe)
                        print("predicting starting")
                        # Combine the scores with the encoded data (for explaination) and unencoded data (for inference)
                        enc_dataframe.reset_index(drop=True,inplace=True)
                        predictedDF = pd.concat([pd.DataFrame(data=predicted,columns=['anomaly']),enc_dataframe],axis=1)
                        predictedDF = predictedDF[predictedDF['anomaly'] > 0]
                        #predictedDF = predictedDF.iloc[:100,1:]
                        if predictedDF.empty:
                            print(f'No anomalies for {auditLog} {customerName}')     
                        else:
                            dataFrame.reset_index(drop=True,inplace=True)
                            finalDF = pd.concat([pd.DataFrame(data=predicted,columns=['anomaly']),dataFrame],axis=1)
                            finalDF = pd.concat([pd.DataFrame(data=predictedScore,columns=['anomaly_score']),finalDF],axis=1)
                            finalDF = finalDF[finalDF['anomaly'] > 0]
                            #finalDF = finalDF.iloc[:100,1:]                  

                            print("explaining starting")    
                            # with mlflow.start_run(run_name=f"Row Explaination", nested=True) as child_run:
                            run_context = mlflow.start_run(run_name=f"Row Explaination", nested=True) if useMlflow == 'True' else contextlib.nullcontext()
                            with run_context: 
                                #Load explaination model
                                if useMlflow == 'True':
                                    model_name = f"Explainer_Sklearn-IF_{auditLog}_{customerName}_base_model"
                                    model_version = "latest"
                                    explainer_uri = f"models:/{model_name}/{model_version}"
                                    explainer = mlflow.shap.load_explainer(explainer_uri)
                                else:                                                          
                                    explainer = joblib.load(f"{dirPath}/Explainer_Sklearn-IF_{auditLog}_{customerName}_base_model.pkl")
                                
                                try: 
                                    shap_values = explainer(predictedDF)
                                    feature_names = shap_values.feature_names
                                
                                    # Generate descriptions for each prediction
                                    explanation = describe_top_shap_values(shap_values.values, predictedDF, feature_names)
                                    finalDF['SHAP_LOCAL'] = explanation
                                
                                    # Global explaination 
                                    dirPath = f"{modelLocation}{auditLog}/varimp"
                                    with open(f'{dirPath}/Var_Imp_{auditLog}_{customerName}_base_model.json', 'r') as file:
                                        var_imp = json.load(file)
                                    finalDF['SHAP_GLOBAL'] = str(var_imp)

                                    ## plot generation - remove later
                                    #shapForcePlot = plt.figure(figsize=(9, 3))
                                    #shap.force_plot(explainer.expected_value, shap_values.values, features=explainer.feature_names, matplotlib=True, show=True)
                                    #mlflow.log_figure(shapForcePlot, f"_{customer}_LocalForcePlot.png")  
                                
                                    #Export Dataset
                                    finalDF['T0083_TYPE'] = f'{auditLog}'
                                    finalDF['DETECTION_METHOD'] = 'unsupervised - base'
                                    finalDF['T0083_TIMESTAMP'] = datetime.datetime.now()
                                    finalDF['REVIEWED'] = 'F'
                                    finalDF = finalDF.loc[:, finalDF.columns.isin(['anomaly','anomaly_score','SAMPLE_NUMBER','EXPERIMENT_NUMBER','TABLE_NAME',
                                            'AUDIT_TIMESTAMP','AUDIT_COUNTER','TRANS_ORDER','AUDIT_TYPE','AUDIT_ACTION',
                                            'TABLE_NAME','TABLE_KEY','USER_NAME','PARENT_TABLE','PARENT_KEY',
                                            'DETECTION_METHOD','T0083_TIMESTAMP','T0083_TYPE','REVIEWED',
                                            'SHAP_LOCAL','SHAP_GLOBAL'])]
                                    name = f'IF_{auditLog}_{customerName}_base_finalDF' 
                                    
                                    dirPath = f'{dataLocation}//Inference//'
                                    Path(dirPath).mkdir(parents=True, exist_ok=True) 
                                    fileName = f'{dirPath}{name}.parquet'
                                    finalDF.to_parquet(fileName) 
                                    if useMlflow == 'True':
                                        mlflow.log_artifact(fileName, artifact_path="inference_results")
                                        #mlflow.log_table(finalDF,f"{name}.json")
                                        print(f"AAR - Load - Base (mlflow) {auditLog} completed for {customerName}")
                                        mlflow.end_run()
                                    else:
                                        print(f"AAR - Load - Base (non-mlflow) {auditLog} completed for {customerName}")
                                except Exception as e:
                                    print(f"Error during explanation phase: {e}")
                                
                else:
                    print("Error - Feature store file not found. Check directory.")

