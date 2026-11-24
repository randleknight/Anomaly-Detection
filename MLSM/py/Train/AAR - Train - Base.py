from pathlib import Path
import pandas as pd
import numpy as np
from feature_engine.encoding import CountFrequencyEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import shap
import joblib 
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
    # auditList = ['Sample']
    # dbType = 'postgres'
    # customerList = ['Default']
    # dataLocation = './/data/'
    # modelLocation = './/model/'
    # useMlflow = 'False'
    # mlflowUri = 'http://localhost:5000'
    # mlflowExperiment = 'Default_Experiment'

# # define Mlflow env. settings
if useMlflow == 'True':
    import mlflow
    import mlflow.sklearn
    #import mlflow.shap
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

# with mlflow.start_run(run_name=f"Sklearn-IF Model", nested=True) as greatgrandparent_run:
run_context = mlflow.start_run(run_name=f"Sklearn-IF Model", nested=True) if useMlflow == 'True' else contextlib.nullcontext()
with run_context:
    for customerName in customerList:
        for auditLog in auditList:
            # with mlflow.start_run(run_name=f"Base context-{auditLog}-{customerName}", nested=True) as grandparent_run:
            run_context = mlflow.start_run(run_name=f"Base context-{auditLog}_{customerName}", nested=True) if useMlflow == 'True' else contextlib.nullcontext()
            with run_context:
                filepath = Path(f'{dataLocation}FS_base_data/{auditLog}/FS_base_data_{auditLog}_{customerName}.parquet')
                if filepath.is_file()==True:
                    dataFrame = pd.read_parquet(filepath, engine='pyarrow')
                    dataFrame = dataFrame.astype(str)
                    features = dataFrame.columns.tolist()
                    categoricalColumns = dataFrame.select_dtypes(include=['object']).columns.tolist()
                    encoder = CountFrequencyEncoder(encoding_method='frequency', variables=categoricalColumns, ignore_format=True)
                    train, test = train_test_split(dataFrame,test_size=0.30)
                    enc_train = encoder.fit_transform(train) #Troubleshooting
                    enc_test = encoder.fit_transform(test)

                    # Define model parameters
                    numEst = 256
                    contamFloat = 0.001
                    #sampleSize = 100
                    random = np.random.RandomState(1234)
                    if useMlflow == 'True':
                        mlflow.log_param("n_estimators", numEst)
                        mlflow.log_param("contamination", contamFloat)

                    # Initialize the model
                    print("Training started!")
                    if_model = IsolationForest(random_state=random, n_estimators=numEst, contamination=contamFloat) #, max_samples=sampleSize, verbose=1

                    #with mlflow.start_run(run_name=f"Model Evaluation", nested=True) as parent_run:
                    run_context = mlflow.start_run(run_name=f"Model Evaluation", nested=True) if useMlflow == 'True' else contextlib.nullcontext()
                    with run_context:
                        # Log model parameters and train
                        # mlflow.log_param("n_estimators", numEst)
                        # mlflow.log_param("sample_size", sampleSize)
                        if_model.fit(enc_train) 

                        ##Export model
                        #logModel = mlflow.sklearn.log_model(sk_model=if_model,artifact_path=f"Sklearn-IF_{auditLog}_{customerName}_base_model",registered_model_name=f"Sklearn-IF_{auditLog}_{customer}_base_model") 
                        #dirPath = f"{config.model_location}{auditLog}"
                        #Path(dirPath).mkdir(parents=True, exist_ok=True) 
                        #joblib.dump(if_model,f"{dirPath}/Sklearn-IF_{auditLog}_{customerName}_base_model.pkl")
                        
                        # Predict anomalies and generate histogram
                        predicted = if_model.predict(enc_test) 
                        predictedScore = if_model.score_samples(enc_test)
                        
                        # Combine the scores with the original data
                        predictedDF = pd.concat([pd.DataFrame(data=predicted,columns=['anomaly']),test],axis=1)
                        predictedDF = pd.concat([pd.DataFrame(data=predictedScore,columns=['anomaly_score']),predictedDF],axis=1)
                        predictedDF = predictedDF[predictedDF['anomaly'] > 0]

                        if predictedDF.empty:
                            print(f'No anomalies for {auditLog} {customerName}')     
                        else:
                            #Export Dataset
                            # mlflow.log_table(predictedDF,"predictedDF.json")
                            # with mlflow.start_run(run_name=f"Model Explaination", nested=True) as child_run:
                            run_context = mlflow.start_run(run_name=f"Model Explaination", nested=True) if useMlflow == 'True' else contextlib.nullcontext()
                            with run_context:
                                #Train explaination model
                                explainer = shap.TreeExplainer(if_model, feature_names=if_model.feature_names_in_) 
                                shap_values = explainer.shap_values(enc_train)

                                ## plot generation
                                shapBarplot = plt.figure(figsize=(9, 3))
                                plt.close(shapBarplot)
                                shap.summary_plot(shap_values,features=explainer.feature_names,max_display=12,plot_type="bar")
                                # mlflow.log_figure(shapBarplot, f"{customerName}_GlobalBarPlot.png")  

                                # Get the absolute mean SHAP values for feature importance
                                mean_abs_shap_values = np.abs(shap_values).mean(axis=0)

                                # Sort features based on their importance
                                sorted_indices = np.argsort(mean_abs_shap_values)[::-1][:10]  # Get indices of top 10 features
                                top_10_shap_values = shap_values[:, sorted_indices]
                                top_10_feature_names = [explainer.feature_names[i] for i in sorted_indices]
                                top_10_mean_shap_values = mean_abs_shap_values[sorted_indices]

                                data = {}
                                for i in range(len(top_10_feature_names)):
                                    data[top_10_feature_names[i]] = top_10_mean_shap_values[i]

                                json_data = json.dumps(data)
                                #mlflow.log_table(data,f"Var_Imp_{auditLog}_{customerName}_base_model.json")
                                dirPath = f"{modelLocation}{auditLog}/varimp"
                                Path(dirPath).mkdir(parents=True, exist_ok=True) 
                                with open(f'{dirPath}/Var_Imp_{auditLog}_{customerName}_base_model.json', 'w') as outfile:
                                    outfile.write(json_data)   

                                #Export Explainer model
                                dirPath = f"{modelLocation}{auditLog}"
                                Path(dirPath).mkdir(parents=True, exist_ok=True) 
                                joblib.dump(explainer,f"{dirPath}/Explainer_Sklearn-IF_{auditLog}_{customerName}_base_model.pkl")
                                joblib.dump(if_model,f"{dirPath}/Sklearn-IF_{auditLog}_{customerName}_base_model.pkl")
                                
                                if useMlflow == 'True':
                                    mlflow.sklearn.log_model(sk_model=if_model, artifact_path="model", registered_model_name=f"Sklearn-IF_{auditLog}_{customerName}_base")
                                    # Issue logging explainer bug 
                                    #logExplainerModel = mlflow.shap.log_explainer(explainer,
                                    #                                            artifact_path=f"Explainer_Sklearn-IF_{auditLog}_{customerName}_base_model",
                                    #                                            serialize_model_using_mlflow=False,
                                    #                                            registered_model_name=f"Explainer_Sklearn-IF_{auditLog}_{customerName}_base_model")
                                    mlflow.log_figure(shapBarplot, f"{customerName}_GlobalBarPlot.png")
                                    mlflow.log_dict(data, f"Var_Imp_{auditLog}_{customerName}_base_model.json")
                                    print(f"Train - Base (mlflow) {auditLog} completed for {customerName}")
                                    mlflow.end_run()
                                else:
                                    print(f"Train - Base (non-mlflow) {auditLog} completed for {customerName}")
                                
                else:
                    print("Error - Feature store file not found. Check directory.")

