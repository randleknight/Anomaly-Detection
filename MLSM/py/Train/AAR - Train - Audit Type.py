from pathlib import Path
import pandas as pd
import numpy as np
from feature_engine.encoding import CountFrequencyEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import shap
import os
import joblib
import json
import contextlib
import sys

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

# define Mlflow env settings
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

def list_folders(path):
    """Lists all folders in the given directory."""

    folders = []
    for entry in os.listdir(path):
        if os.path.isdir(os.path.join(path, entry)):
            folders.append(entry)
    return folders

# with mlflow.start_run(run_name=f"Sklearn-IF Model", nested=True) as greatgrandparent_run:
run_context = mlflow.start_run(run_name=f"Sklearn-IF Model", nested=True) if useMlflow == 'True' else contextlib.nullcontext()
with run_context:
    for customerName in customerList: 
        for auditLog in auditList:  
            
            FilterTypeList = list_folders(f'{dataLocation}FS_audit_type_table_name/{auditLog}/') 
            for FilterType in FilterTypeList:
                # with mlflow.start_run(run_name=f"Audit Type context-{auditLog}-{customerName}", nested=True) as grandparent_run:
                run_context = mlflow.start_run(run_name=f"Audit Type {FilterType} context-{auditLog}_{customerName}", nested=True) if useMlflow == 'True' else contextlib.nullcontext()
                with run_context:   
                    filepath = Path(f'{dataLocation}FS_audit_type_table_name/{auditLog}/{FilterType}/FS_audit_type_table_name_{auditLog}_{FilterType}_{customerName}.parquet') 
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
                        numEst = 250
                        contamFloat = 0.001
                        #sampleSize = 100
                        random = np.random.RandomState(1234)

                        if useMlflow == 'True':
                            mlflow.log_param("n_estimators", numEst)
                            mlflow.log_param("contamination", contamFloat)

                        # Initialize the model
                        if_model = IsolationForest(random_state=random, n_estimators=numEst, contamination=contamFloat) #, max_samples=sampleSize, verbose=1

                        # Start an MLflow run
                        # with mlflow.start_run(run_name=f"Model Evaluation", nested=True) as parent_run:
                        run_context = mlflow.start_run(run_name=f"Model Evaluation", nested=True) if useMlflow == 'True' else contextlib.nullcontext()
                        with run_context: 
                            # Log model parameters and train
                            #mlflow.log_param("ntrees", numEst)
                            #mlflow.log_param("sample_size", sampleSize)
                            if_model.fit(enc_train)        
                            
                            # Predict anomalies and generate histogram
                            predicted = if_model.predict(enc_test)
                            predictedScore = if_model.score_samples(enc_test)            
                            
                            # Combine the scores with the original data
                            predictedDF = pd.concat([pd.DataFrame(data=predicted,columns=['anomaly']),test],axis=1)
                            predictedDF = pd.concat([pd.DataFrame(data=predictedScore,columns=['anomaly_score']),predictedDF],axis=1)
                            predictedDF = predictedDF[predictedDF['anomaly'] > 0]
                            
                            if predictedDF.empty:
                                print(f'No anomalies for {auditLog} {FilterType} {customerName}')     
                            else:
                                #Export Dataset     
                                # with mlflow.start_run(run_name=f"Explain_model", nested=True) as child_run:                    
                                run_context = mlflow.start_run(run_name=f"Model Explaination", nested=True) if useMlflow == 'True' else contextlib.nullcontext()                              
                                with run_context:
                                    #Train explaination model
                                    print('explaining started')

                                    explainer = shap.TreeExplainer(if_model, feature_names=if_model.feature_names_in_) 
                                    shap_values = explainer.shap_values(enc_train)

                                    ## plot generation
                                    shapBarplot = plt.figure(figsize=(9, 3))
                                    shap.summary_plot(shap_values, features=explainer.feature_names , plot_type="bar", max_display=15)
                                    plt.close(shapBarplot)

                                    # Get the absolute mean SHAP values for feature importance
                                    mean_abs_shap_values = np.abs(shap_values).mean(axis=0)

                                    # Sort features based on their importance
                                    sorted_indices = np.argsort(mean_abs_shap_values)[-10:]  # Get indices of top 10 features
                                    top_10_shap_values = shap_values[:, sorted_indices]
                                    top_10_feature_names = [explainer.feature_names[i] for i in sorted_indices]
                                    top_10_mean_shap_values = mean_abs_shap_values[sorted_indices]

                                    data = {}
                                    for i in range(len(top_10_feature_names)):
                                        data[top_10_feature_names[i]] = top_10_mean_shap_values[i]

                                    json_data = json.dumps(data)
                                    #mlflow.log_table(data,f"Var_Imp_{auditLog}_{FilterType}_{customerName}_audit_type_model.json")
                                    dirPath = f"{modelLocation}{auditLog}/varimp"
                                    Path(dirPath).mkdir(parents=True, exist_ok=True) 
                                    with open(f'{dirPath}/Var_Imp_{auditLog}_{FilterType}_{customerName}_audit_type_model.json', 'w') as outfile:
                                        outfile.write(json_data)  

                                    # Export models
                                    print('exporting started')
                                    if useMlflow == 'True':
                                        mlflow.sklearn.log_model(sk_model=if_model, artifact_path="model", registered_model_name=f"Sklearn-IF_{auditLog}_{FilterType}_{customerName}_audit_type")
                                        # Issue logging explainer bug 
                                        #logExplainerModel = mlflow.shap.log_explainer(explainer,
                                        #                                            artifact_path=f"Explainer_Sklearn-IF_{auditLog}_{customerName}_base_model",
                                        #                                            serialize_model_using_mlflow=False,
                                        #                                            registered_model_name=f"Explainer_Sklearn-IF_{auditLog}_{customerName}_base_model")
                                        mlflow.log_figure(shapBarplot, f"{customerName}_{FilterType}_GlobalBarPlot.png")                                       
                                        mlflow.log_dict(data, f"Var_Imp_{auditLog}_{FilterType}_{customerName}_audit_type_model.json")
                                        print(f"AAR - Train - Audit Type (mlflow) {FilterType} completed for {customerName}")
                                        mlflow.end_run()
                                    else:    
                                        dirPath = f"{modelLocation}{auditLog}"
                                        Path(dirPath).mkdir(parents=True, exist_ok=True) 
                                        joblib.dump(if_model,f"{dirPath}/Sklearn-IF_{auditLog}_{FilterType}_{customerName}_audit_type_model.pkl")
                                        joblib.dump(explainer,f"{dirPath}/Explainer_Sklearn-IF_{auditLog}_{FilterType}_{customerName}_audit_type_model.pkl")
                                        print(f"AAR - Train - Audit Type (non-mlflow) {FilterType} completed for {customerName}")
                                        
                    else:
                        print("Error - Feature store file not found. Check directory.")        
                    
