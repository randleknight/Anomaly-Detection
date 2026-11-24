import time
import json
import subprocess
import sys
import shlex
import os

global config_data
global configAuditSelect 
global configDbSelect
global message

def run_process(config_data, process_type):
    # Hardcoded to maintain compatibility with downstream scripts
    configLogLocation = "log//"
    configDataLocation = "data//"
    configModelLocation = "model//"
    
    # Retrieved from the front-end
    configTableLocation = config_data.get('table_location')
    configTrainingSpan = config_data.get('training_span')
    configInferenceSpan = config_data.get('inference_span')
    configCredSelect = config_data.get('cred_name')

    # other db type handling in future build
    configDbSelect = config_data.get('db_driver')
    
    # MLflow Variables Extracted from frontend
    configUseMlflow = config_data.get('use_mlflow', 'False')
    configMlflowExperiment = config_data.get('mlflow_experiment', 'Default_Experiment')
    configMlflowHost = config_data.get('mlflow_host', '')
    configMlflowPort = config_data.get('mlflow_port', '')
    configMlflowUri = f"{configMlflowHost.rstrip('/')}:{configMlflowPort}"

    # read all audit handling in future build
    if config_data.get('audit_select') == "1": #Fix pass List as string
        configAuditSelect = "Sample,Experiment,Table,Invest,Lot,Project,Protocol,Stor_loc,Batch,Charge,Order,Reports"
    elif config_data.get('audit_select') == "2":
        configAuditSelect = "Sample"
    elif config_data.get('audit_select') == "3":
        configAuditSelect = "Experiment"
    elif config_data.get('audit_select') == "4":
        configAuditSelect = "Investigation"
    elif config_data.get('audit_select') == "5":
        configAuditSelect = "Lot"
    elif config_data.get('audit_select') == "6":
        configAuditSelect = "Project"
    elif config_data.get('audit_select') == "7":
        configAuditSelect = "Protocol"
    elif config_data.get('audit_select') == "8":
        configAuditSelect = "Stor_loc"
    elif config_data.get('audit_select') == "9":
        configAuditSelect = "Batch"
    elif config_data.get('audit_select') == "10":
        configAuditSelect = "Charge"
    elif config_data.get('audit_select') == "11":
        configAuditSelect = "Order"
    elif config_data.get('audit_select') == "12":
        configAuditSelect = "Reports"
    elif config_data.get('audit_select') == "13":
        configAuditSelect = "Table"

    try:
        if process_type == 'Troubleshooting':
            script_path = 'py/Troubleshoot/Troubleshoot.py'
            try: 
                output = subprocess.run([sys.executable, script_path, 
                                          configDbSelect, configCredSelect, configMlflowUri]
                                        ,capture_output=True,text=True,check=True)
                print(output.stdout)
            except subprocess.CalledProcessError as e:
                print(f"Error executing script: {e}")
                print(f"Stderr: {e.stderr}")

            message = f"No database connection issues found."           

        if process_type == 'Training/Retraining':
            script_path = "py/Preprocess/AAR - 1 - Raw Data Extraction.py"
            try: 
                output = subprocess.run([sys.executable, script_path, 
                                         configAuditSelect, configDbSelect, configCredSelect, configDataLocation, 
                                         str(configTrainingSpan), str(configInferenceSpan), process_type]
                                        ,capture_output=True,text=True,check=True)
                print(output.stdout)
            except subprocess.CalledProcessError as e:
                print(f"Error executing script: {e}")
                print(f"Stderr: {e.stderr}")

            script_path = "py/Preprocess/AAR - 2 - Feature Engineering Base Data.py"
            try: 
                output = subprocess.run([sys.executable, script_path,
                                         configAuditSelect, configDbSelect, configCredSelect, configDataLocation]
                                         ,capture_output=True,text=True,check=True)
                print(output.stdout)
            except subprocess.CalledProcessError as e:
                print(f"Error executing script: {e}")
                print(f"Stderr: {e.stderr}")
            
            script_path = "py/Preprocess/AAR - 3 - Feature Engineering Audit Type.py"
            try: 
                output = subprocess.run([sys.executable, script_path,
                            configAuditSelect, configDbSelect, configCredSelect, configDataLocation]
                            ,capture_output=True,text=True,check=True)
                print(output.stdout)

            except subprocess.CalledProcessError as e:
                print(f"Error executing script: {e}")
                print(f"Stderr: {e.stderr}")    

            script_path = "py/Train/AAR - Train - Audit Type.py"
            try: 
                output = subprocess.run([sys.executable, script_path,
                                        configAuditSelect, configDbSelect, configCredSelect, configDataLocation, configModelLocation
                                        ,configUseMlflow, configMlflowUri, configMlflowExperiment]
                                        ,capture_output=True,text=True,check=True)
                print(output.stdout)
            
            except subprocess.CalledProcessError as e:
                print(f"Error executing script: {e}")
                print(f"Stderr: {e.stderr}")      
                                        
            script_path = "py/Train/AAR - Train - Base.py"
            try: 
                output = subprocess.run([sys.executable, script_path,
                            configAuditSelect, configDbSelect, configCredSelect, configDataLocation, configModelLocation
                            ,configUseMlflow, configMlflowUri, configMlflowExperiment]
                            ,capture_output=True,text=True,check=True)
                print(output.stdout)

            except subprocess.CalledProcessError as e:
                print(f"Error executing script: {e}")
                print(f"Stderr: {e.stderr}")    

            if config_data.get('grid_search') == 'True':
                message = "Grid Search enabled. Models trained and hyper-parameters optimized."
            else:
                message = "Model training completed."
    
            #configModelLocation = config_data.get('model_location')
            message += f"\nModels saved successfully " #to: {configModelLocation}"

        elif process_type == 'Inference':           
            script_path = "py/Preprocess/AAR - 1 - Raw Data Extraction.py"
            
            try: 
                output = subprocess.run([sys.executable, script_path, 
                                         configAuditSelect, configDbSelect, configCredSelect, configDataLocation, 
                                         str(configTrainingSpan), str(configInferenceSpan), process_type]
                                        ,capture_output=True,text=True,check=True)
                print(output.stdout)
            except subprocess.CalledProcessError as e:
                print(f"Error executing script: {e}")
                print(f"Stderr: {e.stderr}")

            script_path = "py/Preprocess/AAR - 2 - Feature Engineering Base Data.py"
            try: 
                output = subprocess.run([sys.executable, script_path,
                                         configAuditSelect, configDbSelect, configCredSelect, configDataLocation]
                                         ,capture_output=True,text=True,check=True)
                print(output.stdout)
            except subprocess.CalledProcessError as e:
                print(f"Error executing script: {e}")
                print(f"Stderr: {e.stderr}")
            
            script_path = "py/Preprocess/AAR - 3 - Feature Engineering Audit Type.py"
            try: 
                output = subprocess.run([sys.executable, script_path,
                            configAuditSelect, configDbSelect, configCredSelect, configDataLocation]
                            ,capture_output=True,text=True,check=True)
                print(output.stdout)

            except subprocess.CalledProcessError as e:
                print(f"Error executing script: {e}")
                print(f"Stderr: {e.stderr}")    

            script_path = "py/Load/AAR - Load - Base.py"
            try: 
                output = subprocess.run([sys.executable, script_path,
                                         configAuditSelect, configDbSelect, configCredSelect, configDataLocation, configModelLocation
                                         ,configUseMlflow, configMlflowUri, configMlflowExperiment]
                                        ,capture_output=True,text=True,check=True)
                print(output.stdout)

            except subprocess.CalledProcessError as e:
                print(f"Error executing script: {e}")
                print(f"Stderr: {e.stderr}")  

            script_path = "py/Load/AAR - Load - Audit Type.py"
            try: 
                output = subprocess.run([sys.executable, script_path,
                                         configAuditSelect, configDbSelect, configCredSelect, configDataLocation, configModelLocation
                                         ,configUseMlflow, configMlflowUri, configMlflowExperiment]
                                        ,capture_output=True,text=True,check=True)
                print(output.stdout)

            except subprocess.CalledProcessError as e:
                print(f"Error executing script: {e}")
                print(f"Stderr: {e.stderr}")  

            script_path = "py/Postprocess/AAR - Write to LIMS.py"
            try: 
                output = subprocess.run([sys.executable, script_path,
                                         configAuditSelect, configDbSelect, configCredSelect, configDataLocation, configTableLocation]
                                        ,capture_output=True,text=True,check=True)
                print(output.stdout)

            except subprocess.CalledProcessError as e:
                print(f"Error executing script: {e}")
                print(f"Stderr: {e.stderr}") 

            message = f"Predictions generated and saved to table: {configTableLocation}"  
         
        return True, message
    
    except Exception as e:
        error_message = f"An error occurred during {process_type}: {str(e)}"
        print(f"[{process_type}] Error: {error_message}")
        return False, error_message
 
# #Uncomment for Testing
# if __name__ == '__main__':
#     
#     data = {"script": "py/Troubleshoot/Troubleshoot.py",
#         "db_driver": "postgres", 
#         "cred_name": "Default", 
#         "table_location": "T0083_AR", 
#         "data_location": ".//data//",
#         "model_location": ".//model//",
#         "log_location": ".//log//"}
#     success, message = run_process(data, 'Troubleshooting')
#     print(f"Test result: Success={success}, Message={message}")