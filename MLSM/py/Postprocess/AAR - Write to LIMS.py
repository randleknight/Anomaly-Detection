## Load prerequisites libraries
import numpy as np
import pandas as pd
import json
#import pyarrow as pa #remove if not required 
#import pyarrow.parquet as pq #remove if not required 
import glob 
import os
import sys
from cryptography.fernet import Fernet

if len(sys.argv) > 1:
    configAuditSelect = sys.argv[1]
    auditList = configAuditSelect.split(",")  
    dbType = sys.argv[2]
    configCredName = sys.argv[3] 
    customerList = configCredName.split(",")  
    #customerList = [customer.strip() for customer in configCredName.split(",")]
    dataLocation = sys.argv[4]
    configTableLocation = sys.argv[5]
else:
    print("No variable received.")
    sys.exit(1)

for customerName in customerList:
    try:
        with open('credentials.json') as data_file:
            credentials = json.load(data_file)
    except FileNotFoundError:
        print(f"Credential file not found. Cannot proceed.")
        sys.exit(1) 

    def get_decrypted_password(cred_name):
        try:
            if not os.path.exists('secret.key'):
                print(f"Key file not found")
                return ""
                
            with open('secret.key', 'rb') as key_file:
                key = key_file.read()
                
            cipher_suite = Fernet(key)
            encrypted_pwd = credentials[cred_name].get("db_password", "")
            
            if encrypted_pwd:
                return cipher_suite.decrypt(encrypted_pwd.encode()).decode()
            return ""
        except Exception as e:
            print(f"Error decrypting password: {e}")
            return ""    
        
    # Check lims dsn connection to database before proceeding
    def get_lims_connection():
        global customerName
        if dbType == '{PostgreSQL Unicode}':
            import psycopg2
            odbcLimsConnection = psycopg2.connect(f'dbname={credentials[configCredName].get("db_database")} user= {credentials[configCredName].get("db_username")} password = {credentials[configCredName].get("db_password")} host= {credentials[configCredName].get("db_hostname")}')
            odbcLimsConnection.autocommit=True
            return odbcLimsConnection
        
        elif dbType == "oracle":
            import oracledb
            oracledb.init_oracle_client(config.lib_dir)
            odbcLimsConnection = oracledb.connect(user=credentials[customerName].get("db_username"), 
                                                password=credentials[customerName].get("db_password"),
                                                dsn=credentials[customerName].get("db_dsn"))
            odbcLimsConnection.arraysize = 500         

        else:
            import pyodbc
            odbcLimsConnection = pyodbc.connect('DRIVER=' + credentials[customerName].get("db_driver") + ';DATABASE=' + credentials[customerName].get("db_database") +
                                ';UID=' + credentials[customerName].get("db_username") + ';PWD=' + credentials[customerName].get("db_password") +
                                ';SERVER=' + credentials[customerName].get("db_hostname") + ';PORT=' + credentials[customerName].get("db_port") +
                                ';Trusted_Connection=yes')
                    

    if __name__ == '__main__':
        try:  # GET THE CONNECTION OBJECT (ENGINE) FOR THE DATABASE
            get_lims_connection()   
        except Exception as error:
            exit(f"Connection could not be made due to the following error: {error}")

    finalInference = pd.DataFrame()
    for auditLog in auditList:
        print(f"{customerName} {auditLog} read starting")
        #finalInference.iloc[0:0] 
        try:
            baseInferencePath = f'{dataLocation}Inference/IF_{auditLog}_{customerName}_base_finalDF.parquet'
            if os.path.exists(baseInferencePath):
                baseInference = pd.read_parquet(baseInferencePath,engine='pyarrow')
            else:
                print(f'Error Base Inference file {baseInferencePath} not present. No inference made.')
                continue
        
            filterTypeInference = pd.DataFrame()
            FilterTypeFiles = glob.glob(f'{dataLocation}Inference/IF_{auditLog}_*_{customerName}_audit_type_finalDF.parquet')
            
            for filepath in FilterTypeFiles:
                try:
                    tempInference = pd.read_parquet(filepath, engine='pyarrow')
                    filterTypeInference = pd.concat([filterTypeInference, tempInference], ignore_index=True, axis=0)  
                except Exception as e:
                    (f'Error reading {filepath}: {e}.')
                    continue

            auditLogInference = pd.concat([baseInference,filterTypeInference], ignore_index=True, axis=0) 
            finalInference = pd.concat([finalInference,auditLogInference], ignore_index=True, axis=0)

        except Exception as e:
            print(f'Error processing {auditLog}: {e}')
            continue

        # ## Query current log count
        print("Querying started")
        cursor = get_lims_connection().cursor()
        sql_command = "select MAX(COUNTER) from " + configTableLocation
        cursor.execute(sql_command)
        response = cursor.fetchone()
        counter = response[0]

        ## Depending on the value, default either to 1 if empty or add 1 for next insert sequence
        if counter is None:
            id_number = 1
        else:
            id_number = counter + 1

        finalInference.drop_duplicates()
        counterData = (id_number) + np.arange(finalInference.shape[0])
        
        finalInference['counter'] = counterData
        finalInference = finalInference.rename(columns=str.lower)
        finalInference.dropna(how='any',axis=0,inplace=True)

        print(f"{customerName} {auditLog} write starting")
        #print(finalInference) ##troubleshooting
        
        if auditLog == 'Sample':
            #print(type(finalInference)) ##troubleshooting
            sampleNumberArr = finalInference['sample_number'].unique()
            for sampleNumber in sampleNumberArr:
                cursor.execute("update sample set t0083_anomaly = 'T' where sample_number = %s",(sampleNumber,))
            for index, row in finalInference.iterrows():
                #if config.dbType == "postgres":
                cursor.execute("insert into t0083_ar(counter,t0083_type,audit_table_name,audit_table_key,audit_action," 
                            "audit_counter,audit_trans_order,audit_type,audit_timestamp,user_name,audit_main_id," 
                            "detection_method,anomaly,score,shap_local,shap_global,t0083_timestamp,reviewed) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", 
                            (row['counter'],row['t0083_type'],row['table_name'],row['table_key'],row['audit_action'],
                            row['audit_counter'],row['trans_order'],row['audit_type'],row['audit_timestamp'],row['user_name'],row['sample_number'],
                            row['detection_method'],row['anomaly'],row['anomaly_score'],row['shap_local'],row['shap_global'],row['t0083_timestamp'],row['reviewed']))
                #else:
                #    print(f'{config.dbType} table insert is not supported. Please configure')
        elif auditLog == 'Experiment':
            #print(type(finalInference)) ##troubleshooting
            for index, row in finalInference.iterrows():
                #if config.dbType == "postgres":
                cursor.execute("insert into t0083_ar(counter,t0083_type,audit_table_name,audit_table_key,audit_action," 
                            "audit_counter,audit_trans_order,audit_type,audit_timestamp,user_name,audit_main_id,audit_workbook_number,audit_document_number,audit_col_num,audit_row_num," 
                            "detection_method,anomaly,score,shap_local,shap_global,t0083_timestamp,reviewed) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", 
                            (row['counter'],row['t0083_type'],row['table_name'],row['table_key'],row['audit_action'],
                            row['audit_counter'],row['trans_order'],row['audit_type'],row['audit_timestamp'],row['user_name'],row['experiment_number'],row['workbook_number'],row['document_number'],row['col_num'],row['row_num'],
                            row['detection_method'],row['anomaly'],row['anomaly_score'],row['shap_local'],row['shap_global'],row['t0083_timestamp'],row['reviewed']))
        elif auditLog == 'Table':
            #print(type(finalInference)) ##troubleshooting
            for index, row in finalInference.iterrows():
                #if config.dbType == "postgres":
                cursor.execute("insert into t0083_ar(counter,t0083_type,audit_table_name,audit_parent_table,audit_action," 
                            "audit_counter,audit_trans_order,audit_type,audit_timestamp,user_name,audit_main_id," 
                            "detection_method,anomaly,score,shap_local,shap_global,t0083_timestamp,reviewed) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", 
                            (row['counter'],row['t0083_type'],row['table_name'],row['parent_table'],row['audit_action'],
                            row['audit_counter'],row['trans_order'],row['audit_type'],row['audit_timestamp'],row['user_name'],row['parent_key'],
                            row['detection_method'],row['anomaly'],row['anomaly_score'],row['shap_local'],row['shap_global'],row['t0083_timestamp'],row['reviewed']))
        elif auditLog == 'Lot':
            #print(type(finalInference)) ##troubleshooting
            for index, row in finalInference.iterrows():
                cursor.execute("insert into t0083_ar(counter,t0083_type,audit_table_name,audit_table_key,audit_action," 
                            "audit_counter,audit_trans_order,audit_type,audit_timestamp,user_name,audit_main_id," 
                            "detection_method,anomaly,score,shap_local,shap_global,t0083_timestamp,reviewed) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", 
                            (row['counter'],row['t0083_type'],row['table_name'],row['parent_table'],row['audit_action'],
                            row['audit_counter'],row['trans_order'],row['audit_type'],row['audit_timestamp'],row['user_name'],row['parent_key'],
                            row['detection_method'],row['anomaly'],row['anomaly_score'],row['shap_local'],row['shap_global'],row['t0083_timestamp'],row['reviewed']))  
        elif auditLog == 'Batch':
            #print(type(finalInference)) ##troubleshooting
            for index, row in finalInference.iterrows():
                cursor.execute("insert into t0083_ar(counter,t0083_type,audit_table_name,audit_table_key,audit_action," 
                            "audit_counter,audit_trans_order,audit_type,audit_timestamp,user_name,audit_main_id," 
                            "detection_method,anomaly,score,shap_local,shap_global,t0083_timestamp,reviewed) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", 
                            (row['counter'],row['t0083_type'],row['table_name'],row['table_key'],row['audit_action'],
                            row['audit_counter'],row['trans_order'],row['audit_type'],row['audit_timestamp'],row['user_name'],row['batch_name'],
                            row['detection_method'],row['anomaly'],row['anomaly_score'],row['shap_local'],row['shap_global'],row['t0083_timestamp'],row['reviewed']))
        elif auditLog == 'Project':
            #print(type(finalInference)) ##troubleshooting
            for index, row in finalInference.iterrows():
                cursor.execute("insert into t0083_ar(counter,t0083_type,audit_table_name,audit_parent_table,audit_action," 
                            "audit_counter,audit_trans_order,audit_type,audit_timestamp,user_name,audit_main_id," 
                            "detection_method,anomaly,score,shap_local,shap_global,t0083_timestamp,reviewed) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", 
                            (row['counter'],row['t0083_type'],row['table_name'],row['table_key'],row['audit_action'],
                            row['audit_counter'],row['trans_order'],row['audit_type'],row['audit_timestamp'],row['user_name'],row['project_name'],
                            row['detection_method'],row['anomaly'],row['anomaly_score'],row['shap_local'],row['shap_global'],row['t0083_timestamp'],row['reviewed']))    
        elif auditLog == 'Protocol':
            #print(type(finalInference)) ##troubleshooting
            for index, row in finalInference.iterrows():  
                cursor.execute("insert into t0083_ar(counter,t0083_type,audit_table_name,audit_parent_table,audit_action," 
                            "audit_counter,audit_trans_order,audit_type,audit_timestamp,user_name,audit_main_id," 
                            "detection_method,anomaly,score,shap_local,shap_global,t0083_timestamp,reviewed) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", 
                            (row['counter'],row['t0083_type'],row['table_name'],row['parent_table'],row['audit_action'],
                            row['audit_counter'],row['trans_order'],row['audit_type'],row['audit_timestamp'],row['user_name'],row['parent_key'],
                            row['detection_method'],row['anomaly'],row['anomaly_score'],row['shap_local'],row['shap_global'],row['t0083_timestamp'],row['reviewed']))
        elif auditLog == 'Stor_loc':  
            #print(type(finalInference)) ##troubleshooting
            for index, row in finalInference.iterrows(): 
                cursor.execute("insert into t0083_ar(counter,t0083_type,audit_table_name,audit_parent_table,audit_action," 
                            "audit_counter,audit_trans_order,audit_type,audit_timestamp,user_name,audit_main_id," 
                            "detection_method,anomaly,score,shap_local,shap_global,t0083_timestamp,reviewed) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", 
                            (row['counter'],row['t0083_type'],row['table_name'],row['parent_table'],row['audit_action'],
                            row['audit_counter'],row['trans_order'],row['audit_type'],row['audit_timestamp'],row['user_name'],row['location_number'],
                            row['detection_method'],row['anomaly'],row['anomaly_score'],row['shap_local'],row['shap_global'],row['t0083_timestamp'],row['reviewed']))   
        elif auditLog == 'Reports':  
            #print(type(finalInference)) ##troubleshooting
            for index, row in finalInference.iterrows():  
                cursor.execute("insert into t0083_ar(counter,t0083_type,audit_table_name,audit_parent_table,audit_action," 
                            "audit_counter,audit_trans_order,audit_type,audit_timestamp,user_name,audit_main_id," 
                            "detection_method,anomaly,score,shap_local,shap_global,t0083_timestamp,reviewed) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", 
                            (row['counter'],row['t0083_type'],row['table_name'],row['parent_table'],row['audit_action'],
                            row['audit_counter'],row['trans_order'],row['audit_type'],row['audit_timestamp'],row['user_name'],row['report_number'],
                            row['detection_method'],row['anomaly'],row['anomaly_score'],row['shap_local'],row['shap_global'],row['t0083_timestamp'],row['reviewed']))
        elif auditLog == 'Investigation':  
            #print(type(finalInference)) ##troubleshooting
            for index, row in finalInference.iterrows(): 
                cursor.execute("insert into t0083_ar(counter,t0083_type,audit_table_name,audit_parent_table,audit_action," 
                            "audit_counter,audit_trans_order,audit_type,audit_timestamp,user_name,audit_main_id," 
                            "detection_method,anomaly,score,shap_local,shap_global,t0083_timestamp,reviewed) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", 
                            (row['counter'],row['t0083_type'],row['table_name'],row['parent_table'],row['audit_action'],
                            row['audit_counter'],row['trans_order'],row['audit_type'],row['audit_timestamp'],row['user_name'],row['investigation'],
                            row['detection_method'],row['anomaly'],row['anomaly_score'],row['shap_local'],row['shap_global'],row['t0083_timestamp'],row['reviewed']))  
        elif auditLog == 'Order':  
            #print(type(finalInference)) ##troubleshooting
            for index, row in finalInference.iterrows():   
                cursor.execute("insert into t0083_ar(counter,t0083_type,audit_table_name,audit_parent_table,audit_action," 
                            "audit_counter,audit_trans_order,audit_type,audit_timestamp,user_name,audit_main_id," 
                            "detection_method,anomaly,score,shap_local,shap_global,t0083_timestamp,reviewed) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", 
                            (row['counter'],row['t0083_type'],row['table_name'],row['parent_table'],row['audit_action'],
                            row['audit_counter'],row['trans_order'],row['audit_type'],row['audit_timestamp'],row['user_name'],row['order_num'],
                            row['detection_method'],row['anomaly'],row['anomaly_score'],row['shap_local'],row['shap_global'],row['t0083_timestamp'],row['reviewed'])) 
        elif auditLog == 'Formulation':  
            #print(type(finalInference)) ##troubleshooting
            for index, row in finalInference.iterrows():    
                cursor.execute("insert into t0083_ar(counter,t0083_type,audit_table_name,audit_parent_table,audit_action," 
                            "audit_counter,audit_trans_order,audit_type,audit_timestamp,user_name,audit_main_id," 
                            "detection_method,anomaly,score,shap_local,shap_global,t0083_timestamp,reviewed) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", 
                            (row['counter'],row['t0083_type'],row['table_name'],row['parent_table'],row['audit_action'],
                            row['audit_counter'],row['trans_order'],row['audit_type'],row['audit_timestamp'],row['user_name'],row['formulation'],
                            row['detection_method'],row['anomaly'],row['anomaly_score'],row['shap_local'],row['shap_global'],row['t0083_timestamp'],row['reviewed']))
        elif auditLog == 'Charge':  
            #print(type(finalInference)) ##troubleshooting
            for index, row in finalInference.iterrows():   
                cursor.execute("insert into t0083_ar(counter,t0083_type,audit_table_name,audit_parent_table,audit_action," 
                            "audit_counter,audit_trans_order,audit_type,audit_timestamp,user_name,audit_main_id," 
                            "detection_method,anomaly,score,shap_local,shap_global,t0083_timestamp,reviewed) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", 
                            (row['counter'],row['t0083_type'],row['table_name'],row['parent_table'],row['audit_action'],
                            row['audit_counter'],row['trans_order'],row['audit_type'],row['audit_timestamp'],row['user_name'],row['parent_key'],
                            row['detection_method'],row['anomaly'],row['anomaly_score'],row['shap_local'],row['shap_global'],row['t0083_timestamp'],row['reviewed']))         
        else: 
            print(f'{auditLog} not completed.')       

    print(f"{customerName} {auditLog} write completed")

cursor.close()
get_lims_connection.close()
