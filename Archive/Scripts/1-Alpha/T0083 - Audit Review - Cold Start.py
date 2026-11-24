# ====================================================================================
# Abstract:
#        	Cold Start 
# Inputs:
#   Context:
#   Variables (inc global):
#
# Outputs:
#   Context:
#   Variables:
#
# Change History:
# Date			By					Reason
# 06-May-2023	Randle Knight		Initial Version
#  ====================================================================================

# Execution timing
import time
from datetime import datetime
startTime = time.time()

## Load h2o and prerequisites libraries
import h2o
import numpy as np
import pandas as pd
from h2o.estimators import H2OExtendedIsolationForestEstimator
import warnings
import json

import config

# Initialise H2o Server
h2o.init(nthreads=-1, min_mem_size= config.min_memory, max_mem_size= config.max_memory, enable_assertions=False)
warnings.filterwarnings('ignore')
# Clear h2o memory
h2o.remove_all()

# Check lims dsn connection to database before proceeding
def get_lims_connection():
    global odbcLimsConnection
    if config.dbType == "oracle":
        import oracledb
        odbcLimsConnection = oracledb.connect(user=config.limsdb_username, password=config.limsdb_password,dsn=config.limsdb_dsn)
        
    elif config.dbType == "postgres":
        import psycopg
        odbcLimsConnection = psycopg.connect('dbname=' + config.limsdb_database + ' user=' + config.limsdb_username + 
                                             ' password=' + config.limsdb_password + ' host=' + config.limsdb_hostname)
    
    else:
        import pyodbc
        odbcLimsConnection = pyodbc.connect('DRIVER=' + config.limsdb_driver + ';DATABASE=' + config.limsdb_database +
                                        ';UID=' + config.limsdb_username + ';PWD=' + config.limsdb_password +
                                        ';SERVER=' + config.limsdb_hostname + ';PORT=' + config.limsdb_port +
                                        ';Trusted_Connection=yes')
    return odbcLimsConnection

if __name__ == '__main__':
    try:  # GET THE CONNECTION OBJECT (ENGINE) FOR THE DATABASE
        get_lims_connection()
    except Exception as error:
        exit("Connection could not be made due to the following error: \n", error)


# Check mlar dsn connection to database before proceeding
def get_mlar_connection():
    global odbcMlarConnection
    if config.dbType == "oracle":
        import oracledb
        odbcMlarConnection = oracledb.connect(user=config.mlardb_username, password=config.mlardb_password,dsn=config.mlardb_dsn)
    elif config.dbType == "postgres":
        import psycopg
        odbcMlarConnection = psycopg.connect('dbname=' + config.mlardb_database + ' user=' + config.mlardb_username +
                                             ' password=' + config.mlardb_password + ' host=' + config.mlardb_hostname)        
    else:
        import pyodbc
        odbcMlarConnection = pyodbc.connect('DRIVER=' + config.mlardb_driver + ';DATABASE=' + config.mlardb_database +
                                        ';UID=' + config.mlardb_username + ';PWD=' + config.mlardb_password +
                                        ';SERVER=' + config.mlardb_hostname + ';PORT=' + config.mlardb_port +
                                        ';Trusted_Connection=yes')
    return odbcMlarConnection

if __name__ == '__main__':
    try:
        # GET THE CONNECTION OBJECT (ENGINE) FOR THE DATABASE
        get_mlar_connection()
    except Exception as error:
        exit("Connection could not be made due to the following error: \n", error)

# Start and Finish Logs - MOVE TO CONFIG
sql_command = "select MIN(AUDIT_TIMESTAMP),MAX(AUDIT_TIMESTAMP) from SAMPLE_AUDIT_DECRYPT"
with odbcLimsConnection.cursor() as lims_cursor_fetch_one:
    lims_cursor_fetch_one.execute(sql_command)
    response = lims_cursor_fetch_one.fetchone()
startLog = response[0]
endLog = response[1]

# Execution timing
endTime = time.time()
elapsedTime = endTime - startTime
now = datetime.now()
logEntry = "DateTime: " + str(now) + ", Start: " + str(startLog) + ", End: " + str(endLog) + ", Sample Audit, " \
          "Prerequisite Load End Time = " + str(elapsedTime)   
with open ("executionlog.txt", "a") as writer:
    writer.write('\n'+logEntry)
    writer.close()
    print(logEntry)

# Query data and load into dataframe
column_names = ['MLAR_TYPE', 'AUDIT_ACTION', 'AUDIT_TIMESTAMP', 'AUDIT_TYPE',
                'AUDIT_COUNTER', 'TRANS_ORDER', 'TABLE_NAME', 'TABLE_KEY', 'REASON',
                'RECORD_SIGNED', 'USER_NAME',  'USER_ROLE', 'SAMPLE_NUMBER', 'T0083',
                'T0083', 'TRANS_STRING']
sql_command = "select 'sample_audit_log',action,audit_timestamp,audit_type,counter,trans_order,table_name,table_key," \
              "reason,record_signed,user_name,user_role,sample_number,t0083,t0083_severity,trans_string" \
              " from SAMPLE_AUDIT_DECRYPT where audit_timestamp " \
              "between '" + str(startLog) + "' and '" + str(endLog) + "'"
dataFrame = pd.read_sql(sql_command, odbcMlarConnection, columns=column_names)
dataFrame = dataFrame.astype({'trans_string': str})

# Execution timing
endTime = time.time()
elapsedTime = endTime - startTime
now = datetime.now()
logEntry = "DateTime: " + str(now) + ", Start: " + str(startLog) + ", End: " + str(endLog) + ", Sample Audit, " \
           "Query end Time = " + str(elapsedTime)
with open ("executionlog.txt", "a") as writer:
    writer.write('\n'+logEntry)
    writer.close()
    print(logEntry)

# Retrieve datatypes for training Dataframe
# "string" | "float" | "real" | "double" | "int" | "long" | "numeric" | "categorical" | "factor" | "enum" | "time"
sql_command = "select distinct FIELD_NAME, CASE WHEN FIELD_NAME = 'T_MLAR_SEVERITY' THEN 'categorical' WHEN DATA_TYPE " \
              "in ('Text','Long','List','MultiList','Boolean','File','DBFile','Interval','DateTime','Date') THEN " \
              "'string' WHEN DATA_TYPE in ('Integer','Number') THEN 'numeric' ELSE DATA_TYPE END AS DATATYPE from " \
              "FIELD_MASTER where TABLE_NAME in ('SAMPLE_AUDIT_LOG','SAMPLE','TEST','RESULT') order by FIELD_NAME"

with odbcLimsConnection.cursor() as lims_cursor_fetch_all:
    lims_cursor_fetch_all.execute(sql_command)
    queryFetch = lims_cursor_fetch_all.fetchall()

fieldDict = queryFetch

#JSON data transformation
dataFrame['trans_string'] = dataFrame.trans_string.str.replace("\r\n", " ", regex=True)
dataFrame['trans_string'] = dataFrame.trans_string.str.replace("\n", " ", regex=True)
dataFrame['trans_string'] = dataFrame.trans_string.str.replace("\r", " ", regex=True)
dataFrame['trans_string'] = dataFrame.trans_string.str.replace(r'" ', r"> ", regex=True)
dataFrame['trans_string'] = dataFrame.trans_string.str.replace(r' "', r" <", regex=True)
dataFrame['trans_string'] = dataFrame.trans_string.str.replace(r' <,"', r' <","' , regex=True)
dataFrame['trans_string'] = dataFrame.trans_string.str.replace(r'> <"', r'" "' , regex=True)
dataFrame['trans_string'] = dataFrame.trans_string.str.replace(r'<}', r'"}' , regex=True)
dataFrame['trans_string'] = dataFrame.trans_string.str.replace(r'""[A-Z]', r'"[A-Z]' , regex=True)
dataFrame['trans_string'] = dataFrame.trans_string.str.replace(r'\\', r' ' , regex=True)

def loads_json(x):
   try:
       return json.loads(x, strict=False) if pd.notna(x) else pd.Series() #
   except json.JSONDecodeError as e:
       print(f"Error decoding JSON: {e}")
       print(f"Problematic JSON string: {x}")
       return pd.Series()
    
dataFrame['trans_string'] = dataFrame['trans_string'].apply(loads_json)
jsonDataFrame = pd.json_normalize(dataFrame['trans_string'])

# Convert column datatypes to dictionary
fieldDictDF = pd.DataFrame.from_records(fieldDict)
fieldDictDF.set_index(0, inplace=True)
fieldDict = fieldDictDF[1].to_dict()

# Drop unnecessary or duplicated columns, replace NaN values and comma-replacement character
dataFrame = dataFrame.drop(columns=['trans_string'])
jsonDataFrame = jsonDataFrame.drop(columns=['sample_number']).replace(np.nan, "█").replace("▄", ",", regex=True)
encodedDataFrame = pd.concat([dataFrame, jsonDataFrame], axis=1)
unsupervisedFeatures = list(encodedDataFrame.drop(columns=['t_mlar']).columns.values)

# Set hyperparameters MOVE TO CONFIG
numRecords = int(round(len(encodedDataFrame.index)))
numRecordColumn = int(round(len(encodedDataFrame.columns)))
sampleSize = 256  
numTrees = 100
maxDepth = 20  
numFolds = 2

# Execution timing
endTime = time.time()
elapsedTime = endTime - startTime
now = datetime.now()
logEntry = "DateTime: " + str(now) + ", Start: " + str(startLog) + ", End: " + str(endLog) + ", Sample Audit," \
    "json convert Time = " + str(elapsedTime)
with open ("executionlog.txt", "a") as writer:
    writer.write('\n'+logEntry)
    writer.close()
    print(logEntry)

# Define datatypes and convert to H2oFrame
fieldDictUnsupervised = dict(filter(lambda item: item[0] in unsupervisedFeatures, fieldDict.items()))
unsupervisedH2oFrame = h2o.H2OFrame(python_obj=encodedDataFrame, column_types=fieldDictUnsupervised)  

# Execution timing
endTime = time.time()
elapsedTime = endTime - startTime
now = datetime.now()
logEntry = "DateTime: " + str(now) + ", Start: " + str(startLog) + ", End: " + str(endLog) + ", Sample Audit," \
    "H2oFrame Conversion Time = " + str(elapsedTime)
with open ("executionlog.txt", "a") as writer:
    writer.write('\n'+logEntry)
    writer.close()
    print(logEntry)

# Train Unsupervised model
unsupervisedModel = H2OExtendedIsolationForestEstimator(model_id="unsupervisedModel.hex", ntrees=numTrees,
                                                        ignore_const_cols=True, sample_size=sampleSize, seed=1234)
unsupervisedModel.train(x=unsupervisedFeatures, training_frame=unsupervisedH2oFrame) 
h2o.download_model(model=unsupervisedModel, path="Download")

# Execution timing
endTime = time.time()
elapsedTime = endTime - startTime
now = datetime.now()
logEntry = "DateTime: " + str(now) + ", Start: " + str(startLog) + ", End: " + str(endLog) + ", Sample Audit," \
    "Model Train and export Time = " + str(elapsedTime)
with open ("executionlog.txt", "a") as writer:
    writer.write('\n'+logEntry)
    writer.close()
    print(logEntry)

# Add new scores and anomaly columns to dataTable
predictUnsupervised = unsupervisedModel.predict(unsupervisedH2oFrame)

df = predictUnsupervised["mean_length"]
histFileName = "hist-" + config.limsdb_database + "-" + str(startLog).replace(":","") + " to " \
                + str(endLog).replace(":","") + ".png"
predictUnsupervised["mean_length"].hist(save_plot_path=histFileName)

# Set mean_length value for unsupervised
anomalies = unsupervisedH2oFrame[predictUnsupervised["mean_length"] < 8.0]
exportedH2oFrame = unsupervisedModel.predict(anomalies)["mean_length"].cbind(anomalies[unsupervisedFeatures])
exportedH2oFrame = unsupervisedModel.predict(anomalies)["anomaly_score"].cbind(exportedH2oFrame)
rowCount = exportedH2oFrame.nrows
detectionMethod = np.full((rowCount, 1), 'unsupervised')
detectionFrame = h2o.H2OFrame(detectionMethod)
exportedH2oFrame = exportedH2oFrame.cbind(detectionFrame)
exportedH2oFrame.set_name("C1", "detection_method") 

# Execution timing
endTime = time.time()
elapsedTime = endTime - startTime
now = datetime.now()
logEntry = "DateTime: " + str(now) + ", Start: " + str(startLog) + ", End: " + str(endLog) + ", Sample Audit," \
    "Predict Time = " + str(elapsedTime)
with open ("executionlog.txt", "a") as writer:
    writer.write('\n'+logEntry)
    writer.close()
    print(logEntry)

# Query current log count
cursor = odbcLimsConnection.cursor()
sql_command = "select MAX(COUNTER) from " + config.outputTable
cursor.execute(sql_command)
response = cursor.fetchone()
counter = response[0]

# Depending on the value, default either to 1 if empty or add 1 for next insert sequence
if counter is None:
    id_number = 1
else:
    id_number = counter + 1

## Prepare dataframe for sql insertion
rowCount = finalDataFrame.nrows
anomalyData = np.full((rowCount, 1), 'F')
anomalyFrame = h2o.H2OFrame(anomalyData)
counterData = (id_number) + np.arange(finalDataFrame.shape[0])
counterFrame = h2o.H2OFrame(counterData)
syncedData = np.full((rowCount, 1), 'F')
syncedFrame = h2o.H2OFrame(syncedData)
finalDataFrame = finalDataFrame.cbind(counterFrame)
finalDataFrame.set_name("C1", "counter")
finalDataFrame = finalDataFrame.cbind(anomalyFrame)
finalDataFrame.set_name("C1", "reviewed")
finalDataFrame = finalDataFrame.cbind(syncedFrame)
finalDataFrame.set_name("C1", "synced")

## Convert to panda dataframe
finalDataFrame = finalDataFrame.as_data_frame(use_pandas=True)
finalDataFrame['audit_timestamp'] = pd.to_datetime(finalDataFrame['audit_timestamp'], unit='ms')
finalDataFrame.rename(columns={'table_name': 'audit_table_name', 'table_key':'audit_table_key'}, inplace=True)
finalDataFrame.replace('', 'No Data', regex=True)
finalDataFrame.fillna('No Data', inplace=True)

# Sql insertion of anomalies
for index, row in finalDataFrame.iterrows():
    cursor.execute("INSERT INTO T_MLAR_LOG (COUNTER,MLAR_TYPE,AUDIT_TABLE_NAME,AUDIT_TABLE_KEY,AUDIT_ACTION,"
                   "AUDIT_COUNTER,AUDIT_TYPE,REVIEWED,AUDIT_TIMESTAMP,USER_NAME,DETECTION_METHOD,MEAN_LENGTH,"
                   "AUDIT_MAIN_ID,SYNCED,METRIC) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
                   , row.counter, row.mlar_type, row.audit_table_name, row.audit_table_key, row.audit_action,
                   row.audit_counter, row.audit_type, row.reviewed, row.audit_timestamp,  row.user_name,
                   row.detection_method, row.mean_length, row.sample_number, row.synced,
                   row.anomaly_score)

# Close DB connections    
odbcLimsConnection.commit()
cursor.close()

## Clear h2o memory
h2o.remove_all()
h2o.cluster().shutdown()

# Execution timing
endTime = time.time()
elapsedTime = endTime - startTime
now = datetime.now()
logEntry = "DateTime: " + str(now) + ", Start: " + str(startLog) + ", End: " + str(endLog) + ", Sample Audit," \
    "Completion Time = " + str(elapsedTime)
with open ("executionlog.txt", "a") as writer:
    writer.write('\n'+logEntry)
    writer.close()
    print(logEntry)
    