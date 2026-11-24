## Load prerequisites libraries
import pandas as pd
import numpy as np
import pathlib
import glob

import config

## Create FS Data per customer
for customerName in config.customerList:
    for auditLog in config.auditList:
        DF = pd.read_parquet(f'{config.data_location}raw_output_{auditLog}_{customerName}.parquet', engine='pyarrow')
        DF.replace('█', np.nan, inplace=True)
        if auditLog == 'sample_audit_decrypt':
            DF = DF.sort_values(by=['SOURCE_DB','SAMPLE_NUMBER','AUDIT_TIMESTAMP','AUDIT_COUNTER','TRANS_ORDER','TABLE_NAME','TABLE_KEY'],ignore_index=True).ffill() # 'AUDIT_TYPE' 'AUDIT_ACTION' 
        elif auditLog == 'exp_audit_decrypt':      
            DF = DF.sort_values(by=['SOURCE_DB','EXPERIMENT_NUMBER','AUDIT_TIMESTAMP','AUDIT_COUNTER','TRANS_ORDER','TABLE_NAME','TABLE_KEY'],ignore_index=True).ffill() # 'AUDIT_TYPE' 'AUDIT_ACTION' 
        elif auditLog == 'table_audit_decrypt':
            DF = DF.sort_values(by=['SOURCE_DB','TABLE_NAME','AUDIT_TIMESTAMP','AUDIT_COUNTER','TRANS_ORDER','PARENT_TABLE','PARENT_KEY'],ignore_index=True).ffill() # 'AUDIT_TYPE' 'AUDIT_ACTION'
        
        else:
            print('{auditLog} related information not set.')
            pass
        DF.dropna(how='all', axis=1, inplace=True)
        dirPath = f'{config.data_location}//FS_base_data//{auditLog}//'
        pathlib.Path(dirPath).mkdir(parents=True, exist_ok=True)
        fileName = dirPath + 'FS_base_data_'+ auditLog + '_' + customerName +'.parquet'
        DF.to_parquet(path=fileName)