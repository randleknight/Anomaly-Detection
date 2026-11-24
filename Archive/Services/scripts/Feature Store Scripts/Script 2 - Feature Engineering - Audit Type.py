## Load prerequisites libraries
import duckdb as db
import pandas as pd
import numpy as np
import pathlib
import glob
import os

import config

## Create FS Data per customer 
for customerName in config.customerList:
    for auditLog in config.auditList:
        DF = pd.read_parquet(f'{config.data_location}FS_base_data/{auditLog}/FS_base_data_{auditLog}_{customerName}.parquet', engine='pyarrow')
        DF.dropna(how='all',axis=1,inplace=True)
        dirPath = f'{config.data_location}FS_audit_type_table_name//{auditLog}//'
        pathlib.Path(dirPath).mkdir(parents=True, exist_ok=True)
        fileName = f'{dirPath}FS_audit_type_table_name_{auditLog}_{customerName}.parquet'
        DF.to_parquet(path=fileName, engine='pyarrow')

        ## Create FS Data by audit type 
        FilterTypeDF = pd.read_parquet(f'{config.data_location}FS_base_data/{auditLog}/FS_base_data_{auditLog}_{customerName}.parquet', engine='pyarrow')
        FilterTypeList = FilterTypeDF['AUDIT_TYPE'].tolist()
        for FilterType in FilterTypeList:
            dirPath = f'{config.data_location}//FS_audit_type_table_name//{auditLog}//{FilterType}//'
            pathlib.Path(dirPath).mkdir(parents=True, exist_ok=True)       
            fileName = f'{dirPath}FS_audit_type_table_name_{auditLog}_{FilterType}_{customerName}.parquet' 
            try:
                AuditTypeDF = DF.query(f'AUDIT_TYPE=="{FilterType}"')
                #AuditTypeDF.dropna(how='all',axis=1,inplace=True)
                AuditTypeDF.to_parquet(path=fileName, engine='pyarrow')
            except:
                print(f'{FilterType} not present for {customerName}')
