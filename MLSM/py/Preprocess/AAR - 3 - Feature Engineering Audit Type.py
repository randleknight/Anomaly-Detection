import pandas as pd
import numpy as np
import pathlib
#import glob
#import os
import sys

if len(sys.argv) > 1:
    configAuditSelect = sys.argv[1]
    auditList = configAuditSelect.split(",")  
    dbType = sys.argv[2]
    configCredName = sys.argv[3] 
    customerList = configCredName.split(",")  
    dataLocation = sys.argv[4]
else:
    print("No variable received.")
    #sys.exit(1)
    configAuditSelect = 'Sample'
    auditList = configAuditSelect.split(",") 
    dbType ='{PostgreSQL Unicode}'
    configCredName = 'Activity'
    customerList = configCredName.split(",") 
    dataLocation = 'data//'

## Create FS Data per customer 
for customerName in customerList:
    for auditLog in auditList:
        DF = pd.read_parquet(f'{dataLocation}FS_base_data/{auditLog}/FS_base_data_{auditLog}_{customerName}.parquet', engine='pyarrow')
        DF.dropna(how='all',axis=1,inplace=True)
        dirPath = f'{dataLocation}FS_audit_type_table_name//{auditLog}//'
        pathlib.Path(dirPath).mkdir(parents=True, exist_ok=True)
        fileName = f'{dirPath}FS_audit_type_table_name_{auditLog}_{customerName}.parquet'
        DF.to_parquet(path=fileName)

        ## Create FS Data by audit type 
        FilterTypeDF = pd.read_parquet(f'{dataLocation}FS_base_data/{auditLog}/FS_base_data_{auditLog}_{customerName}.parquet', engine='pyarrow')
        FilterTypeList = FilterTypeDF['AUDIT_TYPE'].unique().tolist()
        for FilterType in FilterTypeList:
            dirPath = f'{dataLocation}//FS_audit_type_table_name//{auditLog}//{FilterType}//'
            pathlib.Path(dirPath).mkdir(parents=True, exist_ok=True)       
            fileName = f'{dirPath}FS_audit_type_table_name_{auditLog}_{FilterType}_{customerName}.parquet' 
            try:
                AuditTypeDF = DF.query(f'AUDIT_TYPE=="{FilterType}"')
                AuditTypeDF.to_parquet(path=fileName)
                print(f"AAR - 3 - Feature Engineering Audit Type {FilterType} completed for {customerName}")
            except:
                print(f"Error - AAR - 3 - Feature Engineering Audit Type {FilterType} not completed for {customerName}")
