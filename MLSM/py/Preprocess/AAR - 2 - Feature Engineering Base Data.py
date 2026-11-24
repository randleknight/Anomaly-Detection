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
    sys.exit(1) 

## Create FS Data per customer
for customerName in customerList:
    for auditLog in auditList:
        if auditLog == 'Sample':
            columnNames = ['SOURCE_DB','SAMPLE_NUMBER','AUDIT_TIMESTAMP','AUDIT_COUNTER','TRANS_ORDER',
                           'AUDIT_TYPE','AUDIT_ACTION','TABLE_NAME','TABLE_KEY','REASON','RECORD_SIGNED',
                           'USER_NAME','USER_ROLE','TEXT_ID_JSON','STATUS_JSON', #'T0083','T0083_SEVERITY',
                           'OLD_STATUS_JSON','IN_SPEC_JSON','IN_CAL_JSON','RE_SAMPLE_JSON','ORIGINAL_SAMPLE_JSON',
                           'ALIQUOT_JSON','PARENT_SAMPLE_JSON','PARENT_ALIQUOT_JSON','LOGIN_DATE_JSON','LOGIN_BY_JSON',
                           'RECD_DATE_JSON','RECEIVED_BY_JSON','DATE_STARTED_JSON','STARTED_JSON','DATE_COMPLETED_JSON',
                           'DATE_REVIEWED_JSON','PRODUCT_VERSION_JSON','TEMPLATE_JSON','STANDARD_JSON','LOT_JSON',
                           'CHANGED_ON_JSON','PARTIAL_SPEC_JSON','PRIMARY_IN_SPEC_JSON','RELEASED_JSON', 'BATCH_JSON',
                           'IN_CONTROL_JSON','INVESTIGATED_JSON', 'RELEASED_BY_JSON','RELEASED_ON_JSON',
                           'MODIFIED_RESULTS_JSON','COMPOSITE_JSON','PARENT_COMPOSITE_JSON','HAS_FLAGS_JSON',
                           'SIGNED_JSON','TEST_NUMBER_JSON','CHARGE_ENTRY_JSON','ANALYSIS_JSON','VERSION_JSON',
                           'REPLICATE_COUNT_JSON','DATE_RECEIVED_JSON','PREP_JSON','REVIEWER_JSON',
                           'REPLICATE_TEST_JSON','ORDER_NUMBER_JSON','GROUP_NAME_JSON','RESOLVE_REQD_JSON',
                           'STAGE_JSON','RE_TESTED_JSON','ALIQUOTED_TO_JSON','ON_WORKSHEET_JSON',
                           'ANALYSIS_COUNT_JSON','BATCH_ORIGINAL_TEST_JSON','BATCH_PARENT_TEST_JSON',
                           'CHILD_OUT_SPEC_JSON','CROSS_SAMPLE_JSON','ORIGINAL_TEST_JSON','PARENT_TEST_JSON',
                           'RESULT_NUMBER_JSON','NAME_JSON','RESULT_TYPE_JSON','MINIMUM_JSON','MAXIMUM_JSON',
                           'ALLOW_OUT_JSON','FORMATTED_ENTRY_JSON','NUMERIC_ENTRY_JSON','PLACES_JSON',
                           'ENTERED_ON_JSON','ENTERED_BY_JSON','USES_CODES_JSON','AUTO_CALC_JSON','LIST_KEY_JSON',
                           'ALLOW_CANCEL_JSON','REPORTABLE_JSON','OPTIONAL_JSON','CODE_ENTERED_JSON','ROUND_JSON',
                           'STD_REAG_SAMPLE_JSON','DISPLAYED_JSON','USES_INSTRUMENT_JSON','ENTRY_QUALIFIER_JSON',
                           'HAS_ATTRIBUTES_JSON','ENTRY_TYPE_JSON','CONTROL_1_JSON','REPORTED_NAME_JSON',
                            'PREP_DATE_JSON','PREP_BY_JSON','PROJECT_JSON','SAMPLE_TYPE_JSON','PROTOCOL_LEVEL_JSON',
                           'VARIATION_JSON','TEST_LIST_JSON','INSTRUMENT_JSON','FACTOR_OPERATOR_JSON','ALIAS_NAME_JSON','CONTROL_2_JSON'
                           ]    
            DF = pd.read_parquet(f'{dataLocation}raw_output_{auditLog}_{customerName}.parquet', engine='pyarrow') #, columns=columnNames
            DF = DF.loc[:, DF.columns.isin(columnNames)]
            DF.replace('█', np.nan, inplace=True)
            DF = DF.sort_values(by=['SOURCE_DB','SAMPLE_NUMBER','AUDIT_TIMESTAMP','AUDIT_COUNTER','TRANS_ORDER','TABLE_NAME','TABLE_KEY'],ignore_index=True).ffill() # 'AUDIT_TYPE' 'AUDIT_ACTION'
        
        elif auditLog == 'Experiment': 
            #RK fields not for x7 - ASSIGNED_TO_JSON,PRINTED_ON_JSON,PRINTED_BY_JSON,FILE_INSTR_1_JSON,FILE_INSTR_2_JSON,FILE_INSTR_3_JSON,FILE_INSTR_4_JSON,FILE_INSTR_5_JSON,DIRECT_INSTR_1_JSON,DIRECT_INSTR_2_JSON,DIRECT_INSTR_3_JSON,DIRECT_INSTR_4_JSON,DIRECT_INSTR_5_JSON,CHECKED_OUT_BY_JSON,CHECKED_OUT_ON_JSON,ENTER_BY_JSON,ENTERED_ON_JSON,,WITNESSED_BY_JSON,WITNESSED_ON_JSONINSTRUMENT_JSON,IMAGE_HASH_JSON,IMAGE_CONTENTS_JSON,
            columnNames = ['SOURCE_DB','AUDIT_COUNTER','TABLE_NAME','TABLE_KEY','AUDIT_TYPE','AUDIT_ACTION','REASON','TRANS_ORDER','USER_NAME',
                           'AUDIT_TIMESTAMP','RECORD_SIGNED','EXPERIMENT_NUMBER','WORKBOOK_NUMBER','WORKSHEET','COL_NUM','ROW_NUM','DOCUMENT_NUMBER',
                           'WORKBOOK_NUMBER_JSON','TEMPLATE_JSON','TEMPLATE_VERSION_JSON','STATUS_JSON','CREATED_BY_JSON','CREATED_ON_JSON','CHANGED_ON_JSON',
                           'CHANGED_BY_JSON','COMPLETED_ON_JSON','COMPLETED_BY_JSON','REVIEWED_ON_JSON','REVIEWED_BY_JSON','CANCELED_ON_JSON','CANCELED_BY_JSON',
                           'REMOVED_JSON','APPROVED_JSON','ARCHIVED_JSON','GROUP_NAME_JSON','DATA_MODIFIED_JSON','FILE_LINK_JSON','WORKFLOW_ID_JSON','ANALYSES_JSON',
                           'CHECKED_OUT_JSON','COMPLETED_JSON','REVIEWED_JSON','REVIEW_TYPE_JSON','REVIEW_COMMENT_JSON','SIGNED_JSON','SHEET_JSON','ROW_NUM_JSON','COL_NUM_JSON',
                           'NAME_JSON','VALUE_JSON','REVIEW_STATUS_JSON','MODIFIED_BY_JSON','MODIFIED_ON_JSON','WORKBOOK_OBJ_NO_JSON','FIELD_NAME_JSON','OBJECT_TYPE_JSON',
                           'OBJECT_ID_JSON','OBJECT_NAME_JSON','IN_SPEC_JSON','IN_CONTROL_JSON','IN_CAL_JSON','WITNESSED_JSON','CALCULATED_JSON','PASS_FLAG_JSON','PREVIOUS_VALUE_JSON',
                           'IMAGE_NUMBER_JSON','IMAGE_FORMAT_JSON','OBJECT_INDEX_JSON','SAMPLE_REP_JSON','TEST_REP_JSON','RESULT_REP_JSON','UPDATE_REFERENCE_JSON','MODIFIED_DATA_JSON',
                           'SHEET_1_JSON','SHEET_2_JSON','SHEET_3_JSON','ENTRY_CODE_JSON','SHEET_NAME_JSON','COMPLIANCE_FLAG_JSON','DATA_REVIEW_STATUS_JSON']
            DF = pd.read_parquet(f'{dataLocation}raw_output_{auditLog}_{customerName}.parquet', engine='pyarrow') #, columns=columnNames 
            DF = DF.loc[:, DF.columns.isin(columnNames)]
            DF.replace('█', np.nan, inplace=True)
            DF = DF.sort_values(by=['SOURCE_DB','EXPERIMENT_NUMBER','AUDIT_TIMESTAMP','AUDIT_COUNTER','TRANS_ORDER','TABLE_NAME','TABLE_KEY'],ignore_index=True).ffill() # 'AUDIT_TYPE' 'AUDIT_ACTION'

        elif auditLog == 'Table':
            ## Columns names restriction by lims controlled fields not applied - Numerous tables fields
            #columnNames = ['SOURCE_DB','AUDIT_COUNTER','TABLE_NAME','TABLE_KEY','AUDIT_TYPE','AUDIT_ACTION','REASON','TRANS_ORDER','USER_NAME',
            #               'AUDIT_TIMESTAMP','RECORD_SIGNED','PARENT_TABLE','PARENT_KEY','USER_ROLE',]     
            DF = pd.read_parquet(f'{dataLocation}raw_output_{auditLog}_{customerName}.parquet', engine='pyarrow') #, columns=columnNames
            DF.replace('█', np.nan, inplace=True)
            DF = DF.sort_values(by=['SOURCE_DB','TABLE_NAME','AUDIT_TIMESTAMP','AUDIT_COUNTER','TRANS_ORDER','PARENT_TABLE','PARENT_KEY'],ignore_index=True).ffill() # 'AUDIT_TYPE' 'AUDIT_ACTION'
        
        elif auditLog == 'Investigation':
            columnNames = ['SOURCE_DB','AUDIT_COUNTER','TABLE_NAME','TABLE_KEY','AUDIT_TYPE','AUDIT_ACTION','REASON','TRANS_ORDER','USER_NAME',
                           'AUDIT_TIMESTAMP','RECORD_SIGNED','INVESTIGATION','USER_ROLE','NAME_JSON','CHANGED_ON_JSON',
                           'CREATED_BY_JSON','OWNER_JSON','CLOSED_BY_JSON','TEMPLATE_JSON','GROUP NAME_JSON','SIGNED_JSON','INVESTIGATION_JSON',
                           'OBJECT CLASS_JSON','OBJECT_ID_JSON','ORDER NUMBER_JSON']     
            DF = pd.read_parquet(f'{dataLocation}raw_output_{auditLog}_{customerName}.parquet', engine='pyarrow') #, columns=columnNames
            DF = DF.loc[:, DF.columns.isin(columnNames)]
            DF.replace('█', np.nan, inplace=True)
            DF = DF.sort_values(by=['SOURCE_DB','INVESTIGATION','AUDIT_TIMESTAMP','AUDIT_COUNTER','TRANS_ORDER','TABLE_NAME','TABLE_KEY'],ignore_index=True).ffill() # 'AUDIT_TYPE' 'AUDIT_ACTION'
        
        elif auditLog == 'Lot':
            columnNames = ['SOURCE_DB','AUDIT_COUNTER','TABLE_NAME','AUDIT_TYPE','AUDIT_ACTION','REASON','TRANS_ORDER','USER_NAME',
                           'AUDIT_TIMESTAMP','RECORD_SIGNED','PARENT_TABLE','PARENT_KEY','USER_ROLE','LOT_NUMBER_JSON','LOT_NAME_JSON',
                           'CHANGED_BY_JSON','CHANGED_ON_JSON','GROUP_NAME_JSON','DAE_CREATED_JSON','CLOSED_JSON','REPORT_DATE_JSON','REPORT_NUMBER_JSON',
                           'REPRINT_FLAG_JSON','REPRINT_DATE_JSON','ORIGINAL_LOT_JSON','PARENT_LOT_JSON','FULL_TESTING_REQD_JSON','IN_SPEC_SON',
                           'PRIMARY_IN_SPEC_JSON','TEMPLATE_JSON','SIGNED_JSON','SAMPLING_POINT_JSON','PRODUCT_JSN','PRODUCT_VERSION_JSON',
                           'PRODUCT_GRADE_JSON','PRIMARY_GRADE_JSON','ORDER_NUMBER_JSON','DIPOSITION_DATE_JSON','DISPOSITION_BY_JSON',
                           'RESULTS_MODIFIED_JSON','RESULTS_MISSING_JSON','SAMPLES_REVIEWED_JSON','TESTING_DONE_JSON','RESULTS_REJECTED_JSON']     
            DF = pd.read_parquet(f'{dataLocation}raw_output_{auditLog}_{customerName}.parquet', engine='pyarrow') #, columns=columnNames 
            DF = DF.loc[:, DF.columns.isin(columnNames)]
            DF.replace('█', np.nan, inplace=True)
            DF = DF.sort_values(by=['SOURCE_DB','TABLE_NAME','AUDIT_TIMESTAMP','AUDIT_COUNTER','TRANS_ORDER','PARENT_TABLE','PARENT_KEY'],ignore_index=True).ffill() # 'AUDIT_TYPE' 'AUDIT_ACTION'
        
        elif auditLog == 'Project':
            columnNames = ['SOURCE_DB','AUDIT_COUNTER','TABLE_NAME','AUDIT_TYPE','AUDIT_ACTION','REASON','TRANS_ORDER','USER_NAME',
                           'AUDIT_TIMESTAMP','RECORD_SIGNED','PROJECT_NAME','TABLE_KEY','USER_ROLE','SAMPLE_NUMBER_JSON',
                           'TEXT_ID_JSON','STATUS_JSON','OLD_STATUS_JSON','IN_SPEC_JSON','IN_CAL_JSON','RE_SAMPLE_JSON','ORIGINAL_SAMPLE_JSON',
                           'ALIQUOT_JSON','PARENT_SAMPLE_JSON','PARENT_ALIQUOT_JSON','LOGIN_DATE_JSON','LOGIN_BY_JSON','RECD_DATE_JSON',
                           'RECEIVED_BY_JSON','DATE_STARTED_JSON','STARTED_JSON','DATE_COMPLETED_JSON','DATE_REVIEWED_JSON','PREP_DATE_JSON',
                           'PREP_BY_JSON','PRODUCT_VERSION_JSON','PROJECT_JSON','BATCH_NAME_JSON','BATCH_TEMPLATE_JSON','STEP_JSON','SAMPLE_TYPE_JSON',
                           'TEMPLATE_JSON','STANDARD_JSON','LOT_JSON','CHANGED_ON_JSON','PARTIAL_SPEC_JSON','PRIMARY_IN_SPEC_JSON','BATCH_JSON',
                           'PROTOCOL_LEVEL_JSON','RELEASED_JSON','RELEASED_BY_JSON','RELEASED_ON_JSON','IN_CONTROL_JSON','INVESTIGATION_JSON',
                           'INVESTIGATED_JSON','MODIFIED_RESULTS_JSON','COMPOSITE_JSON','PARENT_COMPOSITE_JSON','HAS_FLAGS_JSON','IMPORTANT_MESSAGE_JSON'
                           'SIGNED_JSON','NAME_JSON','TEMPLATE_VERSION_JSON','DATE_CREATED_JSON','DATE_RECEIVED_JSON','OWNER_JSON','CREATED_BY_JSON',
                           'NUM_SAMPLES_JSON','NUM_U_JSON','NUM_1_JSON','NUM_P_JSON','NUM_C_JSON','NUM_A_JSON','NUM_R_JSON','NUM_X_JSON','GROUP_NAME_JSON',
                           'CLOSED_JSON','DATE_UPDATED_JSON','RESULT_NUMBER_JSON','REPLICATE_COUNT_JSON','ORDER_NUMBER_JSON','RESULT_TYPE_JSON',
                           'MINIMUM_JSON','MAXIMUM_JSON','ALLOW_OUT_JSON','FORMATTED_ENTRY_JSON','PLACES_JSON','ENTERED_ON_JSON','ENTERED_BY_JSON',
                           'USES_CODES_JSON','AUTO_CALC_JSON','LIST_KEY_JSON','ALLOW_CANCEL_JSON','REPORTABLE_JSON','OPTIONAL_JSON','CODE_ENTERED_JSON',
                           'REVIEWER_JSON','STD_REAG_SAMPLE_JSON','DISPLAYED_JSON']     
            DF = pd.read_parquet(f'{dataLocation}raw_output_{auditLog}_{customerName}.parquet', engine='pyarrow') #, columns=columnNames
            DF = DF.loc[:, DF.columns.isin(columnNames)]
            DF.replace('█', np.nan, inplace=True)
            DF = DF.sort_values(by=['SOURCE_DB','PROJECT_NAME','AUDIT_TIMESTAMP','AUDIT_COUNTER','TRANS_ORDER','TABLE_NAME','TABLE_KEY'],ignore_index=True).ffill() # 'AUDIT_TYPE' 'AUDIT_ACTION'
        
        elif auditLog == 'Batch':
            columnNames = ['SOURCE_DB','AUDIT_COUNTER','TABLE_NAME','AUDIT_TYPE','AUDIT_ACTION','REASON','TRANS_ORDER','USER_NAME',
                           'AUDIT_TIMESTAMP','RECORD_SIGNED','BATCH_NAME','TABLE_KEY','USER_ROLE','NAME_JSON','OBJECT_CLASS_JSON',
                           'CHANGED_ON_JSON','CHANGED_BY_JSON','CREATED_BY_JSON','DATE_CREATED_JSON','TEMPLATE_VERSION_JSON','CLOSED_JSON',
                           'INVESTIGATED_JSON','SIGNED_JSON','RESULT_NUMBER_JSON','BATCH_JSON','REPLICATE_COUNT_JSON','ORDER_NUMBER_JSON',
                           'RESULT_TYPE_JSON','MINIMUM_JSON','MAXIMUM_JSON','ALLOW_OUT_JSON','FORMATTED_ENTRY_JSON','NUMERIC_ENTRY_JSON',
                           'PLACES_JSON','STATUS_JSON','OLD_STATUS_JSON','ENTERED_ON_JSON','ENTERED_BY_JSON','USES_CODES_JSON','AUTO_CALC_JSON',
                           'LIST_KEY_JSON','ALLOW_CANCEL_JSON','IN_SPEC_JSON','REPORTABLE_JSON','OPTIONAL_JSON','CODE_ENTERED_JSON',
                           'DATE_REVIEWED_JSON','REVIEWER_JSON','STD_REAG_SAMPLE_JSON','DISPLAYED_JSON']     
            DF = pd.read_parquet(f'{dataLocation}raw_output_{auditLog}_{customerName}.parquet', engine='pyarrow') #, columns=columnNames
            DF = DF.loc[:, DF.columns.isin(columnNames)]
            DF.replace('█', np.nan, inplace=True)
            DF = DF.sort_values(by=['SOURCE_DB','BATCH_NAME','AUDIT_TIMESTAMP','AUDIT_COUNTER','TRANS_ORDER','TABLE_NAME','TABLE_KEY'],ignore_index=True).ffill() # 'AUDIT_TYPE' 'AUDIT_ACTION'
        
        elif auditLog == 'Stor_loc': 
            columnNames = ['SOURCE_DB','AUDIT_COUNTER','AUDIT_TYPE','AUDIT_ACTION','REASON','TRANS_ORDER','USER_NAME',
                           'AUDIT_TIMESTAMP','RECORD_SIGNED','LOCATION_NUMBER','NAME_JSON','GROUP_NAME_JSON',
                           'CHANGED_ON_JSON','CHANGED_BY_JSON','REMOVED_JSON']     
            DF = pd.read_parquet(f'{dataLocation}raw_output_{auditLog}_{customerName}.parquet', engine='pyarrow') #, columns=columnNames
            DF = DF.loc[:, DF.columns.isin(columnNames)]
            DF.replace('█', np.nan, inplace=True)
            DF = DF.sort_values(by=['SOURCE_DB','LOCATION_NUMBER','AUDIT_TIMESTAMP','AUDIT_COUNTER','TRANS_ORDER'],ignore_index=True).ffill() # 'AUDIT_TYPE' 'AUDIT_ACTION'
        
        elif auditLog == 'Reports':
            columnNames = ['SOURCE_DB','AUDIT_COUNTER','AUDIT_TYPE','AUDIT_ACTION','REASON','TRANS_ORDER','USER_NAME',
                           'AUDIT_TIMESTAMP','RECORD_SIGNED','REPORT_NUMBER','REPORT_NUMBER_JSON','DESCRIPTION_JSON',
                           'OBJECT_CLASS_JSON','DATE_CREATED_JSON','CREATED_BY_JSON','PRINT_COUNT_JSON','DATA_CHANGED_JSON',
                           'RESUTS_CHANGED_JSON','REPORT_FILE_NAME_JSON','SUPERSEDED_JSON','SUPERSEDED_BY_JSON','REPORT_STATUS_SON',
                           'FAILURE_REASON_JSON']     
            DF = pd.read_parquet(f'{dataLocation}raw_output_{auditLog}_{customerName}.parquet', engine='pyarrow') #, columns=columnNames 
            DF = DF.loc[:, DF.columns.isin(columnNames)]
            DF.replace('█', np.nan, inplace=True)
            DF = DF.sort_values(by=['SOURCE_DB','REPORT_NUMBER','AUDIT_TIMESTAMP','AUDIT_COUNTER','TRANS_ORDER'],ignore_index=True).ffill() # 'AUDIT_TYPE' 'AUDIT_ACTION'
        

        DF.dropna(how='all', axis=1, inplace=True)
        dirPath = f'{dataLocation}//FS_base_data//{auditLog}//'
        pathlib.Path(dirPath).mkdir(parents=True, exist_ok=True)
        fileName = dirPath + 'FS_base_data_'+ auditLog + '_' + customerName +'.parquet'
        DF.to_parquet(path=fileName, engine='pyarrow')
        print(f"AAR - 2 - Feature Engineering Base Data completed for {customerName}")

