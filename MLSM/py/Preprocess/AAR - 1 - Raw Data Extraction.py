## Load prerequisites libraries
import numpy as np
import pandas as pd
import json
import pyarrow as pa
import pyarrow.parquet as pq
import sys 
import os
from cryptography.fernet import Fernet

if len(sys.argv) > 1:
    configAuditSelect = sys.argv[1]
    auditList = configAuditSelect.split(",")  
    dbType = sys.argv[2]
    configCredName = sys.argv[3] 
    customerList = configCredName.split(",")  
    dataLocation = sys.argv[4]
    trainingSpan = int(sys.argv[5])
    inferenceSpan = int(sys.argv[6])
    processType = sys.argv[7]
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
            odbcLimsConnection = psycopg2.connect(f'dbname={credentials[customerName].get("db_database")} user= {credentials[customerName].get("db_username")} password = {credentials[customerName].get("db_password")} host= {credentials[customerName].get("db_hostname")}')
            #odbcLimsConnection.setAutoCommit(true)
            return odbcLimsConnection
        
        elif dbType == "oracle":
            import oracledb
            #oracledb.init_oracle_client(config.lib_dir)
            odbcLimsConnection = oracledb.connect(user=credentials[customerName].get("db_username"), 
                                                password=credentials[customerName].get("db_password"),
                                                dsn=credentials[customerName].get("db_dsn"))
            odbcLimsConnection.arraysize = 5000

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

    for auditLog in auditList:
        cursor = get_lims_connection().cursor()
        if processType == 'Training/Retraining':
                total_span = trainingSpan + inferenceSpan
                time_clause = f"""where audit_timestamp between (select MAX(audit_timestamp) - interval  '{total_span}' day from sample_audit_uncrpt) 
                and (select MAX(audit_timestamp) - interval '{inferenceSpan}' day from sample_audit_uncrpt)"""
        else: 
                time_clause = f"""where audit_timestamp between (select MAX(audit_timestamp) - interval '{inferenceSpan}' day from sample_audit_uncrpt)
                and (select MAX(audit_timestamp) from sample_audit_uncrpt)"""
        # Retrieve DataFrame
        if auditLog == 'Sample':
            column_names = ['AUDIT_COUNTER', 'TABLE_NAME', 'TABLE_KEY', 'AUDIT_TYPE', 
                    'AUDIT_ACTION', 'REASON', 'TRANS_ORDER', 'USER_NAME',
                    'AUDIT_TIMESTAMP', 'RECORD_SIGNED',  'TRANS_STRING', 
                    'SAMPLE_NUMBER','USER_ROLE']        
            sql_command = f"""select counter,table_name,table_key,audit_type,
                    action,reason,trans_order,user_name,
                    audit_timestamp,record_signed,trans_string,
                    sample_number,user_role from sample_audit_uncrpt 
                    {time_clause}  
                    group by counter,table_name,table_key,audit_type,  
                    action,reason,trans_order,user_name,
                    audit_timestamp,record_signed,trans_string,
                    sample_number,user_role """
            print(sql_command)          
            fieldDataTypeTables = "'SAMPLE_AUDIT_LOG','SAMPLE','TEST','RESULT'"
        elif auditLog == 'Experiment':   
            column_names = ["AUDIT_COUNTER","TABLE_NAME","TABLE_KEY","AUDIT_TYPE",
                    "AUDIT_ACTION","REASON","TRANS_ORDER","USER_NAME",
                    "AUDIT_TIMESTAMP","RECORD_SIGNED","TRANS_STRING",
                    "EXPERIMENT_NUMBER","WORKBOOK_NUMBER","WORKSHEET",
                    "COL_NUM","ROW_NUM","DOCUMENT_NUMBER"]
            sql_command = f"""select counter,table_name,table_key,audit_type,
                    action,reason,trans_order,user_name,
                    audit_timestamp,record_signed,trans_string,
                    experiment_number,workbook_number,worksheet,
                    col_num,row_num,document_number from experiment_audit_uncrpt 
                    group by counter,table_name,table_key,audit_type,
                    action,reason,trans_order,user_name,
                    audit_timestamp,record_signed,trans_string,
                    experiment_number,workbook_number,worksheet,
                    col_num,row_num,document_number {time_clause}"""              
            fieldDataTypeTables = "'EXP_AUDIT_LOG','EXPERIMENT','WORKBOOK','WORKBOOK_COLLABOR','WORKBOOK_DATA'," \
                    "'WORKBOOK_IMAGE','WORKBOOK_OBJECTS','WORKBOOK_SHEET'"
        elif auditLog == 'Table':   
            column_names = ["AUDIT_COUNTER","TABLE_NAME","AUDIT_TYPE",
                    "AUDIT_ACTION","REASON","TRANS_ORDER","USER_NAME",
                    "AUDIT_TIMESTAMP","RECORD_SIGNED","TRANS_STRING",
                    "PARENT_TABLE","PARENT_KEY","USER_ROLE"]
            sql_command = f"""select counter,table_name,audit_type,"
                    action,reason,trans_order,user_name," 
                    audit_timestamp,record_signed,trans_string," 
                    parent_table,parent_key,user_role " 
                    from table_audit_uncrpt where table_name not like 'T_%'"
                    group by counter,table_name,audit_type,action,reason,trans_order,user_name,
                    audit_timestamp,record_signed,trans_string,parent_table,parent_key,user_role 
                    {time_clause} """        
            fieldDataTypeTables = "'TABLE_AUDIT_LOG','ACCESS_ENTRY','ACCESS_ROUTINES','AD_HOC_COMPONENT','ANALYSIS','ANALYSIS_LIMITS'," \
                "'ANALYSIS_METHOD','ANALYSIS_VARIATION','APROVAL','APPROVAL_DETAILS','APPROVAL_GROUPS','APPROVAL_ROLE','APPROVAL_ROUTING'," \
                "'APPROVAL_STEPS','ARRAY_TAG_PROPS','AUDIT_EIG_EVENTS','AUTOMATION_SCRIPT','BASIC_SCHEDULER','BATCH_CALCULATION','BATCH_CALC_VAR'," \
                "'BATCH_COMPONENT','BATCH_HDR_TEMPLATE','BATCH_HDR_TMP_FLDS','BATCH_LINK','BATCH_PROTOCOL','BATCH_STANDARD','BROWSE_TAG_PROPS'," \
                "'CALCULATION','CALC_VARIABLES','CHRGE_CODE','CHARGES','CHART_GROUPS','CHART_TEMPLATE','CM_INSTANCE','CM_MACHINE','CM_SERVICE'," \
                "'CODE_SNIPPETS','COMMON_NAME','OMPONENT','COMP_VARIATION','CONDITION','CONFIG_PACKAGE','CONFIG_PACK_FILE','CONFIG_PACK_FUNCTION'," \
                "'CONFIG_PACK_OBJECT','CONFIG_PACK_POST_SUB','CONFIG_PACK_SQL','CONFIG_PACK_TABLE','CONFIG_PACK_WINDOW','CONFIG_SYS','CONTAINER_TYPE'," \
                "'DATABASE_EVENTS','DATA_EXPLORER','DATA_EXPLORER_FLDS','DB_DIRECTORY','DEVICE_HOST_PROPS','DEVICE_TASK_ENTRY','DEVICE_TASK_LIST'," \
                "'EVENTS','XP_ID_CNFG','EXP_TEMPLATE','EXP_TEMP_WRKBK','EXP_TMP_FLDS','EXTENDED_LB_FUNCTION','FACILITY','FIELD_MASTER','FOLDER_GROUP'," \
                "'FOLDER_TEMPLATE','FORMAT_CALCULATION','GENEALOGY_ACTIONS','GENEALOGY_CHILD_GRPS','GENEALOGY_TEMPLATE','GROUP_ENTRY','HTML_FILES'," \
                "'ICON_RULES','ID_CONFIGURATION','INCRMNTS_TO_SEQNCS','INFO_RULES','INSTALL_LOG','INST_EVENT_TMP','INST_EVNT_TMP_FLDS','INSTRUMENT_LOG'," \
                "'INSTRUMENT_METHOD','INSTRUMENTS','INSTRUMENT_TMP','INST_TMP_FIELDS','INVENTORY_FILTER','INVENTORY_ID_CONFIG','INVENTORY_TMP'," \
                "'INVENTORY_TMP_FIELDS','INVENTORY_TMP_GROUPS','INV_ENTRY_ID_CONFIG','INV_ENTRY_TMP','INV_ENTRY_TMP_FIELDS','INVEST_ID_CONFIG'," \
                "'INVESTIGATION_TMP','INVEST_TMP_FIELDS','INV_FILTER_FIELDS','INV_FLTER_DSPLY_FLDS','INV_TRANS_TMP','IV_TRANS_TMP_FIELDS'," \
                "'ITEM_TEMPLATE','ITEM_TEMPLATE_FLDS','LABEL_PRINTER','LDAP_SERVER_MAPPING','LDAP_SERVERS','LIMS_CONSTANS','LIMS_HTML_NOTES'," \
                "'LIMS_LOG_CATEGORY','LIMS_USERS','LIST','LIST_ENTRY','LOCATION','LOT_ID_CONFIG','LOT_TEMP_FIELDS','LOT_TEMP_GROUPS','LOT_TEMPLATE'," \
                "'MENU_ROUTINE','METHOD','OBJ_GRP_TEMPLATE','PARSING_FORMULA','PARSING_SCRIPTS','PASSWORD_HISTOY','PDF_CONVERT','PLANT'," \
                "'PROCESS_SCHEDULE','PROCESS_UNIT','PROC_SCHED_PARENT','PROD_GRADE_STAGE','PROD_TEMP_FIELDS','PROD_EMP_GROUPS','PRODUCT'," \
                "'PRODUCT_GRADE','PRODUCT_SPEC','PRODUCT_TEMPLATE','PROJECT_FILTER','PROJECT_ID_CONFIG','PROJECT_TEMP_IELDS','PROJECT_TEMPLATE'," \
                "'PROJ_TEMP_GROUPS','PROMPT_TAG_PROPS','PROT_CELL_SMP_TMP','PROT_CELL_TST_TMP','PROT_DISPLAY_FLDS''PROTOCOL_TMP','QC_SAMPLES'," \
                "'QUERY_TAG','QUERY_TAG_PROPS','REPORTS','REPORT_WIZARD','REQ_ENT_TMP','REQ_ENT_TMP_FLDS','REQUET_ID_CONFIG','REQUEST_TMP'," \
                "'REQUEST_TMP_FIELDS','REQUEST_TMP_GROUPS','RESOURCE_BUNDLE','SAMPLE_FILTER','SAMPLE_FILTER_FIELDS','SAMPLE_TEMPLATE','SAMPLING_POINT'," \
                "'SAMP_TEMP_GROUPS','SCHEDULER','SEARCH_TEMP_FLDS','SEARCH_TEMP_GRPS','SEARCH_TEMPLATE','SERVICE_MANAGER','SERVICES','SQL_SNIPPET'," \
                "'SQL_SNIPPET_ENTRY','STANDARD_REAGENT','STATUS_RULES','STD_ID_CONFIG','STD_REA_TEMP','STOCK','STORAGE_LOC_TEMP','STORAGE_MOVE_TEMP'," \
                "'STORED_QUERY','STOR_LOC_ID_CONFIG','STOR_LOC_TMP_FLDS','STYLE_LIST','SUBROUTINE','SUPPLIER','TABLE_LIST_DETAIL','TABLE_LIST_HEADER'," \
                "'TABLE_MASTER','TABLE_TEMP_FIELDS','TABLE_TEMP_GROUPS','TABLE_TEMPLATE','T_ALIQUOT_GROUP','T_BUILD_NOTE','T_BUILD_NOTE_DETAIL'," \
                "'T_BULK_RESULT_ENTRY','T_BUTTONGROUP','T_BUTTONGROUP_CLICK','T_CAS_NUMBER','T_CLIENT','T_CONFIG_PARSER','T_CONFIG_PARSER_DTL'," \
                "'T_CONFIG_PARSER_SQL','T_CORRECTIVE_ACTION','T_DSE_TEMLATE','T_EM_PLAN','T_EM_PLAN_UNITS','TEMPLATE_FIELDS','TEST_EVENTS','TEST_LIST'," \
                "'TEST_LIST_ENTRY','TEST_TEMPLATE','T_IM_ENTY_ACTIONS','TIMEPOINT','T_INST_EXP_FILE','T_INST_EXP_FILE_DTL','T_INVENTORY_SUBCAT'," \
                "'T_KANBAN','T_KANBAN_ACTION','T_KANBAN_COLUMN','T_KANBAN_TYPE','T_KANBAN_TYPE_AC','T_KPI_QUERY2','T_LABEL_CHAR_SUBS','T_LABEL_FORMAT'," \
                "'T_LABEL_FORMAT_FONT','T_LABE_FORMAT_LINE','T_LABEL_IMAGE','T_LABEL_OBJECT','T_LABEL_PLAN','T_LABEL_PLAN_TRIGGER','T_LABEL_TRIGGER'," \
                "'T_LEM_CALLBACK','T_LOT_DISP_CHECKS','T_LOT_DISP_CHK_ITEM','T_LOT_DISP_CHK_TYPE','T_LOT_DISP_VALUE','T_ME_EXTRACT','T_ME_EXTRACT_RULE'," \
                "'T_ME_SERICE_POOL','T_ME_TYPE','T_ME_TYPE_EXT_RULE','T_NLS_OBJ_TBL_CONFIG','T_NLS_SESSION'," \
                "'T_NLS_SESSON_DTL','T_NLS_UTIL_DETAIL','T_NLS_UTIL_GROUP','T_NLS_VWF_VALID_NLS','T_OBJECT_ACCESS','T_OBJECT_ACCESS_ROW'," \
                "'T_OBJECT_ACCES_TBL','T_ORGANISM','T_PERSONNEL','T_PH_AQL_PLAN_SIZES','T_PH_AQL_SAMPLE_PLAN','T_PH_COA','T_PH_COA_ADDRESS'," \
                "'T_PH_COA_SETUP','T_PH_CONDUCTIVITY_2','T_PH_GRADE','T_PH_INST_EVENT','T_PH_ITEM_CODE','T_PH_ITEM_CODE_CERT','T_PH_ITEM_CODE_SPEC'," \
                "'T_PH_TEM_CODE_SUPP','T_PH_LOT_SAMPLES','T_PH_MV_PROTOCOL','T_PH_MV_PROT_PARAM','T_PH_PACK_TYPE','T_PH_PROD_COMP','T_PH_PROD_COMP_ITEM'," \
                "'T_PH_PROD_LABELS','T_PHRASE_H_P','T_PH_SAMPLE_PLAN','T_PH_SAMPLE_PLAN_EN','T_PH_SPEC_TYPE','T_PH_STAB_OVERAGE','T_P_STAB_PROFILE'," \
                "'T_PH_STAGE','T_PH_STB_STUDY_APPRV','T_PH_STB_SUM_RPT','T_PH_STB_SUM_RPT_ADR','T_PH_STB_SUM_RPT_SET','T_PH_TST_ARTICLE'," \
                "'T_PLANT_ORGANISM','T_PRINTER','T_REPORT_GALLERY','T_REPORT_GALLERY_ROW','T_REPORT_TEXT','T_SAMPLE_TYPE','T_SEACH_CONFIG'," \
                "'T_SEARCH_CONFIG_FLD','T_SEARCH_FILTER','T_SEARCH_FILTER_COL','T_SEARCH_FILTER_CRIT','T_SEARCH_TYPE','T_SEARCH_TPE_TBL'," \
                "'T_SERVER_ACTION_LOG','T_SETUP','T_SETUP_CONFIG_SYS','T_SETUP_FILES','T_SETUP_LIMS_CONST','T_SETUP_NOTES','T_SETUP_BJ_PATHS'," \
                "'T_SETUP_SQL','T_STOCK_SUPPLIER','T_SUITE','T_VWF_DETAIL_LINK','T_WINDOW_ABBREV','UNITS','USER_DIALOG_TEMP','USERDLG_MENUS'," \
                "'USER_DLG_TMP_FLDS','USER_DLG_TMP_GRPS','USER_ROLE','VARIABLE_CHART','VENDOR','VISUAL_WORKFLOW','VW_HTML_ELEMENT','VW_HTML_PARAM'," \
                "'WEB_CONFIG','WEB_SERVICE','WEB_SUBROUTINE','WORKBOOK_TEMPLATE','WORKFLOW_DETAIL','WORKSPACE','WORKSTATIO_ENTRY','WORKSTATION_TYPE'," \
                "'WS_CONTENT'"       
        elif auditLog == 'Lot':   
            column_names = ["AUDIT_COUNTER","TABLE_NAME","AUDIT_TYPE",
                            "AUDIT_ACTION","REASON","TRANS_ORDER","USER_NAME",
                            "AUDIT_TIMESTAMP","RECORD_SIGNED","TRANS_STRING",
                            "PARENT_TABLE","PARENT_KEY","USER_ROLE"]                             
            sql_command = f"""select counter,table_name,audit_type,
                    action,reason,trans_order,user_name,
                    audit_timestamp,record_signed,trans_string,
                    parent_table,parent_key,user_role " 
                    from lot_audit_uncrptwhere table_name not like 'T_%' 
                    group by counter,table_name,audit_type,action,reason,trans_order,user_name, 
                    audit_timestamp,record_signed,trans_string,parent_table,parent_key,user_role  
                    {time_clause}"""
            fieldDataTypeTables = "'LOT','LOT_SAMPLING_POINT'"
        elif auditLog == 'Batch':   
            column_names = ["AUDIT_COUNTER","TABLE_NAME","AUDIT_TYPE",
                            "AUDIT_ACTION","REASON","TRANS_ORDER","USER_NAME",
                            "AUDIT_TIMESTAMP","RECORD_SIGNED","TRANS_STRING",
                            "BATCH_NAME","TABLE_KEY","USER_ROLE"]                             
            sql_command = f"""select counter,table_name,audit_type,
                    action,reason,trans_order,user_name,
                    audit_timestamp,record_signed,trans_string,
                    batch_name,table_key,user_role 
                    from batch_audit_uncrptwhere table_name not like 'T_%'
                    group by counter,table_name,audit_type,action,reason,trans_order,user_name, 
                    audit_timestamp,record_signed,trans_string,batch_name,table_key,user_role 
                    {time_clause}"""
            fieldDataTypeTables = "'BATCH','BATCH_RESULT'"
        elif auditLog == 'Project':   
            column_names = ["AUDIT_COUNTER","TABLE_NAME","AUDIT_TYPE",
                            "AUDIT_ACTION","REASON","TRANS_ORDER","USER_NAME",
                            "AUDIT_TIMESTAMP","RECORD_SIGNED","TRANS_STRING",
                            "PROJECT_NAME","TABLE_KEY","USER_ROLE"]                              
            sql_command = f"""select counter,table_name,audit_type,
                    action,reason,trans_order,user_name,
                    audit_timestamp,record_signed,trans_string,
                    project_name,table_key,user_role 
                    from project_audit_uncrptwhere table_name not like 'T_%'
                    group by counter,table_name,audit_type,action,reason,trans_order,user_name, 
                    audit_timestamp,record_signed,trans_string,project_name,table_key,user_role 
                    {time_clause}"""
            fieldDataTypeTables = "'SAMPLE','PROJECT','PROJECT_RESULT','T_PH_STAB_OVERAGE'" \
                "'T_PH_STAB_AMENDMENTS'"
        elif auditLog == 'Protocol':   
            column_names = ["AUDIT_COUNTER","TABLE_NAME","AUDIT_TYPE",
                            "AUDIT_ACTION","REASON","TRANS_ORDER","USER_NAME",
                            "AUDIT_TIMESTAMP","TRANS_STRING",
                            "PROJECT_NAME","TABLE_KEY","USER_ROLE"]                              
            sql_command = f"""select counter,table_name,audit_type,
                    "action,reason,trans_order,user_name,
                    "audit_timestamp,trans_string,
                    "parent_table,table_key,name 
                    "from protocol_audit_uncrpt where table_name not like 'T_%'
                    "group by counter,table_name,audit_type,action,reason,trans_order,user_name, 
                    "audit_timestamp,trans_string,parent_table,table_key,name 
                    {time_clause}"""
            fieldDataTypeTables = "'PROTOCOL','PROT_CELL_SAMPLE','PROT_CELL_TEST'"
        elif auditLog == 'Stor_loc':   
            column_names = ["AUDIT_COUNTER","AUDIT_TYPE",
                            "AUDIT_ACTION","REASON","TRANS_ORDER","USER_NAME",
                            "AUDIT_TIMESTAMP","RECORD_SIGNED","TRANS_STRING",
                            "LOCATION_NUMBER"]                              
            sql_command = f"""select counter,audit_type,
                    action,reason,trans_order,user_name,
                    audit_timestamp,record_signed,trans_string,
                    location_number 
                    from stor_loc_audit_uncrpt
                    group by counter,audit_type,action,reason,trans_order,user_name, 
                    audit_timestamp,record_signed,trans_string,location_number 
                    {time_clause}"""
            fieldDataTypeTables = "'LOCATION'"   
        elif auditLog == 'Reports':   
            column_names = ["AUDIT_COUNTER","AUDIT_TYPE",
                            "AUDIT_ACTION","REASON","TRANS_ORDER","USER_NAME",
                            "AUDIT_TIMESTAMP","RECORD_SIGNED","TRANS_STRING",
                            "REPORT_NUMBER"]                              
            sql_command = f"""select counter,audit_type,
                    action,reason,trans_order,user_name,
                    audit_timestamp,record_signed,trans_string,
                    report_number 
                    from reports_audit_uncrpt 
                    group by counter,audit_type,action,reason,trans_order,user_name, 
                    audit_timestamp,record_signed,trans_string,report_number 
                    {time_clause}"""
            fieldDataTypeTables = "'REPORTS'"  
        elif auditLog == 'Investigation':   
            column_names = ["AUDIT_COUNTER","TABLE_NAME","AUDIT_TYPE",
                            "AUDIT_ACTION","REASON","TRANS_ORDER","USER_NAME",
                            "AUDIT_TIMESTAMP","RECORD_SIGNED","TRANS_STRING",
                            "INVESTIGATION","TABLE_KEY","USER_ROLE"]                              
            sql_command = f"""select counter,table_name,audit_type,
                    action,reason,trans_order,user_name,
                    audit_timestamp,record_signed,trans_string,
                    investigation,table_key,user_role 
                    from invest_audit_uncrpt where table_name not like 'T_%'
                    group by counter,table_name,audit_type,action,reason,trans_order,user_name, 
                    audit_timestamp,record_signed,trans_string,investigation,table_key,user_role 
                    {time_clause}"""
            fieldDataTypeTables = "'INVESTIGATION','INVEST_OBJECTS'"  
        elif auditLog == 'Order':   
            column_names = ["AUDIT_COUNTER","TABLE_NAME","AUDIT_TYPE",
                            "AUDIT_ACTION","REASON","TRANS_ORDER","USER_NAME",
                            "AUDIT_TIMESTAMP","RECORD_SIGNED","TRANS_STRING",
                            "ORDER_NUM","TABLE_KEY","USER_ROLE"]                              
            sql_command = f"""select counter,table_name,audit_type,
                    action,reason,trans_order,user_name,
                    audit_timestamp,record_signed,trans_string,
                    order_num,table_key,user_role 
                    from order_audit_uncrpt where table_name not like 'T_%'
                    group by counter,table_name,audit_type,action,reason,trans_order,user_name, 
                    audit_timestamp,record_signed,trans_string,order_num,table_key,user_role 
                    {time_clause}"""
            fieldDataTypeTables = "'ORDERS','ORDER_ENTRY'"
        elif auditLog == 'Formulation':   
            column_names = ["AUDIT_COUNTER","TABLE_NAME","AUDIT_TYPE",
                            "AUDIT_ACTION","REASON","TRANS_ORDER","USER_NAME",
                            "AUDIT_TIMESTAMP","RECORD_SIGNED","TRANS_STRING",
                            "FORMULATION","TABLE_KEY","USER_ROLE"]                              
            sql_command = f"""select counter,table_name,audit_type,
                    action,reason,trans_order,user_name,
                    audit_timestamp,record_signed,trans_string,
                    formulation,table_key,user_role 
                    from form_audit_uncrpt where table_name not like 'T_%'
                    group by counter,table_name,audit_type,action,reason,trans_order,user_name, 
                    audit_timestamp,trans_string,formulation,table_key,user_role 
                    {time_clause}"""
            fieldDataTypeTables = "'FORMULATION'"
        elif auditLog == 'Charge':   
            column_names = ["AUDIT_COUNTER","TABLE_NAME","AUDIT_TYPE",
                            "AUDIT_ACTION","REASON","TRANS_ORDER","USER_NAME",
                            "AUDIT_TIMESTAMP","RECORD_SIGNED","TRANS_STRING",
                            "PARENT_TABLE","PARENT_KEY"]                              
            sql_command = f"""select counter,table_name,audit_type,
                    action,reason,trans_order,user_name,
                    audit_timestamp,record_signed,trans_string,
                    parent_table,parent_key 
                    from charge_audit_uncrpt where table_name not like 'T_%'
                    group by counter,table_name,audit_type,action,reason,trans_order,user_name, 
                    audit_timestamp,record_signed,trans_string,parent_table,parent_key 
                    {time_clause}"""
            fieldDataTypeTables = "'CATALOGUE','CATALOGUE_ITEM','PRE_INVOICE','PRE_INVOICE_ITEM'," \
                "'PRE_INVOICE_ADJ','INVOICE','INVOICE_ITEM','INVOICE_ADJUSTMENT'"
        else:
            print('{auditLog} related information not set.')
            pass
              
        cursor.execute(sql_command)
        queryFetch = cursor.fetchall()
        auditLogDF = pd.DataFrame.from_records(queryFetch, columns = column_names) 

        # Retrieve datatypes for Dataframe
        # "string" | "float" | "real" | "double" | "int" | "long" | "numeric" | "categorical" | "factor" | "enum" | "time"
        sql_command = "select distinct FIELD_NAME, CASE WHEN FIELD_NAME = 'T_MLAR_SEVERITY' THEN 'categorical' WHEN DATA_TYPE " \
                    "in ('Text','Long','List','MultiList','Boolean','File','DBFile','Interval','DateTime','Date') THEN " \
                    "'string' WHEN DATA_TYPE in ('Integer','Number') THEN 'numeric' ELSE DATA_TYPE END AS DATATYPE from " \
                    "FIELD_MASTER where TABLE_NAME in ("+fieldDataTypeTables+") order by FIELD_NAME"

        with get_lims_connection().cursor() as lims_cursor_fetch_all:
            lims_cursor_fetch_all.execute(sql_command)
            fieldDatatypes = lims_cursor_fetch_all.fetchall()
            
        # Convert column datatypes to dictionary
        fieldDatatypesDF = pd.DataFrame.from_records(fieldDatatypes)
        fieldDatatypesDF.set_index(0, inplace=True)
        fieldDatatypes = fieldDatatypesDF[1].to_dict()

        # JSON data transformation 
        auditLogDF['TRANS_STRING'] = auditLogDF.TRANS_STRING.str.replace("\r\n", " ", regex=True)
        auditLogDF['TRANS_STRING'] = auditLogDF.TRANS_STRING.str.replace("\n", " ", regex=True)
        auditLogDF['TRANS_STRING'] = auditLogDF.TRANS_STRING.str.replace("\r", " ", regex=True)
        auditLogDF['TRANS_STRING'] = auditLogDF.TRANS_STRING.str.replace(r'":" ', r'":"', regex=True)
        auditLogDF['TRANS_STRING'] = auditLogDF.TRANS_STRING.str.replace(r'" ', r"'' ", regex=True)
        auditLogDF['TRANS_STRING'] = auditLogDF.TRANS_STRING.str.replace(r' "', r" ''", regex=True)
        auditLogDF['TRANS_STRING'] = auditLogDF.TRANS_STRING.str.replace(r'":" .', r'":"', regex=True) #remove trailing space between JSONs
        auditLogDF['TRANS_STRING'] = auditLogDF.TRANS_STRING.str.replace(r'. ":"', r'":"', regex=True) #remove leading space between JSONs
        auditLogDF['TRANS_STRING'] = auditLogDF.TRANS_STRING.str.replace(r"\\", r'\\\\', regex=True) #sanitise backslash
        auditLogDF['TRANS_STRING'] = auditLogDF.TRANS_STRING.str.replace(r'\("', r"(''", regex=True) #correct LIMS basic parameter quote start
        auditLogDF['TRANS_STRING'] = auditLogDF.TRANS_STRING.str.replace(r'"\)', r"'')", regex=True) #correct LIMS basic parameter quote end
        auditLogDF['TRANS_STRING'] = auditLogDF.TRANS_STRING.str.replace(r'", ', r"'', ", regex=True)
        auditLogDF['TRANS_STRING'] = auditLogDF.TRANS_STRING.str.replace(r' ,"', r" ,''", regex=True)
        auditLogDF['TRANS_STRING'] = auditLogDF.TRANS_STRING.str.replace(r'&\"', r"&''", regex=True)
        auditLogDF['TRANS_STRING'] = auditLogDF.TRANS_STRING.str.replace(r"''}", r'"}', regex=True) #terminate string
        auditLogDF['TRANS_STRING'] = auditLogDF.TRANS_STRING.str.replace(r"(''\w{3,})\",\"", r"\1'',''", regex=True) #correct LIMS basic parameter comma seperation
        auditLogDF['TRANS_STRING'] = auditLogDF.TRANS_STRING.str.replace(r'(\(\w{1,}),\"', r"\1,''", regex=True) #correct LIMS basic parameter comma seperation
        auditLogDF['TRANS_STRING'] = auditLogDF.TRANS_STRING.str.replace(r",,\"(\w{1,})''", r",,''\1''", regex=True) #correct LIMS basic parameter comma seperation
        auditLogDF['TRANS_STRING'] = auditLogDF.TRANS_STRING.str.replace(r'\",([^\"]\))', r"'',\1", regex=True) 
        auditLogDF['TRANS_STRING'] = auditLogDF.TRANS_STRING.str.replace(r'([^\"]),\"', r"\1,''", regex=True) 
        auditLogDF['TRANS_STRING'] = auditLogDF.TRANS_STRING.str.replace(r'(\w{2,})\",\"(\w{1,}[.]\w{3,} \w{2}'')', r"\1'',''\2", regex=True)
        auditLogDF['TRANS_STRING'] = auditLogDF.TRANS_STRING.str.replace(r"(''\w{1,})\",\"(\w{1,}'')", r"\1'',''\2", regex=True)
        auditLogDF['TRANS_STRING'] = auditLogDF.TRANS_STRING.str.replace(r'(\w{1,})\",\"(\w{1,})\",\"(\w{1,})', r"\1'',''\2'',''\3", regex=True)

        # BELOW MARKED FOR REMOVAL - RK 
        #auditLogDF['TRANS_STRING'] = auditLogDF.TRANS_STRING.str.replace(r' <,"', r" <,''", regex=True)
        #auditLogDF['TRANS_STRING'] = auditLogDF.TRANS_STRING.str.replace(r'> <"', r'" "', regex=True)
        #auditLogDF['TRANS_STRING'] = auditLogDF.TRANS_STRING.str.replace(r'<}', r'"}', regex=True)
        #auditLogDF['TRANS_STRING'] = auditLogDF.TRANS_STRING.str.replace(r'""[A-Z]', r'"[A-Z]', regex=True)
        #auditLogDF['TRANS_STRING'] = auditLogDF.TRANS_STRING.str.replace(r'\\', r' ', regex=True)
        #auditLogDF['TRANS_STRING'] = auditLogDF.TRANS_STRING.str.replace(r'[A-Z]"x', r'[A-Z] x', regex=True) #customer specific
        #auditLogDF['TRANS_STRING'] = auditLogDF.TRANS_STRING.str.replace(r'[A-Z]"","', r'[A-Z]","' , regex=True)
        #auditLogDF['TRANS_STRING'] = auditLogDF.TRANS_STRING.str.replace(r':', r'\:' , regex=True)
        #auditLogDF['TRANS_STRING'] = auditLogDF.TRANS_STRING.str.replace(r'[A-Z]"\:"[A-Z]', r'[A-Z]":"[A-Z]' , regex=True)
        #auditLogDF['TRANS_STRING'] = auditLogDF.TRANS_STRING.str.replace(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}\:\d{2}' , regex=True)
        #auditLogDF['TRANS_STRING'] = auditLogDF.TRANS_STRING.str.replace(r'[0-9]:[0-9]:[0-9]', r'[0-9]:[0-9]\:[0-9]', regex=True)

        def loads_json(x):
            try:
                return json.loads(x, strict=False) if pd.notna(x) else pd.Series() #
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                print(f"Problematic JSON string: {x}")
                return pd.Series()
            
        auditLogDF['TRANS_STRING'] = auditLogDF['TRANS_STRING'].apply(loads_json)
        jsonDF = pd.json_normalize(auditLogDF['TRANS_STRING'])

        # Tag JSON fields, replace chars
        jsonDF = jsonDF.replace(np.nan, "█").replace("▄", ",", regex=True)
        jsonDF = jsonDF.add_suffix('_JSON')

        # Join audit log and json dataframes
        auditLogDF = auditLogDF.drop(columns=['TRANS_STRING'])
        rawDataFrame = pd.concat([auditLogDF, jsonDF], axis=1)

        # Tag with source DB
        if not customerName:
            customerName = ""
        else:    
            rawDataFrame['SOURCE_DB'] = customerName

        # Convert to table for parquet conversion
        table = pa.Table.from_pandas(rawDataFrame)

        # Convert to parquet
        fileName = dataLocation + "raw_output_" + auditLog + '_' + customerName +'.parquet'
        pq.write_table(table,fileName)
        print("AAR - 1 - Raw Data Extraction completed")

get_lims_connection().close()
cursor.close()