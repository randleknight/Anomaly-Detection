import sys
import json
import pandas as pd

import sys

with open('credentials.json') as data_file:
    credentials = json.load(data_file)

if len(sys.argv) > 1:
    dbType = sys.argv[1]
    configCredName = sys.argv[2] 
    configTableLocation = sys.argv[3]
else:
    print("No variable received.")
    dbType = "postgres"
    configCredName = "Default"
    configTableLocation = "SAMPLE"


def get_lims_connection():
    global customerName
    if dbType == "postgres":
        import psycopg2
        odbcLimsConnection = psycopg2.connect(f'dbname={credentials[configCredName].get("db_database")} user= {credentials[configCredName].get("db_username")} password = {credentials[configCredName].get("db_password")} host= {credentials[configCredName].get("db_hostname")}')
        #odbcLimsConnection.setAutoCommit(true)
        return odbcLimsConnection
    
    elif dbType == "oracle":
        import oracledb
        oracledb.init_oracle_client(config.lib_dir)
        odbcLimsConnection = oracledb.connect(user=credentials[configCredName].get("db_username"),password=credentials[configCredName].get("db_password"),
                                            dsn=credentials[configCredName].get("db_dsn"))
        odbcLimsConnection.arraysize = 5000

    else:
        import pyodbc
        odbcLimsConnection = pyodbc.connect('DRIVER=' + credentials[configCredName].get("db_driver") + ';DATABASE=' + credentials[configCredName].get("db_database") +
                            ';UID=' + credentials[configCredName].get("db_username") + ';PWD=' + credentials[configCredName].get("db_password") +
                            ';SERVER=' + credentials[configCredName].get("db_hostname") + ';PORT=' + credentials[configCredName].get("db_port") +
                            ';Trusted_Connection=yes')
                

    if __name__ == '__main__':
        try:  # GET THE CONNECTION OBJECT (ENGINE) FOR THE DATABASE
            get_lims_connection()
        except Exception as error:
            exit(f"Connection could not be made due to the following error: {error}")

print("Testing Connection")
cursor = get_lims_connection().cursor()
sql_command = f"select * from {configTableLocation} LIMIT 1"
cursor.execute(sql_command)
response = cursor.fetchall()
print(response)

get_lims_connection().close()
cursor.close()
