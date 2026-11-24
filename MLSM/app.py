from flask import Flask, request, jsonify, render_template
from cryptography.fernet import Fernet
import mlhandler
import json
import os

app = Flask(__name__)

def get_or_create_key():
    if not os.path.exists('secret.key'):
        key = Fernet.generate_key()
        with open('secret.key', 'wb') as key_file:
            key_file.write(key)
        print("New encryption key generated and saved.")
    else:
        with open('secret.key', 'rb') as key_file:
            key = key_file.read()
        print("Existing encryption key loaded.")
    return key

# Initialize Fernet cipher
cipher_suite = Fernet(get_or_create_key())


def execute_python_script(data, process_type):   
    if not data:
         return {"status": "error", "error": "No data received.", "details": "The request body was empty."}, 400

    try:
        success, output_message = mlhandler.run_process(data, process_type)
        # -------------------------------
        
        if success:
            log = f"{process_type} Succeeded!\n\n"
            log += output_message
            return {"status": "success", "log_output": log}, 200
        else:
            # If the script returns False (an internal error)
            
            return {
                "status": "error", 
                "error": f"Script failed: {process_type}", 
                "details": output_message
            }, 500

    except ImportError:
        # Handle the case where ml_processor.py is missing
        return {
            "status": "error", 
            "error": "External script not found.", 
            "details": "Could not import 'ml_processor.py'. Ensure it is in the same directory as app.py."
        }, 500
    except Exception as e:
        # Handle unexpected errors during the call
        return {
            "status": "error", 
            "error": f"Internal Flask Error during {process_type} call.", 
            "details": str(e)
        }, 500

@app.route('/')
def index():                    
    try:
        with open('credentials.json', 'r') as file:
            credentials = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        credentials = {}
    # Flask looks for 'index.html' inside the 'templates' folder
    return render_template('index.html', saved_creds=list(credentials.keys()))                    
       
@app.route('/train', methods=['POST'])
def train_route():                                                     
    data = request.get_json()
    response, status = execute_python_script(data, "Training/Retraining")
    return jsonify(response), status

@app.route('/infer', methods=['POST'])
def infer_route():
    data = request.get_json()
    response, status = execute_python_script(data, "Inference")
    return jsonify(response), status

@app.route('/troubleshoot', methods=['POST'])
def troubleshoot_route():
    data = request.get_json()
    response, status = execute_python_script(data, "Troubleshooting")
    return jsonify(response), status

@app.route('/update_db_creds', methods=['POST'])
def update_cred_route():
    data = request.get_json()
    
    if not data or 'cred_name' not in data:
        return {"status": "error", "message": "Missing credential name."}, 400
 
    # Load existing credentials
    try:
        with open('credentials.json', 'r') as file:
            credentials = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        credentials = {}
 
    credentials[data['cred_name']] = {
        "db_driver": data.get('db_driver', ''),
        "db_database": data.get('db_database', ''),
        "db_username": data.get('db_username', ''),
        "db_password": data.get('db_password', ''),
        "db_hostname": data.get('db_hostname', ''),
        "db_port": data.get('db_port', '')
    }
 
    with open('credentials.json', 'w') as file:
        json.dump(credentials, file, indent=4)
 
    return {"status": "success", "message": "Credentials saved successfully."}, 201

@app.route('/delete_db_cred', methods=['POST'])
def delete_db_cred_route():
    data = request.get_json()
    cred_name = data.get('cred_name')
    try:
        with open('credentials.json', 'r') as file:
            credentials = json.load(file)
        if cred_name in credentials:
            del credentials[cred_name]
            with open('credentials.json', 'w') as file:
                json.dump(credentials, file, indent=4)
            return jsonify({"status": "success"}), 200
        return jsonify({"status": "error", "message": "Not found"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)