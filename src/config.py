import os

CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'secrets', 'config.txt')

config_data = {}
with open(CONFIG_FILE, 'r') as f:
    for line in f:
        line = line.strip()
        if line and '=' in line:
            key, val = line.split('=', 1)
            config_data[key] = val

DB_SERVER = config_data.get('DB_SERVER', '')
DB_NAME = config_data.get('DB_NAME', '')
USE_WINDOWS_AUTH = config_data.get('USE_WINDOWS_AUTH', 'true').lower() == 'true'
DB_USER = config_data.get('DB_USER', '')
DB_PASSWORD = config_data.get('DB_PASSWORD', '')
GEMINI_API_KEY = config_data.get('GEMINI_API_KEY', '')
USE_OFFLINE_MODEL = config_data.get('USE_OFFLINE_MODEL', 'false').lower() == 'true'
OLLAMA_MODEL = config_data.get('OLLAMA_MODEL', 'llama3.2')
RETRIEVAL_TOP_N = int(config_data.get('RETRIEVAL_TOP_N', '10'))
FINAL_TOP_N = int(config_data.get('FINAL_TOP_N', '5'))

if GEMINI_API_KEY:
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
