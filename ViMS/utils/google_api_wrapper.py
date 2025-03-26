#!/opt/py37_env/bin/python3.7

import sys
import json
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Google API Scope
SCOPE = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

# Load credentials
def get_credentials(credentials_file):
    return ServiceAccountCredentials.from_json_keyfile_name(credentials_file, SCOPE)

# Append a log entry to Google Sheets
def append_log_entry(credentials_file, sheet_id, step_name, status, warnings=None, plot_link=None):
    credentials = get_credentials(credentials_file)
    service = build("sheets", "v4", credentials=credentials)
    sheet = service.spreadsheets().values()

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    row = [timestamp, step_name, status, warnings if warnings else "", plot_link if plot_link else ""]
    
    sheet.append(spreadsheetId=sheet_id, range="A1", valueInputOption="RAW", body={"values": [row]}).execute()

# Upload a plot to Google Drive
def upload_plot_to_drive(credentials_file, plot_path):
    credentials = get_credentials(credentials_file)
    drive_service = build('drive', 'v3', credentials=credentials)

    file_metadata = {'name': plot_path.split('/')[-1]}
    media = MediaFileUpload(plot_path, mimetype='image/png')

    file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    file_id = file.get('id')

    # Make file publicly accessible
    drive_service.permissions().create(
        fileId=file_id,
        body={'role': 'reader', 'type': 'anyone'},
    ).execute()

    file_url = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
    return file_url

# CLI Interface (so Python 3.10 can call it)
if __name__ == "__main__":
    command = sys.argv[1]
    credentials_file = sys.argv[2]

    if command == "append_log":
        sheet_id = sys.argv[3]
        step_name = sys.argv[4]
        status = sys.argv[5]
        warnings = sys.argv[6] if len(sys.argv) > 6 else ""
        plot_link = sys.argv[7] if len(sys.argv) > 7 else ""
        append_log_entry(credentials_file, sheet_id, step_name, status, warnings, plot_link)
    
    elif command == "upload_plot":
        plot_path = sys.argv[3]
        print(upload_plot_to_drive(credentials_file, plot_path))  # Return the link

