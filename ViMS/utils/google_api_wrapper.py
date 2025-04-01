#!/opt/py37_env/bin/python3.7

import sys
import json
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Google API Scope
SCOPE = ["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/documents"]
credentials_file = '/data/victoria/vims-454810-b4d44dd0769a.json'

# Load credentials
def get_credentials():
    return ServiceAccountCredentials.from_json_keyfile_name(credentials_file, SCOPE)

def check_or_create_doc(doc_name='ViMS Pipeline Log'):
    """
    Check if the Google Doc exists; if not, create it.
    """
    credentials = get_credentials()
    docs_service = build("docs", "v1", credentials=credentials)
    drive_service = build("drive", "v3", credentials=credentials)
    
    # Try to get the list of Google Docs
    results = drive_service.files().list(q="mimeType='application/vnd.google-apps.document'").execute()
    files = results.get('files', [])

    # Check if the document already exists
    for file in files:
        if file['name'] == doc_name:
            return None # The Doc already exists

    # If the document does not exist, create it
    document = docs_service.documents().create(body={"title": doc_name}).execute()

def get_doc_id(doc_name='ViMS Pipeline Log'):
    """
    Get the Google Doc ID.
    """
    credentials = get_credentials()
    docs_service = build("docs", "v1", credentials=credentials)
    drive_service = build("drive", "v3", credentials=credentials)

    result = drive_service.files().list(q="mimeType='application/vnd.google-apps.document'").execute()
    files = result.get('files', [])
    for file in files:
        if file['name'] == doc_name:
            return file['id']

    return None

check_or_create_doc()

def clear_doc():
    """
    Clears all content from the Google Document specified by doc_id.
    """
    # Authenticate using service account
    credentials = get_credentials()
    docs_service = build("docs", "v1", credentials=credentials)
    doc_id = get_doc_id()

    # Get the document structure to find the endIndex
    doc = docs_service.documents().get(documentId=doc_id).execute()
    content = doc.get("body", {}).get("content", [])

    # Find the last endIndex
    if len(content) > 1:
        end_index = content[-1].get("endIndex", 1) - 1  # Avoid deleting the newline character
    else:
        end_index = 1  # If document is already empty

    # Check if there's anything to delete
    if end_index <= 1:
        return

    # Define request to delete content
    requests = [
        {
            "deleteContentRange": {
                "range": {
                    "startIndex": 1,
                    "endIndex": end_index,
                }
            }
        }
    ]

    # Execute batch update request
    docs_service.documents().batchUpdate(
        documentId=doc_id, body={"requests": requests}
    ).execute()



def setup_doc():
    """
    Set up the document (set the title, style first line, and share with emails).
    """
    credentials = get_credentials()
    docs_service = build("docs", "v1", credentials=credentials)
    drive_service = build("drive", "v3", credentials=credentials)
    doc_id = get_doc_id()
    
    # Set the title and style the first line
    docs_service.documents().batchUpdate(
        documentId=doc_id,
        body={
            "requests": [
                {
                    "insertText": {
                        "location": {"index": 1},
                        "text": "ViMS Pipeline Log\n"
                    }
                },
                {
                    "updateTextStyle": {
                        "range": {
                            "startIndex": 1,
                            "endIndex": 19  # Length of "ViMS Pipeline Log"
                        },
                        "textStyle": {
                            "bold": True,
                            "fontSize": {"magnitude": 24, "unit": "PT"},
                            "foregroundColor": {"color": {"rgbColor": {"blue": 1.0}}}
                        },
                        "fields": "bold, fontSize, foregroundColor"
                    }
                }
            ]
        }
    ).execute()

    # Share with anyone with the link
    drive_service.permissions().create(
        fileId=doc_id,
        body={'role': 'reader', 'type': 'anyone'},  # Makes the document publicly accessible for reading
    ).execute()

# Append a log entry to Google Docs
def append_log_entry(step_name, status, warnings=None, plot_link=None):
    credentials = get_credentials()
    service = build("docs", "v1", credentials=credentials)
    doc_id = get_doc_id()
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"{timestamp} - {step_name}: {status}\n"
    if warnings:
        log_entry += f"Warnings: {warnings}\n"
    if plot_link:
        log_entry += f"Plot Link: {plot_link}\n"

    # Get the current length of the document
    document = service.documents().get(documentId=doc_id).execute()
    end_index = document['body']['content'][-1]['endIndex']

    # Then append text at the end of the document
    requests = [
        {
            'insertText': {
                'location': {
                    'index': end_index - 1,  # Insert at the end of the document
                },
                'text': log_entry
            }
        }
    ]  

    service.documents().batchUpdate(documentId=doc_id, body={'requests': requests}).execute()


# Upload a plot to Google Drive
def upload_plot_to_drive(plot_path):
    credentials = get_credentials()
    drive_service = build('drive', 'v3', credentials=credentials)
    doc_id = get_doc_id()

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

def initialize_google_doc():
    """
    Initializes the Google Document only once per pipeline execution.
    It creates the doc if it does not exist, clears it, and sets up permissions.
    """
    check_or_create_doc()  # Create if it doesn't exist
    clear_doc()            # Clear previous logs
    setup_doc()            # Setup title, formatting, sharing

    # Print the document link only once
    doc_id = get_doc_id()
    print(f"Google Doc link: https://docs.google.com/document/d/{doc_id}/edit")

# CLI Interface (so Python 3.10 can call it)
if __name__ == "__main__":
    command = sys.argv[1]

    if command == "init_doc":
        initialize_google_doc()  # Run once before starting the pipeline
    
    elif command == "append_log":
        step_name = sys.argv[2]
        status = sys.argv[3]
        warnings = sys.argv[4] if len(sys.argv) > 4 else ""
        plot_link = sys.argv[5] if len(sys.argv) > 5 else ""
        append_log_entry(step_name, status, warnings, plot_link)
        
    elif command == "upload_plot":
        plot_path = sys.argv[2]
        print(upload_plot_to_drive(plot_path))  # Return the link