#!/usr/bin/env python3

import requests

def append_log(step_name, status, warnings="", plot_link="", doc_name="ViMS Pipeline Log"):
    url = "http://34.173.57.167:5000/append_log"
    data = {
        "step_name": step_name,
        "status": status,
        "warnings": warnings,
        "plot_link": plot_link,
        "doc_name": doc_name
    }
    response = requests.post(url, json=data)
    return response.json()

def upload_plot(plot_path):
    url = "http://34.173.57.167:5000/upload_plot"
    with open(plot_path, 'rb') as f:
        files = {'file': (plot_path.split('/')[-1], f, 'image/png')}
        response = requests.post(url, files=files)
    return response.json().get("plot_link")

def check_or_create_doc(doc_name="ViMS Pipeline Log"):
    url = "http://34.173.57.167:5000/check_or_create_doc"
    data = {"doc_name": doc_name}
    response = requests.post(url, json=data)
    return response.json()

def get_doc_id(doc_name="ViMS Pipeline Log"):
    url = "http://34.173.57.167:5000/get_doc_id"
    data = {"doc_name": doc_name}
    response = requests.post(url, json=data)
    return response.json().get("doc_id")

def clear_doc():
    url = "http://34.173.57.167:5000/clear_doc"
    response = requests.post(url)
    return response.json()

def setup_doc():
    url = "http://34.173.57.167:5000/setup_doc"
    response = requests.post(url)
    return response.json()

def initialize_google_docs():
    url = "http://34.173.57.167:5000/initialize_google_docs"
    response = requests.post(url)
    return response.json()

def create_table():
    url = "http://34.173.57.167:5000/create_table"
    response = requests.post(url)
    return response.json()

def update_cell(obs_id, column_name, content, is_image=False):
    url = "http://34.173.57.167:5000/update_cell"
    data = {"obs_id": obs_id, "column_name": column_name, "content": content, "is_image": is_image}
    response = requests.post(url, json=data)
    return response.json()