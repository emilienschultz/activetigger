import streamlit as st
import threading
import plotly.graph_objects as go
import json
import requests as rq
import pandas as pd
import time
import asyncio
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import importlib
import numpy as np
from io import BytesIO
import textwrap
from streamlit_autorefresh import st_autorefresh


import urllib3
from urllib3.exceptions import InsecureRequestWarning
urllib3.disable_warnings(InsecureRequestWarning)

import streamlit as st

URL_SERVER = "http://127.0.0.1:5000"
update_time = 2000
count = st_autorefresh(interval=update_time, limit=None, key="fizzbuzzcounter")

if not "header" in st.session_state:
    st.session_state.header = None

# Internal functions
# ------------------

def _post(route:str, 
        params:dict|None = None, 
        files:str|None = None,
        json_data:dict|None = None,
        data:dict|None = None) -> dict:
    """
    Post to API
    """
    url = URL_SERVER + route
    r = rq.post(url, 
                params = params,
                json = json_data,
                data = data,
                files = files,
                headers = st.session_state.header, 
                verify = False)
    #print(url, r.content, st.session_state.header)
    if r.status_code == 422:
        return {"status":"error", "message":"Not authorized"}
    return json.loads(r.content)

def _get(route:str, 
        params:dict|None = None, 
        data:dict|None = None,
        is_json = True) -> dict:
    """
    Get from API
    """
    url = URL_SERVER + route
    r = rq.get(url, 
                params = params,
                data = data,
                headers = st.session_state.header,
                verify=False)
    #print(url, r.content, st.session_state.header)
    if r.status_code == 422:
        return {"status":"error", "message":"Not authorized"}
    if is_json:
        return json.loads(r.content)
    return r.content

def _connect_user(user:str, password:str) -> bool:
    """
    Connect account and get auth token
    """
    form = {
            "username":user,
            "password":password
            }
    
    r = _post("/token", data = form)
    if not "access_token" in r:
        print(r)
        return False

    # Update widget configuration
    st.session_state.header = {"Authorization": f"Bearer {r['access_token']}", "username":user}
    st.session_state.user = user
    return True

def _get_state() -> dict:
    """
    Get state variable
    """
    # only if a current project is selected
    if "current_project" in st.session_state:
        state = _get(route = f"/state/{st.session_state.current_project}")
        if state["status"]=="error":
            print(state)
            return {}
        return state["data"]
    return {}

def _delete_project(project_name:str) -> dict:
    """
    Delete existing project
    """
    params = {
            "project_name": project_name,
            "user":st.session_state.user
            }  
    r = _post(route = "/projects/delete", 
              params = params)
    return r

def _create_scheme(s:str):
    """
    Create new scheme
    """
    if s == "":
        return "Empty"
    params = {
            "project_name":st.session_state.current_project
            }
    data = {
        "project_name": st.session_state.current_project,
        "name":s,
        "tags":[],
            }
    r = _post("/schemes/add", 
                params = params, 
                json_data = data)
    return r

def _create_label(label:str):
    """
    Create label in a scheme
    """
    if label == "":
        return "Empty"
    params = {"project_name":st.session_state.current_project,
                "scheme": st.session_state.current_scheme,
                "label":label,
                "user":st.session_state.user}
    r = _post("/schemes/label/add", 
                    params = params)
    return r

def _delete_scheme(s:str):
    """
    Delete scheme
    """
    if s == "":
        return "Empty"
    params = {"project_name":st.session_state.current_project}
    data = {
            "project_name":st.session_state.current_project,
            "name":s,
            }
    r = _post("/schemes/delete", 
                params = params, 
                json_data = data)
    return r

def _delete_label(label:str):
    """
    Delete label in a scheme
    """
    if label == "":
        return "Empty"
    params = {"project_name":st.session_state.current_project,
                "scheme":st.session_state.current_scheme,
                "label":label,
                "user":st.session_state.user}
    r = _post("/schemes/label/delete", 
                    params = params)
    return r

def _delete_feature(feature_name) -> bool:
    """
    Delete existing feature
    """
    r = _post(f"/features/delete/{feature_name}", 
                params = {"project_name":st.session_state.current_project,
                            "user":st.session_state.user})
    return True

def _add_feature(feature_name, feature_params) -> bool:
    """
    Compute feature
    """
    if not feature_name in st.session_state.state["features"]["options"].keys():
        return "This feature doesn't exist"
    try:
        feature_params = json.loads(feature_params)
    except:
        raise ValueError("Problem in the json parameters")
    r = _post(f"/features/add/{feature_name}", 
                params ={
                        "project_name":st.session_state.current_project,
                        "user":st.session_state.user
                        },
                json_data = {"params":feature_params}
                )
    return True

def _add_regex(value:str, name:str|None = None) -> bool:
    """
    Add regex as feature
    """
    if name is None:
        name = value
    
    name = f"regex_{st.session_state.user}_{name}"

    data = {
        "project_name":st.session_state.current_project,
        "name":name,
        "value":value,
        "user":st.session_state.user
        }
    
    r = _post("/features/add/regex",
        params = {"project_name":st.session_state.current_project},
        json_data=data)
    return True

# Interface organization
#-----------------------

def main():
    # initialize variables
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'page' not in st.session_state:
            st.session_state['page'] = "Projects"
    st.session_state.state = _get_state()

    # start the interface
    if not st.session_state['logged_in']:
        login_page()
    if st.session_state['logged_in']:
        app_navigation()

def login_page():
    """
    Page to log in
    """
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if _connect_user(username, password):
            st.session_state['logged_in'] = True
        else:
            st.error("Incorrect username or password")

def app_navigation():
    """
    Select page
    """
    # creating the menu
    st.sidebar.title("Menu")
    options = ["Projects", "Schemes", "Features", "Annotate"]
    st.session_state['page'] = st.sidebar.radio("Navigate", 
                                                options, 
                                                key="menu", 
                                                index = options.index(st.session_state['page']))

    # navigating
    if st.session_state['page'] == "Projects":
        projects()
    elif st.session_state['page'] == "Schemes":
        schemes()
    elif st.session_state['page'] == "Features":
        features()
    elif st.session_state['page'] == "Annotate":
        annotate()

def projects():
    """
    Projects page
    - select a project
    - create one
    """
    if not "new_project" in st.session_state:
        st.session_state.new_project = False
    r = _get("/server")
    existing = r["data"]["projects"]

    # display menu
    st.title("Projects")
    st.write("Load or create project")
    option = st.selectbox(
        "Select existing projects:",
        existing)
    
    col1, col2 = st.columns(2)

    # load a project
    with col1:
        if st.button("Load"):
            st.session_state.current_project = option
            st.session_state.page = "Schemes"
            return None
    
    # delete a project
    with col2:
        if st.button("Delete"):
            st.write(f"Deleting{option}")
            _delete_project(option)

    st.markdown("<hr>", unsafe_allow_html=True)

    # create a project
    if st.button("New project"):
        st.session_state['new_project'] = True

    # display the creation menu
    if st.session_state.get('new_project', True):
        project_name = st.text_input("Project name", value="")
        dic_langage = {"French":"fr",
                       "English":"en",
                       "Spanish":"es"}
        language = st.selectbox("Language:",list(dic_langage))
        file = st.file_uploader("Load file (CSV or Parquet)", 
                                type=['csv', 'parquet'], 
                                accept_multiple_files=False)
        if file:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith('.parquet'):
                df = pd.read_parquet(file)
            else:
                st.error("Type not supported")
            st.dataframe(df.head())
            st.write("Select columns")
            column_id = st.selectbox("Ids:",list(df.columns))
            column_text = st.selectbox("Texts:",list(df.columns))
            column_label = st.selectbox("Labels:",list(df.columns))
            columns_context = st.multiselect("Context:",list(df.columns))
            n_train = st.number_input("N train", min_value=100, max_value=len(df),key="n_train")
            n_test = st.number_input("N test", min_value=100, max_value=len(df),key="n_test")

            if st.button("Create"):
                data = {
                        "project_name": project_name,
                        "user":st.session_state.user,
                        "col_text": column_text,
                        "col_id":column_id,
                        "col_label":column_label,
                        "cols_context": columns_context,
                        "n_train":n_train,
                        "n_test":n_test, 
                        "language":dic_langage[language]
                        }
                print(data)
                buffer = BytesIO()
                df.to_csv(buffer)
                buffer.seek(0)
                files = {'file': (file.name, buffer)}
                r = _post(route="/projects/new", 
                         files=files,
                         data=data
                         )
                print(r)
                if r["status"] == "error":
                    print(r["message"])
                
    return None


def schemes():
    """
    Scheme page
    """
    st.title("Schemes")
    st.write("Interface to manage schemes & label")
    st.subheader("Schemes")
    options_schemes = list(st.session_state.state["schemes"]["available"].keys())
    scheme = st.selectbox(label="",options = options_schemes, index=None, placeholder="Select a scheme")
    st.session_state.current_scheme = scheme # select scheme
    if st.button("Delete scheme"):
        if scheme is not None:
            st.write(f"Deleting scheme {scheme}")
            _delete_scheme(scheme)
    new_scheme = st.text_input(label="", placeholder="New scheme name")
    if st.button("Create scheme"):
        if new_scheme is not None:
            st.write(f"Creating scheme {new_scheme}")
            _create_scheme(new_scheme)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Labels")
    options_labels = []
    if st.session_state.current_scheme is not None:
        options_labels = st.session_state.state["schemes"]["available"][st.session_state.current_scheme]
    label = st.selectbox(label="",options = options_labels, index=None, placeholder="Select a label")
    if st.button("Delete label"):
        if label is not None:
            st.write(f"Deleting label {label}")
            _delete_label(label)
    new_label = st.text_input(label="", placeholder="New label name")
    if st.button("Create label"):
        if new_label is not None:
            st.write(f"Creating label {new_label}")
            _create_label(new_label)

    st.markdown("<hr>", unsafe_allow_html=True)
    if st.button("Next step : compute features"):
        if st.session_state.current_scheme is not None:
            st.session_state.page = "Features"

def features():

    st.title("Features")
    st.write("Interface to manage features.")

    c = st.session_state.state["features"]["training"]
    if len(c) == 0:
        c = "None"
    st.html(f"<div style='background-color: #ffcc00; padding: 10px;'>Processes currently running: {c}</div>")

    feature = st.selectbox(label="Available",
                           options = st.session_state.state["features"]["available"])
    if st.button("Delete feature"):
        if feature is not None:
            st.write(f"Deleting feature {feature}")
            _delete_feature(feature)
    
    add_feature = st.selectbox(label="Add",
                           options = list(st.session_state.state["features"]["options"].keys()))
    
    params = ""
    if add_feature in st.session_state.state["features"]["options"]:
        params = st.session_state.state["features"]["options"][add_feature]
        params = st.text_area(label="", value = params)

    if st.button("Compute feature"):
        if add_feature is not None:
            st.write(f"Computing feature {add_feature}")
            _add_feature(add_feature, params)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.subheader("Add regex")
    regex = st.text_input(label="", placeholder="Write your regex")
    if st.button("Create regex"):
        if regex is not None:
            st.write(f"Computing regex {regex}")
            _add_regex(regex)

    st.markdown("<hr>", unsafe_allow_html=True)
    if st.button("Next step : annotate"):
        if st.session_state.current_scheme is not None:
            st.session_state.page = "Annotate"


def annotate():
    st.title("Annotate")
    st.write("Interface to annotate data.")

if __name__ == "__main__":
    main()

