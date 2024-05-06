import streamlit as st
import plotly.graph_objects as go
import json
import requests as rq
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import importlib
import numpy as np
from io import BytesIO
import textwrap
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# A faire
# - plotly rescale : mettre en paramètre les éléments pour ne pas les reseter à chaque fois


URL_SERVER = "http://127.0.0.1:5000"
update_time = 2000
count = st_autorefresh(interval=update_time, limit=None, key="fizzbuzzcounter")

if not "header" in st.session_state:
    st.session_state.header = None
if not "history" in st.session_state:
    st.session_state.history = []

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

def _get_next_element() -> bool:
    """
    Get next element from the current widget options
    """
    # try:
    #     f = visualization.children[0]
    #     x1y1x2y2 = [f['layout']['xaxis']['range'][0], 
    #                 f['layout']['yaxis']['range'][0],
    #                 f['layout']['xaxis']['range'][1], 
    #                 f['layout']['yaxis']['range'][1]]
    # except:
    #     x1y1x2y2 = []
    x1y1x2y2 = []

    params = {
            "project_name":st.session_state.current_project,
            "scheme":st.session_state.current_scheme,
            "selection":st.session_state.selection,
            "sample":st.session_state.sample,
            "user":st.session_state.user,
            "tag":st.session_state.tag,
            "frame":x1y1x2y2
            }
    
    r = _get(route = "/elements/next",
                    params = params)
    
    if r["status"] == "error":
        print(r["message"])
        st.write(r["message"])
        return False

    st.session_state.current_element = r["data"]
#    self._textarea.value = self.current_element["text"]
#    self.info_element.value = str(self.current_element["info"])
#    self.info_predict.value = f"Predict SimpleModel: <b>{self.current_element['predict']['label']}</b> (p = {self.current_element['predict']['proba']})"
    return True

def _send_tag(label):
    data = {
            "project_name":st.session_state.current_project,
            "scheme":st.session_state.current_scheme,
            "element_id":st.session_state.current_element["element_id"],
            "tag":label,
            "user":st.session_state.user,
            "selection":st.session_state.current_element["selection"] #mode of selection of the element
            }
    
    r = _post(route = "/tags/add",
                    params = {"project_name":st.session_state.current_project},
                    json_data = data)
    
    # add in history
    if "error" in r:
        st.write(r)
    else:
        st.session_state.history.append(st.session_state.current_element["element_id"])

    # TODO # check if simplemodel need to be retrained
    # if self.is_simplemodel() and (len(self.history) % self.simplemodel_autotrain.value == 0):
    #     sm = self.state["simplemodel"]["available"][self.user][self.select_scheme.value]
    #     self.create_simplemodel(self.select_scheme.value,
    #                 model = sm["name"], 
    #                 parameters = sm["params"], 
    #                 features = sm["features"])


def _display_element(element_id):
    """
    Display specific element
    """
    r = _get(route = f"/elements/{element_id}",
                    params = {"project_name":st.session_state.current_project,
                            "scheme":st.session_state.current_scheme})
    # Managing errors
    if r["status"]=="error":
        print(r)
        return False
    # Update interface
    print(r["data"])
    st.session_state.current_element = r["data"]
    return True

def _get_previous_element() -> bool:
    """
    Load previous element in history
    """
    if len(st.session_state.history) == 0:
        st.write("No element in history")
        return False
    element_id = st.session_state.history.pop()
    r = _display_element(element_id) 
    return r

def _compute_projection():
    """
    Start computing projection
    """
    params = {
            "project_name":st.session_state.current_project,
            "user":st.session_state.user
            }
    
    try:
        proj_params = json.loads(st.session_state.projection_params)
    except:
        raise ValueError("Problem in the json parameters")

    data = {
        "method":st.session_state.projection_method, 
        "features":st.session_state.projection_features,
        "params":proj_params,
        }
    
    r = _post("/elements/projection/compute",
        params = params,
        json_data = data)
    if r["status"] == "waiting":
        st.session_state.projection_data = "computing"
        st.write(st.session_state.projection_data)
    else:
        print(r)

def _plot_visualisation():
    """
    Produce the visualisation for the projection
    """
    df = st.session_state.projection_data
    df["to_show"] = df.apply(lambda x : f"{x.name}<br>{'<br>'.join(textwrap.wrap(x['texts'][0:300],width=30))}...", axis=1)
    f = go.FigureWidget([go.Scatter(x=df[df["labels"]==i]["0"], 
                                    y=df[df["labels"]==i]["1"], 
                                    mode='markers', 
                                    name = i,
                                    customdata = np.stack((df[df["labels"]==i]["to_show"],), axis=-1),
                                    hovertemplate="%{customdata[0]}",
                                    showlegend = True) for i in df["labels"].unique()])
    f.layout.hovermode = 'closest'
    def update_point(trace, points, selector):
        # select specific text
        if len(points.point_inds)>0:
            element_id = trace.customdata[points.point_inds][0][0].split("<br>")[0] #TODO améliorer
            _display_element(element_id)
    for i in range(0,len(f.data)):
        f.data[i].on_click(update_point)
    return f

def _get_projection_data():
    """
    Get projection data
    """
    params = {
            "project_name":st.session_state.current_project, 
            "user":st.session_state.user,
            "scheme":st.session_state.current_scheme
            }
    r = _get("/elements/projection/current",
        params = params)
    return r

def _get_statistics():
    params = {"project_name":st.session_state.current_project, 
            "scheme":st.session_state.current_scheme, 
            "user":st.session_state.user}
    r = _get("/description",params = params)
    if r["status"]=="error":
        return r["message"]
    text = ""
    for k,v in r["data"].items():
        text += f"<br>- <b>{k}</b>: {v}"
    return text

def _get_table():
    """
    Get data as a table
    """
    params = {
                "project_name":st.session_state.current_project,
                "scheme":st.session_state.current_scheme,
                "min":st.session_state.data_min,
                "max":st.session_state.data_max,
                "mode":st.session_state.data_mode
                }
    r = _get("/elements/table", params = params)
    df = pd.DataFrame(r["data"])
    return df

def _send_table():
    """
    Send table modified
    """
    data = {
        "scheme":st.session_state.current_scheme,
        "list_ids":list(st.session_state.data_df.index),
        "list_labels":list(st.session_state.data_df["labels"])
    }
    r = _post("/elements/table", 
                json_data = data, 
                params = {"project_name":st.session_state.current_project,
                            "user":st.session_state.user
                            })
    if r["status"] == "error":
        st.write(r["message"])
    st.write("Data saved")

# Interface organization
#-----------------------

def main():
    # initialize variables
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'page' not in st.session_state:
            st.session_state['page'] = "Projects"
    st.session_state.state = _get_state()   
    st.sidebar.write(datetime.datetime.now())
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
    options = ["Projects",
               "Schemes",
               "Features",
               "Annotate",
               "Description",
               "Active Model",
               "Global Model",
               "Export"]
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
    elif st.session_state['page'] == "Description":
        description()
    elif st.session_state['page'] == "Active Model":
        simplemodels()
    elif st.session_state['page'] == "Global Model":
        bertmodels()
    elif st.session_state['page'] == "Export":
        export()

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
    scheme = st.selectbox(label="Current scheme:", options = options_schemes, index=0, placeholder="Select a scheme")
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
    """
    Feature page
    """
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
    """
    Annotate page
    """
    # default options
    mode_selection = st.session_state.state["next"]["methods_min"]
    mode_sample = st.session_state.state["next"]["sample"]
    if "selection" not in st.session_state:
        st.session_state.selection = mode_selection[0]
    if "sample" not in st.session_state:
        st.session_state.sample = mode_sample[0]
    if "tag" not in st.session_state:
        st.session_state.tag = None
        
    # get next element with the current options
    if "current_element" not in st.session_state:
        _get_next_element()

    # display page
    st.title("Annotate")
    st.write("Interface to annotate data.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.selectbox(label="", options = mode_selection, key = "selection")
    with col2:
        st.selectbox(label="", options = mode_sample, key = "sample")
    with col3:
        st.selectbox(label="", options = [], key = "tag")
    if st.button("Back"):
                st.write("Back")
                _get_previous_element()
    st.markdown(f"""
        <div>{st.session_state.current_element["predict"]}</div>
        <div style="
            border: 2px solid #4CAF50;
            padding: 10px;
            border-radius: 5px;
            color: #4CAF50;
            font-family: sans-serif;
            text-align: justify;
            margin: 10px;
        ">
            {st.session_state.current_element["text"]}
        </div>

    """, unsafe_allow_html=True)

    labels = st.session_state.state["schemes"]["available"][st.session_state.current_scheme]
    cols = st.columns(len(labels))
    for col, label in zip(cols, labels):
        with col:
            if st.button(label):
                _send_tag(label)
                _get_next_element()

    st.markdown("<hr>", unsafe_allow_html=True)

    # managing projection display
    if "projection" not in st.session_state:
        st.session_state.projection = False
    # switch button
    if st.button("Projection"):
        if not st.session_state.projection:
            st.session_state.projection = True
        else:
            st.session_state.projection = False
    # displaying menu
    if st.session_state.projection:
        st.selectbox(label="Method", 
                     options = list(st.session_state.state["projections"]["available"].keys()), 
                     key = "projection_method")
        st.text_area(label="", 
                     value=json.dumps(st.session_state.state["projections"]["available"]["umap"], indent=2),
                     key = "projection_params")
        st.multiselect(label="Features", options=st.session_state.state["features"]["available"],
                       key = "projection_features")
        if st.button("Compute"):
            st.write("Computing")
            _compute_projection()
        
        # if visualisation available, display it
        if ("projection_data" in st.session_state) and (type(st.session_state.projection_data) == str):
            r = _get_projection_data()
            if ("data" in r) and (type(r["data"]) is dict):
                st.session_state.projection_data = pd.DataFrame(r["data"],)
                st.session_state.projection_visualization = _plot_visualisation()
        if "projection_visualization" in st.session_state:
            st.plotly_chart(st.session_state.projection_visualization, use_container_width=True)

def description():
    """
    Description page
    """
    st.title("Description")
    st.subheader("Statistics")
    st.write("Description of the current data")
    statistics = _get_statistics()
    st.markdown(statistics, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Display data")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.selectbox(label="Sample", options=["all","tagged","untagged","recent"], index=3, key="data_mode") 
    with col2:
        st.number_input(label="min", key="data_min", min_value=0, value=0, step=1)
    with col3:
        st.number_input(label="min", key="data_max", min_value=0, value=10, step=1)
    if st.button(label="Send changes"):
        st.write("Send changes")
        _send_table()

    st.session_state.data_df = df = _get_table()

    # make the table editable
    labels =  st.session_state.state["schemes"]["available"][st.session_state.current_scheme]
    st.session_state.data_df["labels"] = (
        st.session_state.data_df["labels"].astype("category").cat.remove_categories(
            st.session_state.data_df['labels']).cat.add_categories(labels)
    )
    st.data_editor(st.session_state.data_df[["labels", "text"]], disabled=["text"])

    # https://discuss.streamlit.io/t/editable-dataframe-dropdown/38713/3

def simplemodels():
    st.title("Active learning")
    st.write("Configure active learning model") 

def bertmodels():
    st.title("Global model")
    st.write("Train, test and predict with final model") 

def export():
    st.title("Export")
    st.write("Export your data and models") 


if __name__ == "__main__":
    main()