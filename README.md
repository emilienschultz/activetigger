# Active Tigger

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/emilienschultz/pyactivetigger/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/downloads/)
[![PyPI Version](https://img.shields.io/pypi/v/activetigger)](https://pypi.org/project/activetigger/)
![React](https://img.shields.io/badge/React-18.0.0-blue)

ActiveTigger is an text annotation web tool dedicated for computational social sciences. It is designed to assist exploration and model (BERT) fine-tuning to annotate text dataset relying on active learning.

The app is currently 🚧 under development 🚧.

The current version is a refactor of [R Shiny ActiveTigger app (Julien Boelaert & Etienne Ollion)](https://gitlab.univ-lille.fr/julien.boelaert/activetigger)

## Install the app

The app is built on a client/API architecture :

- the server runs an API with FastAPI
- the client is in [React](https://reactjs.org/)

### Install the Python API

Clone the repository

```bash
git clone https://github.com/emilienschultz/activetigger.git
```

Create a virtual environnement with Python 3.11 and install requirements

```bash
pip install -r activetigger/api/requirements.txt
```

Add a specific `config.yaml` file in the api directory if you want to specify the path of the static files and database (you can modify and rename the `config.yaml.sample` or use the default config):

- `path` : path to store files (for instance `./data`)
- `path_models` : path to store the models (for instance `./data/models`)

Launch the server (on 0.0.0.0 port 5000 by default, you can configurate exposed port if needed with -p PORTNUM).

```bash
cd api
python -m activetigger
```

### Install the React frontend

## Users management

You can create a `add_users.yaml` file with the users to add when the server starts (and then delete the file) :

```
user1: password1
user2: password2
```

## Technical specifications

- REST-like client/server architecture
  - Moslty stateless
  - Client-side parameters
- Mixed data storage : sqlite + files
- Backend Python
  - FastAPI
  - Multiprocessing for CPU-bound tasts
- Frontend
  - React/Typescript
- Role-Based Access Control (manager/annotator)
