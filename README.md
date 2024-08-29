# Active Tigger

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/emilienschultz/pyactivetigger/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/downloads/)
[![PyPI Version](https://img.shields.io/pypi/v/activetigger)](https://pypi.org/project/activetigger/)
![React](https://img.shields.io/badge/React-18.0.0-blue)

ActiveTigger is an active learning online app for text classification in computational social sciences. It is designed to assist exploration and fine-tuning of BERT models.

The app is currently 🚧 Under development 🚧.

[Python/React refactoring of R Shiny ActiveTigger app (Julien Boelaert & Etienne Ollion)](https://gitlab.univ-lille.fr/julien.boelaert/activetigger)


## Installation

The application has a client/server architecture :

- the server is in Python using FastAPI
- the client is in [React](https://reactjs.org/)

## Start the server

Clone the repository

```bash
git clone https://github.com/emilienschultz/activetigger.git
```

Create a virtual environnement with Python 3.11 and install requirements

```bash
pip install -r activetigger/api/requirements.txt
```

The server configuration is achieved with the `config.yaml` file the api directory (you can use the `config.yaml.sample` as an example):

- `path` : path to store files (for instance `./data`)
- `path_models` : path to store the models (for instance `./data/models`)

Then, to launch the server (on 0.0.0.0 port 5000 by default, you can configurate exposed port if needed).

```bash
cd api
python -m activetigger
```

Otherwise, you can launch only the API with `uvicorn`:

```
uvicorn activetigger.api:app --host 0.0.0.0 --port 80 --reload
```

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
