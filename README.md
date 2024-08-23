# (py) Active Tigger

🚧 Under development 🚧

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/emilienschultz/pyactivetigger/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/downloads/)
[![PyPI Version](https://img.shields.io/pypi/v/activetigger)](https://pypi.org/project/activetigger/)

[Python refactoring of R Shiny ActiveTigger app (Julien Boelaert & Etienne Ollion)](https://gitlab.univ-lille.fr/julien.boelaert/activetigger)

Server/client BERT fine tunning app using active learning

- API in Python
- Frontend in React

## Installation

For the moment, it is better to install `activetigger` via Github (be careful to use Python 3.11):

```bash
git clone https://github.com/emilienschultz/pyactivetigger.git
pip install -r pyactivetigger/requirements.txt
```

## Start the server

Create a config file `config.yaml` in the directory where you want to launch the server (you can use the `config.yaml.sample` as an example):

- `path` : path to store files (for instance `./data`)
- `secret_key` : to secure tokens. To generate it `openssl rand -hex 32C`

Then, to launch the server (on 0.0.0.0 port 8000 by default). You can configurate exposed port if needed.

```python
python -m activetigger
```

Otherwise, you can launch only the API with `uvicorn`:

```
uvicorn activetigger.api:app --host 0.0.0.0 --port 80 --reload
```

## Users management

You can create a `add_users.yaml` file with the users to add when the server starts :

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
