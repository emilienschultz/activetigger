# (py) Active Tigger

🚧 Under development 🚧

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/emilienschultz/pyactivetigger/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/downloads/)
[![PyPI Version](https://img.shields.io/pypi/v/activetigger)](https://pypi.org/project/activetigger/)

Server/client BERT fine tunning app using active learning

[Python refactoring of R Shiny ActiveTigger app (Julien Boelaert & Etienne Ollion)]( https://gitlab.univ-lille.fr/julien.boelaert/activetigger)

## Installation

You can install `activetigger` via pip (be careful to use Python 3.11):

```bash
pip install activetigger
```

## Start the server

Create a config file `config.yaml` in the directory where you want to launch the server :

- `path` : path to store files (for instance `./data`)
- `secret_key` : to secure tokens. To generate it `openssl rand -hex 32C`
- `path_models`:  absolute path to fasttext models
- `users`: list of user:password

Then, to launch the server (on 0.0.0.0 port 8000 by default). You can configurate exposed port if needed.

```python
python -m activetigger
```

Otherwise, you can launch only the API with `uvicorn`:

```
uvicorn activetigger.api:app --host 0.0.0.0 --port 80 --reload
```

## Technical specifications

- REST-like client/server architecture
- Mixed data storage : sqlite + files
- Backend Python
    - FastAPI
    - independant Processes to CPU-bound tasts (embeddings/bertmodels)
- Frontend
    - Streamlit (prototyping)