# (py) Active Tigger

🚧 Under development 🚧

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/emilienschultz/pyactivetigger/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/downloads/)
[![PyPI Version](https://img.shields.io/pypi/v/activetigger)](https://pypi.org/project/activetigger/)

Server/client BERT fine tunning app using active learning

[Python refactoring of R Shiny ActiveTigger app (Julien Boelaert & Etienne Ollion)]( https://gitlab.univ-lille.fr/julien.boelaert/activetigger)

## Installation

You can install `activetigger` via pip:

```bash
pip install activetigger
```

You need also to download fasttext models https://fasttext.cc/docs/en/crawl-vectors.html and specify the folder in the install file in the `config.yaml`.

## Start the server

Create a config file `config.yaml` in the directory where you want to launch the server :

- `path` : path to store files (for instance `./data`)
- `secret_key` : to secure tokens. To generate it `openssl rand -hex 32C`
- `path_models`:  absolute path to fasttext models
- `users`: list of user:password

Then, to launch the server (on 0.0.0.0 port 8000 by default)

```python
python -m activetigger
```

Otherwise, you can launch with `uvicorn`:

```
uvicorn activetigger.api:app --host 0.0.0.0 --port 80 --reload
```

To use the widget in the `widget` module:

```python
from activetigger.widget import Widget
at = Widget(URL_SERVER="http://0.0.0.0:8000")
```

## Technical specifications

- REST-like client/server architecture
- Mixed data storage : sqlite + files
- Backend Python
    - FastAPI
    - independant Processes to CPU-bound tasts (embeddings/bertmodels)
- Frontend
    - IPython widget
    - To do : Javascript (React ? Vue ?)