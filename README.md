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

And install the requirements with 


```bash
pip install -r requirements.txt
```

## Use

To launch the server

```python
python -m activetigger
```

To use the widget in the `widget` module, you can do the following:

```python
from activetigger import widget
at = widget.Widget()
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