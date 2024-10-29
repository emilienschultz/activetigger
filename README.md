# Active Tigger

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/emilienschultz/pyactivetigger/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/downloads/)
[![PyPI Version](https://img.shields.io/pypi/v/activetigger)](https://pypi.org/project/activetigger/)
![React](https://img.shields.io/badge/React-18.0.0-blue)

ActiveTigger[^1] is an text annotation web tool dedicated for computational social sciences. It is designed to assist exploration and model (BERT) fine-tuning to annotate text dataset relying on active learning.

> [!IMPORTANT] 
> The app is currently 🚧 **under the beta test phase** 🚧.
>
> [**how to beta test Active Tigger**](https://github.com/emilienschultz/activetigger/blob/main/LICENSE)


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
cd activetigger/api
python -m activetigger
```

You can also install the last stable version of the API from PyPi with

```bash
pip install activetigger
```

### Install the React frontend

The frontend is written in React/Typescript. To run the dev version and to build the app, you need first to install node.js and npm (version > 20).

```bash
sudo apt-get install nodejs npm
```

Then you can install the npm packages

```bash
cd frontend
npm i
```

You can then run the dev version

```bash
npm run dev
```

To compile

```bash
npm run compile
```

To build

```bash
npm run build
```

You can deploy the app with Github Pages for tests

## Documentation

🚧 to write 🚧

## How to contribute

🚧 to write 🚧

## How to cite

🚧 to write 🚧


[^1]: The current version is a refactor of [R Shiny ActiveTigger app (Julien Boelaert & Etienne Ollion)](https://gitlab.univ-lille.fr/julien.boelaert/activetigger)
