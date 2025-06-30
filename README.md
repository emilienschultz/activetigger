# Active Tigger

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/emilienschultz/pyactivetigger/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/downloads/)
[![PyPI Version](https://img.shields.io/pypi/v/activetigger)](https://pypi.org/project/activetigger/)
![React](https://img.shields.io/badge/React-18.0.0-blue)

Hi❗

ActiveTigger[^1] is a text annotation web tool dedicated to computational social sciences. It is designed to assist exploration and model (BERT) fine-tuning to annotate text dataset relying on active learning.

Designed primarily by researchers in social sciences, its use can extend to all users that need to annotate textual data.

> [!IMPORTANT]
> The app is currently 🚧 **under the beta test phase** 🚧. The stable version is planned for mid-June.

[Presentation of ActiveTigger 🇫🇷](https://emilienschultz.github.io/activetigger-presentation/#/title-slide)

Stable version mid-June 2025 (hopefully) 🚧



## Run the app with Docker (recommended)

If you don't already have it installed, install docker first.

Clone the repository

```bash
git clone https://github.com/emilienschultz/activetigger.git
```

Then, you can run the app with docker compose. You need to be in the `docker` directory.

```bash
cd docker
docker compose -p activetigger up
```

The configuration file is `./docker/.env`.

Docker will start:

- Postgresql
- Python API (and install all requirements)
- frontend
- a reverse proxy with Nginx

*By default the docker stack is in mode DEV.* This means that both API and FRONTEND are started using the local code in watch mode. You can update the code the service will be restarted. You can therefor use docker for development.

## Install the app

The app is built on a client/API/Database architecture :

- the server runs an API with FastAPI
- the client is in [React](https://reactjs.org/)
- The database can be either SQLite or Postgresql (if you use Postgresql, you need to install it first)

First, clone the repository

```bash
git clone https://github.com/emilienschultz/activetigger.git
```

### Install the Python API

Create a virtual environnement with Python 3.11, for instance with Anaconda

```bash
conda create -n activetigger python=3.11
conda activate activetigger
```

Install requirements

```bash
pip install -r activetigger/api/requirements.txt
```

Add a specific `config.yaml` file in the api directory if you want to specify the path of the static files and database (you can modify and rename the `config.yaml.sample` or use the default config):

- `path` : path to store files (for instance `./data`)
- `path_models` : path to store the models (for instance `./data/models`)

Launch the server (on 0.0.0.0 port 5000 by default, you can configure exposed port if needed with -p PORTNUM).

```bash
cd activetigger/api
python -m activetigger
```

You can also install the last stable version of the API from PyPi with

```bash
pip install activetigger
```

#### Optional: Install GPU Support for UMAP

To enable GPU support, [install Rapids Cuml](https://docs.rapids.ai/install/#system-req). For instance, for CUDA 12

```bash
pip install --extra-index-url=https://pypi.nvidia.com "cuml-cu12==24.10.*"
```

#### Troubleshooting

On Ubuntu, you need to install some drivers for the postgresql database and the python package psycopg2.

```bash
sudo apt-get install libpq-dev
```

#### Temporary: some models (ModernBERT) need unpublished versions of Huggingface Transformers

The requirements contain the clone of the repository.

### Install the React frontend

The frontend is written in React/Typescript. To run the dev version and to build the app, you need first to install node.js and npm (version > 20).

For linux :

```bash
sudo apt-get install nodejs npm
```

For mac, you can install brew https://brew.sh/ and

```bash
brew install node
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

If you run the backend on a different port, mind the fact that you need to change the address in the `frontend/.env` file to set the correct port.

To compile

```bash
npm run compile
```

To build

```bash
npm run build
```

You can deploy the app with Github Pages for tests

### Install Postgresql

If you want to use Postgresql, you need to install it first. You can use the docker image or install it locally.

**to write**

## Python client

A python client is under development to interact with the API https://github.com/emilienschultz/activetigger-client

## Documentation

The documentation is [here](https://emilienschultz.github.io/activetigger/docs/)

## Contributing

Something in the code looks ugly ? Do you miss a feature? Please add your knowledge to the common effort.

1. Create a fork
2. Create your feature branch: `git checkout -b my-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request 🚀

## Funding

The development of Active Tigger is supported by : [DRARI Île-de-France](https://www.enseignementsup-recherche.gouv.fr/fr/drari-ile-de-france) [ECODEC](https://labex-ecodec.ensae.fr/) [Progedo](https://www.progedo.fr/)

## How to cite

🚧 please do not cite for the moment 🚧

[^1]: The current version is a refactor of [R Shiny ActiveTigger app (Julien Boelaert & Etienne Ollion)](https://gitlab.univ-lille.fr/julien.boelaert/activetigger). Active Tigger name is a pun that draws on the similarity between the words 'Tagger' and 'Tigger.'
