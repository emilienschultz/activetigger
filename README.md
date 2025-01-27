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


## Install the app

The app is built on a client/API architecture :

- the server runs an API with FastAPI
- the client is in [React](https://reactjs.org/)

### Install the Python API

Clone the repository

```bash
git clone https://github.com/emilienschultz/activetigger.git
```

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

To compile

```bash
npm run compile
```

To build

```bash
npm run build
```

You can deploy the app with Github Pages for tests

### Python client

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
