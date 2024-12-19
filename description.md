# Architecture

This is a collection of technical points/choices for the app

Overall architecture :

- **backend** : Python/FastAPI
- **frontend** : React/Typescript

## Backend

- `config.yaml` define the parameters at the server launch
- The unit is the project, composes of different classes
    - Features
    - Schemes
    - Simplemodels
    - Bertmodels
    - Users
- CPU/GPU bound computation is managed in separated processes with a queue
- State of the service is checked at each request (with a threshold)

### Data management

- Tabular data is stored as separated parquet files divided in train / test / complete
- SQLite database to manage annotations/parameters/users/logs
- Projects are loaded into memory to facilitate computation (filter, etc.)
    - Unloaded after one day
- Bert models are saved in dedicated filesystems

### Processes

- ProcessPoolExecutor with workers
    - https://superfastpython.com/processpoolexecutor-in-python/
- Different type of parallel process : training ; predicting
- Only one process possible by user/project

### Users role

- Role-Based Access Control (RBAC) - 3 roles : root, manager, annotator
- Authentification with OAuth2 and token in header
    - Table of valid tokens
- A table of authorization defines the relation users/projects
- Different uses can modify a same project : no lock

## Select element to annotate

The selection combines different strategy : filters and/or active learning.

Active learning is a prediction with a model trained on already annotated data.

- Different modes of selection
    - deterministic
    - aleatory
    - maxprob for a label
    - max entropy

- Pipeline of choice
    - sample (tagged, untagged, all)
    - regex
    - proba / entropie


## Frontend 

### State management

- Each project is described by its general state (not user specific)
    - Computed/computing elements