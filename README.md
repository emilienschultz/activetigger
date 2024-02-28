# (py) Active Tigger

Python refactoring of ActiveTigger (Julien Boelaert & Etienne Ollion) https://gitlab.univ-lille.fr/julien.boelaert/activetigger

Work in progress ...

# Current roadmap

- To add : history of actions
- To add : multiuser
- To add : queue management (rq)
- To add : test units

# Technical specifications

- REST-like client/server architecture
- Mixed data storage : sqlite + files
- Backend Python
    - FastAPI
    - independant Processes to CPU-bound tasts (embeddings/bertmodels)
- Frontend
    - IPython widget
    - To do : Javascript (React ? Vue ?)
