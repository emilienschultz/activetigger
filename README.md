# (py) Active Tigger

Python refactoring of ActiveTigger (Julien Boelaert & Etienne Ollion) https://gitlab.univ-lille.fr/julien.boelaert/activetigger

Work in progress ...

# Current roadmap

- fastapi structure
- database/storage
- finish bertmodel implementation
- queue management (rq)
- multiuser
- test units
- clean ipywidget

# Technical specifications

- REST-like client/server architecture
    - JSON for data interchange format
    - State-ful serversite
- mixed data storage : sqlite + files
- backend Python
    - FastAPI
    - Job management with RQ
- frontend
    - Jupyter widget (simplified)
    - Javascript (React ? Vue ?)
