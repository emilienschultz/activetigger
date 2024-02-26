# PyActiveTigger
import uvicorn
from fastapi import FastAPI
import server
import widget

__version__ = "0.0.1"

if __name__ == "__main__":
    uvicorn.run("api:app", port=8000, reload=True)