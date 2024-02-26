# PyActiveTigger
import uvicorn

__version__ = "0.0.1"

if __name__ == "__main__":
    uvicorn.run("api:app", port=8000, reload=True)