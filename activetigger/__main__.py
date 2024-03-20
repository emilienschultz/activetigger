# PyActiveTigger
import uvicorn

if __name__ == "__main__":
    uvicorn.run("api:app", port=8000, reload=True)
