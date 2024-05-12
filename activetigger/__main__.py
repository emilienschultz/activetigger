# PyActiveTigger
import uvicorn
import argparse
from pathlib import Path
import subprocess
import importlib.util

def get_streamlit_path():
    """
    Look for path
    """
    if Path("./activetigger/frontend.py").exists():
        return "./activetigger/frontend.py"
    lib = importlib.util.find_spec("activetigger")
    if lib:
        path = lib.origin.replace("/__init__.py","frontend.py")
        if Path(path).exists():
            return path
    return None

if __name__ == "__main__":
    """
    Launch the service
    - launch FASTAPI backend
    - launch Streamlit app
    """
    parser = argparse.ArgumentParser(description="Run pyActiveTigger")
    parser.add_argument('-s', '--server', type=bool, default=False,
                        help='Server only (FASTAPI). Default is False.')
    parser.add_argument('-a', '--adress', type=str, default='0.0.0.0',
                        help='IP address the application will listen on. Default is "0.0.0.0".')
    parser.add_argument('-p', '--portapi', type=int, default=5000,
                        help='Port number for the API. Default is 5000.')
    parser.add_argument('-q', '--portfront', type=int, default=8000,
                        help='Port number for the API. Default is 8000.')
    parser.add_argument('-c', '--config', type=str, default="./config.yaml",
                        help='Config file. Default is local directory.')
    args = parser.parse_args()

    # if no config file, create default one
    if not Path(args.config).exists():
        print("Creating config.yaml file by default")
        content = """path: ./projects
secret_key: f63aeb7426d2c8a3defc02a3e788c2f311482d6cff557c2c5bdebc71d67b507a
path_models: ./models
"""
        with open("./config.yaml","w") as f:
            f.write(content)

    # launch streamlit app
    if not args.server:
        print('Start streamlit app')
        path = get_streamlit_path()
        if path is None:
            print("Error finding the streamlit app")
        else:
            process = subprocess.Popen(["python", "-m" "streamlit", "run", path,
                        "--server.address",args.adress,"--server.port", str(args.portfront)])
        
    # launch fastapi app
    print("Streamlit app launched")
    uvicorn.run("activetigger.api:app", host=args.adress, port=args.portapi, reload=True)