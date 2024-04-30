# PyActiveTigger
import uvicorn
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the PyActiveTigger API server.")
    parser.add_argument('-a', '--adress', type=str, default='0.0.0.0',
                        help='IP address the application will listen on. Default is "0.0.0.0".')
    parser.add_argument('-p', '--port', type=int, default=8000,
                        help='Port number the application will listen on. Default is 8000.')
    parser.add_argument('-c', '--config', type=str, default="./config.yaml",
                        help='Config file. Default is local directory.')
    args = parser.parse_args()

    # if no config file, create default one
    if not Path(args.config).exists():
        print("Creating config.yaml file by default")
        content = """path: ./projects
secret_key: f63aeb7426d2c8a3defc02a3e788c2f311482d6cff557c2c5bdebc71d67b507a
path_fasttext:
users:
"""
        with open("./config.yaml","w") as f:
            f.write(content)

    uvicorn.run("activetigger.api:app", host=args.adress, port=args.port, reload=True)