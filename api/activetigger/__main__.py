import argparse
import os
from pathlib import Path

import uvicorn
import yaml  # type: ignore[import]

if __name__ == "__main__":
    """
    Launch activetigger API
    """

    parser = argparse.ArgumentParser(description="Run pyActiveTigger")

    parser.add_argument(
        "-p",
        "--portapi",
        type=int,
        default=5000,
        help="Port number for the API. Default is 5000.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./config.yaml",
        help="Config file. Default is local directory.",
    )

    args = parser.parse_args()

    # create environement variables using config file
    os.environ["ACTIVETIGGER_PATH"] = "./projects"
    os.environ["ACTIVETIGGER_MODEL"] = "./models"
    os.environ["ACTIVETIGGER_USER_HDD_MAX"] = "30"
    if Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
        if "path" in config:
            os.environ["ACTIVETIGGER_PATH"] = config["path"]
        if "path_models" in config:
            os.environ["ACTIVETIGGER_MODEL"] = config["path_models"]
    else:
        print("Creating config.yaml file by default")
        content = """path: ./projects
        path_models: ./models
        """
        with open("./config.yaml", "w") as f:
            f.write(content)

    print(os.environ["ACTIVETIGGER_PATH"], os.environ["ACTIVETIGGER_MODEL"])

    uvicorn.run(
        "activetigger.app.main:app", host="0.0.0.0", port=args.portapi, reload=True
    )
