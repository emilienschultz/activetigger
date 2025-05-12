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
        default=".env",
        help="Config file. Default is local directory.",
    )

    args = parser.parse_args()

    # create environement variables using config file
    if "DATA_PATH" not in os.environ:
        os.environ["DATA_PATH"] = "."
    os.environ["ACTIVETIGGER_USER_HDD_MAX"] = "30"
    if Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
        if "DATA_PATH" in config and Path(config["DATA_PATH"]).exists():
            os.environ["DATA_PATH"] = config["DATA_PATH"]

    uvicorn.run(
        "activetigger.app.main:app",
        host="0.0.0.0",
        port=args.portapi,
        reload=os.environ["MODE"] == "dev",
    )
