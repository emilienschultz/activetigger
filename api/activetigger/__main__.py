import argparse
from pathlib import Path

import uvicorn

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

    # if no config file, create default one
    if not Path(args.config).exists():
        print("Creating config.yaml file by default")
        content = """path: ./projects
path_models: ./models
"""
        with open("./config.yaml", "w") as f:
            f.write(content)

    uvicorn.run(
        "activetigger.app.main:app", host="0.0.0.0", port=args.portapi, reload=True
    )
