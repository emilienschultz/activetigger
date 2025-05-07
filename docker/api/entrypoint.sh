#!/bin/sh
set -e

if [ "$MODE" = "dev" ]; then
  echo "/!\\ Mode is set to DEV /!\\"
else
  echo "/!\\ Mode is set to PRODUCTION /!\\"
fi
echo "(i) Python version is $(python3 --version)"

echo
echo " ~"
echo " ~ Install dependencies"
echo " ~"
echo
# install python deps
mkdir -p /home/python/venv
python -m venv /home/python/venv
PATH="/home/python/venv/bin:$PATH"
cd /api
pip3 install -r requirements.txt

# Check for a config.yaml file in the api directory
if [ ! -f activetigger/api/config.yaml ]; then
    if [ -f activetigger/api/config.yaml.sample ]; then
        echo "No config.yaml found. Copying config.yaml.sample to config.yaml..."
        cp activetigger/api/config.yaml.sample activetigger/api/config.yaml
        echo "You can now edit activetigger/api/config.yaml to modify paths for static files and database."
    else
        echo "No config.yaml.sample found in activetigger/api. Please ensure your configuration is set as needed."
    fi
fi

# Launch the server
echo "Launching API on port $API_PORT..."

if [ "$MODE" = "dev" ]; then
  echo
  echo " ~"
  echo " ~ Start api dev"
  echo " ~"
  echo
  python3 -m activetigger -p "$API_PORT"
else
  echo
  echo " ~"
  echo " ~ Start api production"
  echo " ~"
  echo
  echo "TODO"
  python3 -m activetigger -p "$API_PORT"
fi
