# How to deploy ActiveTigger on a production linux based server?

The recommended method to deploy ActiveTigger on a linux based server is to use docker.
If you wish to deploy without docker follow the docker configuration files as a guid to install and configure outside docker.

No test were made in a non-linux host server. Docker should work but the sections of this guide for GPU/HTTPS parts require a linux host.

## Requirements

To safely run activetigger we recommend this server configuration:

- ??Go RAM by concurrent user
- ??Go disk space by stored dataset
- ?? processor by concurrent user
- ??Go GPU RAM by concurrent user

## Architecture

Activetigger is fueled by three services:

- a PostGreSQL database (using SQLite is possible but not recommended in production)
- a python API process
- a reverse proxy serving the built JavaScript client code and routing the API calls

## Prepare your server

1- docker

First you need to install docker on your host machine.
Please follow the [docker documentation](https://docs.docker.com/engine/install/).

Do not forget the post-install requirements. In particular to enable docker services

```bash
sudo systemctl enable docker.service
sudo systemctl enable containerd.service
```

2- user system

It's a good practice to create a system user for the application to make sure maintenance can be achieved by multiple physical person using there own credentials to access the server.

```bash
sudo adduser activetigger
# add user to the docker group
sudo adduser activetigger docker
```

You might want to deny SSH access by this generic user:

```bash
sudo vi /etc/ssh/ssh_config
...
DenyUsers activetigger
...
```

Restart SSH

```bash
sudo systemctl restart ssh
```

3- create the app directory

Depending on your server configuration carefully chose where to store the app directory.
This is where all the data will be stored so make sure to have enough disk space.

```bash
sudo mkdir /opt/activetigger
sudo chown activetigger:activetigger /opt/activetigger
```

4- clone source code

Use the generic system user to clone the source code in the app directory.
If your system does not already have git installed install it first.
Make sure to use the production branch.

```bash
sudo su activetigger
cd /opt/activetigger
git clone https://github.com/emilienschultz/activetigger.git .
git checkout production
```

## NVIDIA GPU

### NVIDIA driver

You first have to install the Nvidia card driver available here: https://www.nvidia.com/en-us/drivers/

### NVIDIA Container Toolkit

Make sure to install NVIDIA Container Toolkit by following [official installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

As an example in an ubuntu here is the process:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
  sudo apt-get install -y \
      nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Test your config

```bash
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
Tue Jul  1 10:55:16 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.169                Driver Version: 570.169        CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Quadro RTX 5000                Off |   00000000:00:06.0 Off |                    0 |
|  0%   39C    P2             25W /  230W |       0MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

## configure docker

## HTTPS
