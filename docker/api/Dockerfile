ARG PYTHON_VERSION="3.11"
FROM python:${PYTHON_VERSION}

ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd python --gid $GROUP_ID
RUN useradd -ms /bin/bash python --uid $USER_ID --gid $GROUP_ID

USER python


