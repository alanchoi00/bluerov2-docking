FROM ghcr.io/alanchoi00/blue-sim:jazzy-desktop-nvidia

ENV USER_WORKSPACE=/home/ubuntu/ws_docking
WORKDIR $USER_WORKSPACE

# Install debugging/linting tools
RUN /home/ubuntu/.venv/blue/bin/pip install \
    pre-commit \
    mypy

USER root
RUN apt-get update && apt-get install -y \
    ros-jazzy-aruco-ros \
    && rm -rf /var/lib/apt/lists/*
USER ubuntu

ENV PYTHONWARNINGS="ignore"
