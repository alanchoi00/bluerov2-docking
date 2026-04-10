FROM ghcr.io/alanchoi00/blue-sim-humble:humble-desktop-nvidia

ENV USERNAME=blue
ENV USER_WORKSPACE=/home/$USERNAME/ws
WORKDIR $USER_WORKSPACE

# Install debugging/linting tools
RUN python3 -m pip install \
    pre-commit \
    mypy

ENV PYTHONWARNINGS="ignore"
