#!/bin/bash

# Install Poetry using the official installation script
curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to the PATH
export PATH="$HOME/.local/bin:$PATH"

# Make sure Poetry is installed correctly
poetry --version

# Install the project dependencies defined in pyproject.toml
poetry install

# Activate the virtual environment
poetry shell

# Verify the environment is activated
which python
