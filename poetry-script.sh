#!/bin/bash

# Install Poetry using the official installation script
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"

# Verify Poetry installation
poetry --version

# Install pyenv (if it's not already installed)
if ! command -v pyenv &> /dev/null
then
    echo "pyenv not found. Installing pyenv..."
    curl https://pyenv.run | bash

    # Initialize pyenv by adding configuration to the appropriate files
    echo -e 'export PYENV_ROOT="$HOME/.pyenv"\n' \
            'export PATH="$PYENV_ROOT/bin:$PATH"\n' \
            'eval "$(pyenv init --path)"\n' \
            'eval "$(pyenv init -)"\n' \
            'eval "$(pyenv virtualenv-init -)"\n' >> ~/.bashrc

    # Source the bashrc file to update the shell environment for this script
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init --path)"
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
fi

# Install Python 3.9.20 using pyenv
echo "Installing Python 3.9.20 using pyenv..."
pyenv install 3.9.20
pyenv global 3.9.20

# Verify Python version
python --version

# Use the specified Python version for Poetry's virtual environment
poetry env use python3.9


# Install project dependencies using Poetry
poetry install

# Activate the Poetry virtual environment
poetry shell