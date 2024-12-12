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
    # Add pyenv to bash profile or zsh profile
    echo -e 'export PATH="$HOME/.pyenv/bin:$PATH"\n' \
            'eval "$(pyenv init --path)"\n' \
            'eval "$(pyenv init -)"\n' >> ~/.bashrc
    # Reload shell configuration
    source ~/.bashrc
fi

# Install Python 3.9.20 using pyenv
echo "Installing Python 3.9.20 using pyenv..."
pyenv install 3.9.20
pyenv global 3.9.20

# Verify Python version
python --version

# Use the specified Python version for Poetry's virtual environment
poetry env use python3.9

# Navigate to the project directory (replace with your project path)
cd /path/to/your/project

# Install project dependencies using Poetry
poetry install

# Activate the Poetry virtual environment
poetry shell