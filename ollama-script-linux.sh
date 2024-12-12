#!/bin/bash

# Install Ollama using the one-liner installation script
curl -fsSL https://ollama.com/install.sh | sh

# Verify Ollama is installed
ollama --version

# Start Ollama serve in a separate background process
ollama serve &

# You can optionally add some delay to ensure the serve command runs properly
sleep 5

# Start Ollama with your desired model (replace 'model_name')
ollama run llama3.2
