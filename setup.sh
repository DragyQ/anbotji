#!/bin/bash

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt file not found in the current directory."
    exit 1
fi

# Install packages using pip3
echo "Installing packages from requirements.txt..."
pip3 install -r requirements.txt

# Check the exit status of pip install
if [ $? -eq 0 ]; then
    echo "Successfully installed all packages from requirements.txt"
else
    echo "Error: Failed to install some or all packages"
    exit 1
fi