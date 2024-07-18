#!/bin/bash

# Function to compare versions
version_compare() {
    local v1=$1
    local v2=$2
    if [[ $(printf "%s\n" "$v1" "$v2" | sort -V | head -n 1) != "$v2" ]]; then
        return 1  # v1 < v2
    else
        return 0  # v1 >= v2
    fi
}

# Check if Mojo is available and its version is at least 24.4.0
mojo_version=$(mojo -v | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+')
if [[ ! $mojo_version || $(version_compare "$mojo_version" "24.4.0") -ne 0 ]]; then
    echo "Mojo version 24.4.0 or higher is required but not found."
    echo "Please install Mojo by following the instructions at: https://docs.modular.com/max/install"
    exit 1
fi

# Check if MAX is available and its version is at least 24.4.0
max_version=$(max -v | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+')
if [[ ! $max_version || $(version_compare "$max_version" "24.4.0") -ne 0 ]]; then
    echo "MAX version 24.4.0 or higher is required but not found."
    echo "Please install MAX by following the instructions at: https://docs.modular.com/max/install"
    exit 1
fi

# Check Modular version
modular_version=$(modular -v | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+')
echo "Modular version $modular_version"

# Install Python libraries if not already installed
pip install -r requirements.txt

# Create a temporary main file to run tests
echo "from tests.run_tests import run_tests

def main():
    run_tests()" > temp_main.mojo

# Run the temporary main file
mojo temp_main.mojo

# Remove the temporary main file
rm temp_main.mojo
