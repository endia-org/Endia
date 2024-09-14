#!/bin/bash

# ANSI color codes
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e ""

# Function to display animated spinner
spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='|/-\'
    echo -n "🔥 Installing Endia 🔥 "
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf "%c" "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        echo -ne "\b"
        sleep $delay
    done
    echo -ne "\b "
}

# Save the original directory
ORIGINAL_DIR=$(pwd)

# Create a temporary directory
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

# Clone and install in the background
(
    git clone https://github.com/endia-org/Endia.git > /dev/null 2>&1
    cd Endia
    git checkout nightly > /dev/null 2>&1
    cd ..
    mojo package ./Endia/endia -o "$ORIGINAL_DIR/endia.📦" > /dev/null 2>&1
    # Move back to the original directory before removing the temp directory
    cd "$ORIGINAL_DIR"
    rm -rf "$TEMP_DIR"
) &

# Display spinner while the installation is in progress
spinner $!

# Wait for the background process to complete before printing the success message
wait

# Print success message in bold
echo -e "\n\n\033[1m🎉 Installation successful!\033[0m"

echo -e "\n🗂️  The Endia package has been placed in $ORIGINAL_DIR"

# Make "nightly build" bold within the message
echo -e "\n${YELLOW}⚠️${NC}  \033[1mNote:\033[0m This version requires the \033[1mMojo nightly build\033[0m.${NC}\n"