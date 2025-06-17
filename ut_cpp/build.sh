# Make sure to run this from the project directory?
#!/bin/bash

# Set script to fail on first error
set -e

# Define variables
PROJECT_DIR="ut_cpp"
BUILD_DIR="$PROJECT_DIR/build"
MODULE_NAME="pl_qcqp"

echo "[*] Cleaning previous build..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "[*] Running CMake..."
cmake ..

echo "[*] Building module..."
make -j4

# Find the output .so file
SO_FILE=$(find . -name "${MODULE_NAME}*.so" | head -n 1)

if [ -z "$SO_FILE" ]; then
    echo "[!] Could not find the compiled .so file."
    exit 1
fi

# Rename the .so file to pl_qcqp.so in the project root
TARGET_SO="../${MODULE_NAME}.so"
cp "$SO_FILE" "$TARGET_SO"

echo "[✓] Module built and copied as $TARGET_SO"
