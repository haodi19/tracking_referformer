#!/bin/bash
# unzip is required

OUTPUT_PATH="LaSOTTest"
ZIP_PATH="zip"

mkdir $OUTPUT_PATH

for ZIP_FILE in $ZIP_PATH/"*.zip"; do
    echo $ZIP_FILE
    unzip "$ZIP_FILE" -d "$PWD/$OUTPUT_PATH"
done