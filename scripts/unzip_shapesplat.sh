#!/bin/bash
# Directory containing the .zip files
SOURCE_DIR=gs_data/shapesplat/
# Directory where all extracted files will be merged
DEST_DIR=gs_data/shapesplat/shapesplat_ply

# check if SOURCE_DIR is exist
if [ ! -d "$SOURCE_DIR" ]; then
    echo "$SOURCE_DIR does not exist."
    exit 1
fi



# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"


for file in "$SOURCE_DIR"/*.zip; 
do
  # Extract the .zip file
  unzip "$file" -d "$DEST_DIR"

done

if [ -d "$DEST_DIR/03001627_0" ]; then
    mkdir -p "$DEST_DIR/03001627"
    mv "$DEST_DIR"/03001627_0/* "$DEST_DIR"
    rm -r "$DEST_DIR"/03001627_0
fi
if [ -d "$DEST_DIR/03001627_1" ]; then
    mkdir -p "$DEST_DIR/03001627"
    mv "$DEST_DIR"/03001627_1/* "$DEST_DIR"
    rm -r "$DEST_DIR"/03001627_1
fi

if [ -d "$DEST_DIR/04379243_0" ]; then
    mkdir -p "$DEST_DIR/04379243"
    mv "$DEST_DIR"/04379243_0/* "$DEST_DIR"
    rm -r "$DEST_DIR"/04379243_0
fi
if [ -d "$DEST_DIR/04379243_1" ]; then
    mkdir -p "$DEST_DIR/04379243"
    mv "$DEST_DIR"/04379243_1/* "$DEST_DIR"
    rm -r "$DEST_DIR"/04379243_1
fi


