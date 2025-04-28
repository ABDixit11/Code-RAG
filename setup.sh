#!/bin/bash

# Step 1: Run the Python script get_data.py
echo "Running get_data.py..."
python3 get_data.py

# Check if get_data.py was successful
if [ $? -ne 0 ]; then
  echo "Error executing get_data.py. Exiting..."
  exit 1
fi

# Step 2: Run the compiled C++ executable load_data
echo "Running ./load_data..."
./load_data

# Check if ./load_data was successful
if [ $? -ne 0 ]; then
  echo "Error executing ./load_data. Exiting..."
  exit 1
fi

# Step 3: Run the compiled C++ executable batch_fetch
echo "Running ./batch_fetch..."
./batch_fetch

# Step 4: Run the Python script vector_storage.py
echo "Running vector_storage.py..."
python3 vector_storage.py

# Check if vector_storage.py was successful
if [ $? -ne 0 ]; then
  echo "Error executing vector_storage.py. Exiting..."
  exit 1
fi

# Step 5: Delete the folders batch_json_files and embeddings_batches if they exist
echo "Checking and removing folders..."
if [ -d "batch_json_files" ]; then
  rm -rf batch_json_files
  echo "Deleted batch_json_files folder."
fi

if [ -d "embeddings_batches" ]; then
  rm -rf embeddings_batches
  echo "Deleted embeddings_batches folder."
fi

# Reminder to run coderag.py
echo "Please run the Python script coderag.py like this:"
echo "python3 coderag.py"
