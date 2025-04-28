from datasets import load_dataset
import json

# Load the dataset
ds = load_dataset("code_search_net", "python", trust_remote_code=True)

# Get the 'train' split
train_data = ds['train']

# Initialize a list to store the serialized rows
all_data = []

# Iterate over all rows in the dataset and serialize each row
for row in train_data:
    # Serialize the row to a JSON string
    json_data = json.dumps(row)
    
    # Add the serialized data to the list
    all_data.append(json_data)

# Save all serialized rows to a JSON file
with open('all_rows_serialized.json', 'w') as f:
    json.dump(all_data, f)

print("All rows have been serialized and saved to 'all_rows_serialized.json'.")
