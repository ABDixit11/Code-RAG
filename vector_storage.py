import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def load_data_from_batches(batch_folder="/app/batch_json_files"):
    """
    Load data from JSON batch files stored in the specified folder.
    """
    data = []
    if not os.path.exists(batch_folder):
        print(f"Error: Folder '{batch_folder}' does not exist.")
        return data

    batch_files = [os.path.join(batch_folder, f) for f in os.listdir(batch_folder) if f.endswith(".json")]
    print(f"Found {len(batch_files)} batch files in '{batch_folder}'.")

    for batch_file in batch_files:
        try:
            with open(batch_file, "r") as f:
                batch_data = json.load(f)
                data.append(batch_data)  # Append entire batch data
            print(f"Loaded {len(batch_data)} records from '{batch_file}'.")
        except Exception as e:
            print(f"Error reading '{batch_file}': {e}")
    print(f"Total batches loaded: {len(data)}.")
    
    return data

def load_data_serial(file_path="all_rows_serialized.json"):
    try:
        with open(file_path, "r") as f:
            serialized_data = json.load(f)
        data = [json.loads(item) for item in serialized_data]
        print(f"Loaded {len(data)} data points.")
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return []

def generate_embeddings(data, model_name="all-MiniLM-L6-v2", embedding_folder="embeddings_batches"):
    """
    Generate embeddings for each batch and save them to the embedding folder.
    """
    if not os.path.exists(embedding_folder):
        os.makedirs(embedding_folder)
        print(f"Created folder: {embedding_folder}")

    print("Generating embeddings...")

    model = SentenceTransformer(model_name)

    for batch_index, batch in enumerate(data):
        # Extract the "func_code_string" values from each batch (assuming batch is a list of dictionaries)
        code_snippets = [item.get("func_code_string", "") for item in batch]

        # Filter out empty or missing 'func_code_string'
        code_snippets = [snippet for snippet in code_snippets if snippet.strip()]

        if not code_snippets:
            print(f"Warning: No valid 'func_code_string' found in batch {batch_index}. Skipping.")
            continue

        # Generate embeddings for the entire batch
        embeddings = model.encode(code_snippets, convert_to_tensor=True)  # Batch encoding

        # Save the batch embeddings
        embedding_file = os.path.join(embedding_folder, f"embedding_batch_{batch_index}.npy")
        np.save(embedding_file, embeddings)
        print(f"Saved embeddings for batch {batch_index} to '{embedding_file}'.")

    print(f"All embeddings saved in folder: {embedding_folder}")

def combine_embeddings(embedding_folder="embeddings_batches", output_file="combined_embeddings.npy"):
    """
    Combine all individual batch embeddings into a single file, sorted numerically.
    """
    embedding_files = [os.path.join(embedding_folder, f) for f in os.listdir(embedding_folder) if f.endswith(".npy")]
    print(f"Found {len(embedding_files)} embedding files in '{embedding_folder}'.")

    # Sort the files numerically based on the batch index (e.g., embedding_batch_0.npy, embedding_batch_1.npy)
    embedding_files.sort(key=lambda f: int(re.search(r'(\d+)', f).group()))  # Extract numeric part of the filename

    embeddings = []
    for embedding_file in embedding_files:
        embeddings.append(np.load(embedding_file))

    embeddings = np.vstack(embeddings)
    np.save(output_file, embeddings)
    print(f"Combined embeddings saved to '{output_file}'.")
    return embeddings


def create_faiss_index(embeddings, use_ivf=False, nlist=100):
    print("Creating Faiss index...")
    dimension = embeddings.shape[1]  

    if use_ivf:
        quantizer = faiss.IndexFlatL2(dimension) 
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        index.train(embeddings)
    else:
        index = faiss.IndexFlatL2(dimension)  

    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    print(f"Number of vectors in the index: {index.ntotal}")

    faiss.write_index(index, "faiss_index.bin")
    print("Faiss index saved.")
    return index


def query_faiss_index(query, index, model, top_k=5):
    print(f"Processing query: {query}")
    query_vector = model.encode(query).astype("float32")
    faiss.normalize_L2(query_vector.reshape(1, -1))  
    distances, indices = index.search(np.array([query_vector]), k=top_k)

    print(f"Top {top_k} matches retrieved.")
    return distances, indices

def map_results_to_metadata(indices, metadata):
    results = []
    for idx in indices[0]:  
        if idx < len(metadata):  
            results.append(metadata[idx].get("func_code_string", ""))
        else:
            print(f"Warning: Index {idx} out of metadata range.")
    return results


def main():
    try:
        # Load serialized data or create it if it doesn't exist
        if not os.path.exists("all_rows_serialized.json"):
            try:
                data = load_data_from_batches()
                with open("all_rows_serialized.json", "w") as f:
                    json.dump(data, f)
                print("All rows have been serialized and saved to 'all_rows_serialized.json'.")
            except Exception as e:
                print(f"Error serializing and saving rows: {e}")
                return
        else:
            try:
                data = load_data_serial()
                print("Data loaded from 'all_rows_serialized.json'.")
            except Exception as e:
                print(f"Error loading data from 'all_rows_serialized.json': {e}")
                return

        # Generate embeddings if not already present
        if not os.path.exists("combined_embeddings.npy"):
            try:
                generate_embeddings(data)
                embeddings = combine_embeddings()
                print("Embeddings generated and combined.")
            except Exception as e:
                print(f"Error generating or combining embeddings: {e}")
                return
        else:
            try:
                embeddings = np.load("combined_embeddings.npy")
                print("Loaded pre-existing combined embeddings.")
            except Exception as e:
                print(f"Error loading pre-existing embeddings: {e}")
                return

        # Create or load FAISS index
        try:
            index = faiss.read_index("faiss_index.bin")
            print("Loaded pre-existing Faiss index.")
        except Exception as e:
            print(f"Error loading Faiss index: {e}. Creating a new one.")
            try:
                index = create_faiss_index(embeddings)
                print("New Faiss index created and saved.")
            except Exception as faiss_error:
                print(f"Error creating Faiss index: {faiss_error}")
                return

        print("Setup complete.")

    except Exception as main_error:
        print(f"An unexpected error occurred in the setup process: {main_error}")

if __name__ == "__main__":
    main()