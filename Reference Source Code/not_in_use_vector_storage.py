import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import rocksdb

db_path = "testdb"
db = rocksdb.DB(db_path,rocksdb.Options(create_if_missing=False))

def load_data_from_rocksdb(limit=None):
    """Load data from RocksDB. If `limit` is specified, load only that many records."""
    try:
        it = db.iterkeys()
        it.seek_to_first()
        data = []
        for idx, key in enumerate(it):
            if limit and idx >= limit:
                break
            value = db.get(key)
            data.append(json.loads(value.decode()))  # Deserialize the stored JSON
        print(f"Loaded {len(data)} data points from RocksDB.")
        return data
    except Exception as e:
        print(f"Error loading data from RocksDB: {e}")
        return []

        
# def load_data(file_path="all_rows_serialized.json"):
#     try:
#         with open(file_path, "r") as f:
#             serialized_data = json.load(f)
            
#         data = [json.loads(item) for item in serialized_data]
#         print(f"Loaded {len(data)} data points.")
#         return data
#     except FileNotFoundError:
#         print(f"Error: File {file_path} not found.")
#         return []

def generate_embeddings(data, model_name="all-MiniLM-L6-v2"):
    print("Generating embeddings...")
    model = SentenceTransformer(model_name)
    embeddings = []

    for item in data:
        code_snippet = item.get("func_code_string", "")
        if not code_snippet.strip():
            print("Warning: Skipping empty or missing 'func_code_string'.")
            continue
        embeddings.append(model.encode(code_snippet))

    embeddings = np.array(embeddings)
    np.save("embeddings.npy", embeddings)
    with open("metadata.json", "w") as f:
        json.dump(data, f)

    print("Embeddings and metadata saved.")
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


def query_faiss_index(query, index, model, top_k=3):
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


def save_results_to_json(query, results, file_path="query_results.json"):
    output_data = {
        "query": query,
        "results": results,
    }

    try:
        with open(file_path, "a") as f:  
            json.dump(output_data, f)
            f.write("\n")  
        print(f"Results saved to {file_path}.")
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    
    data = load_data_from_rocksdb()
    if not data:
        print("No data loaded from RocksDB. Exiting...")
        exit()
    try:
        embeddings = np.load("embeddings.npy")
        print("Loaded pre-existing embeddings.")
    except FileNotFoundError:
        if data:
            embeddings = generate_embeddings(data)
        else:
            print("Error: No data to generate embeddings.")
            exit()

    try:
        index = faiss.read_index("faiss_index.bin")
        print("Loaded pre-existing Faiss index.")
    except:
        index = create_faiss_index(embeddings)


    try:
        with open("metadata.json", "r") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print("Error: Metadata file not found.")
        exit()

    model = SentenceTransformer("all-MiniLM-L6-v2")

    while True:
        user_query = input("\nEnter your query (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            print("Exiting...")
            break

        distances, indices = query_faiss_index(user_query, index, model)

        results = map_results_to_metadata(indices, metadata)

        save_results_to_json(user_query, results)

        print("\nRetrieved Snippets:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result}")
