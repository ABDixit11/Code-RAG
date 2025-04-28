# from rank_bm25 import BM25Okapi ##too slow
import bm25s
import Stemmer
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from vector_storage import *
from collections import defaultdict
from ragatouille import RAGPretrainedModel


def map_results_to_indices(corpus, results):
    doc_to_index = {doc: idx for idx, doc in enumerate(corpus)}
    indices = []
    for i in range(len(results)):
        doc_text = results[i]   

        original_index = doc_to_index.get(doc_text, -1)

        if original_index != -1:
            indices.append(original_index)
        else:
            print(f"Warning: Could not find original index for retrieved document.")

    return indices


def retrieve_bm25(query, bm25, stemmer, corpus, top_k=5):
    query_tokens = bm25s.tokenize(query, stemmer=stemmer)
    results, scores = bm25.retrieve(query_tokens, corpus=corpus, k=top_k)
    indices = map_results_to_indices(corpus, results[0])

    return indices, scores[0]


def reciprocal_rank_fusion(ranked_lists, k=60):
    fusion_scores = defaultdict(float)

    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            fusion_scores[doc_id] += 1 / (k + rank)

    sorted_docs = sorted(fusion_scores.items(), key=lambda x: -x[1])
    return [doc_id for doc_id, _ in sorted_docs]


def save_results_to_txt(query, results, file_path="query_results.txt"):
    try:
        with open(file_path, "w") as f:  
            f.write(f"Query: {query}\n")
            f.write("Results:\n")
            
            for i, result in enumerate(results, 1):
                f.write(f" {i}. {result}\n")

        print(f"Results saved to {file_path}.")
    except Exception as e:
        print(f"Error saving results: {e}")


if __name__ == "__main__":
    
    metadata = load_data_serial()

    index = faiss.read_index("/app/faiss_index.bin")
    print("Loaded pre-existing Faiss index.")

    corpus = []
    print("Loading some metadata for bm25.")
    for item in metadata:
        code_snippet = item.get("func_code_string", "")
        repo = item.get("repository_name", "")
        language = item.get("language", "")
        documentation = item.get("func_documentation_string", "")

        extended_corpus = [code_snippet, repo, language, documentation]
        extended_corpus = ' [SEP] '.join(extended_corpus)

        if not extended_corpus.strip():
            print("Warning: Skipping empty or missing keys.")
            continue
        corpus.append(extended_corpus)

    print("Loaded some metadata for bm25.")

    print("Initializing retrieval models.")
    # tokenized_corpus = [doc.split(" ") for doc in corpus]
    # bm25 = BM25Okapi(tokenized_corpus)
    stemmer = Stemmer.Stemmer("english")
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)
    bm25 = bm25s.BM25()
    bm25.index(corpus_tokens)
    print("Initialized bm25.")
    dense_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Initialized sentence transformer.")


    while True:
        user_query = input("\nEnter your query (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            print("Exiting...")
            break

        use_colbert=input("\nEnter 'True' to use colbert reranking otherwise enter 'False': ")

        bm25_indices, bmscore = retrieve_bm25(user_query, bm25, stemmer, corpus, top_k=10)
        bm25_results = bm25_indices
        print(bm25_results)
        # print(bmscore)
        distances, dense_indices = query_faiss_index(user_query, index, dense_model, top_k=10)
        dense_results = dense_indices[0].tolist()
        print(dense_results)

        # Combine rankings using RRF
        ranked_lists = [bm25_results, dense_results]
        fused_results = reciprocal_rank_fusion(ranked_lists)
        print("RRF results: ",fused_results)

        candidate_docs = [corpus[i] for i in fused_results]
        # print(candidate_docs)

        # Re-rank with ColBERT 
        if use_colbert=="True":
            RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
            colbert_results = RAG.rerank(query=user_query, documents=candidate_docs, k=5)

            content = [res['content'] for res in colbert_results]
            indices = map_results_to_indices(corpus, content)
        else: 
            indices = fused_results[:5]           

        # print(content)

        results = map_results_to_metadata([indices], metadata)

        save_results_to_txt(user_query, results, "query_responses_top5.txt") # does not append

        print("\nRetrieved Snippets:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result}")
