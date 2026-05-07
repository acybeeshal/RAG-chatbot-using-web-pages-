import os
import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


# URL LIST 
URLS = [
    "https://upayogi.com/",
    "https://en.wikipedia.org/wiki/Machine_learning"
]

# FETCH WEB PAGE TEXT
def fetch_text(url):
    print(f"Fetching: {url}")

    headers = {"User-Agent": "Mozilla/5.0 (RAG Assignment)"}
    response = requests.get(url, headers=headers, timeout=10)

    soup = BeautifulSoup(response.text, "html.parser")

    # remove noise
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    return " ".join(text.split())


# CHUNKING (SLIDING WINDOW)
def chunk_text(text, chunk_size=400, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap

    return chunks


# BUILD KNOWLEDGE BASE
def build_knowledge_base():
    all_chunks = []
    metadata = []

    for url in URLS:
        text = fetch_text(url)
        chunks = chunk_text(text)[:120]  # limit for stability

        all_chunks.extend(chunks)
        metadata.extend([url] * len(chunks))

    print(f"\nTotal chunks: {len(all_chunks)}")
    print("Creating embeddings...")

    # SAFE BATCH EMBEDDING
    embeddings = model.encode(
        all_chunks,
        batch_size=8,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype("float32")

    # normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # FAISS INDEX
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index, embeddings, all_chunks, metadata


# RETRIEVAL FUNCTION
def retrieve(query, index, chunks, metadata, k=3):
    query_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, k)

    results = []
    for i in indices[0]:
        results.append((chunks[i], metadata[i]))

    return results


# ANSWER GENERATION (EVIDENCE-BASED)
def generate_answer(query, results):
    context = "\n\n".join([c for c, _ in results])

    answer = f"""
==============================
QUESTION: {query}
==============================

ANSWER (based on retrieved web content):

{context[:1200]}

==============================
"""
    return answer

# MAIN PROGRAM

def main():
    print(" Building RAG Knowledge Base...\n")

    index, embeddings, chunks, metadata = build_knowledge_base()

    print("\n Knowledge Base Ready!\n")

    while True:
        query = input("Ask a question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        results = retrieve(query, index, chunks, metadata)

        # SHOW EVIDENCE

        print("\n Retrieved Evidence:\n")

        for i, (chunk, url) in enumerate(results):
            print(f"[{i+1}] Source: {url}")
            print(chunk[:300], "\n")

        # ANSWER
        print("\n Final Answer:\n")
        print(generate_answer(query, results))
        print("=" * 60)




if __name__ == "__main__":
    main()