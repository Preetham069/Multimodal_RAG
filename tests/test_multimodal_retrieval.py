import os

from dotenv import load_dotenv
from google import genai

import chromadb


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found in the .env file."
    )


# ============================================================
# 2. CREATE GEMINI CLIENT
# ============================================================

client = genai.Client(api_key=api_key)

EMBEDDING_MODEL = "gemini-embedding-2"


# ============================================================
# 3. CONNECT TO CHROMADB
# ============================================================

chroma_client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = chroma_client.get_collection(
    name="multimodal_rag"
)


# ============================================================
# 4. TEXT QUERY ABOUT THE IMAGE
# ============================================================

query = "An image showing Lord Ganesha sitting on an ornate throne"

print("\n========================================")
print("MULTIMODAL RETRIEVAL TEST")
print("========================================")

print("\nUser Query:")
print(query)


# ============================================================
# 5. CREATE QUERY EMBEDDING
# ============================================================

query_result = client.models.embed_content(
    model=EMBEDDING_MODEL,
    contents=query
)

query_embedding = query_result.embeddings[0].values

print("\nQuery embedding created!")
print("Embedding dimensions:", len(query_embedding))


# ============================================================
# 6. SEARCH ONLY IMAGE RECORDS
# ============================================================

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=1,
    where={
        "type": "image"
    }
)


# ============================================================
# 7. DISPLAY RESULT
# ============================================================

print("\n========================================")
print("RETRIEVED IMAGE")
print("========================================")

if len(results["ids"][0]) > 0:

    print("\nID:")
    print(results["ids"][0][0])

    print("\nMetadata:")
    print(results["metadatas"][0][0])

    print("\nDistance:")
    print(results["distances"][0][0])

    print("\nDocument:")
    print(results["documents"][0][0])

    print("\nIMAGE RETRIEVAL SUCCESSFUL!")

else:

    print("\nNo image was retrieved.")