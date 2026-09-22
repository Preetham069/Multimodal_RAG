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
    raise ValueError("GEMINI_API_KEY was not found in the .env file.")


# ============================================================
# 2. CREATE GEMINI CLIENT
# ============================================================

client = genai.Client(api_key=api_key)

EMBEDDING_MODEL = "gemini-embedding-2"


# ============================================================
# 3. CONNECT TO EXISTING CHROMADB
# ============================================================

chroma_client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = chroma_client.get_collection(
    name="multimodal_rag"
)


# ============================================================
# 4. USER QUERY
# ============================================================

query = "What technologies are used in the proposed system?"

print("\nUSER QUERY")
print("----------")
print(query)


# ============================================================
# 5. CREATE EMBEDDING FOR THE QUERY
# ============================================================

query_result = client.models.embed_content(
    model=EMBEDDING_MODEL,
    contents=query
)

query_embedding = query_result.embeddings[0].values


print("\nQuery embedding created!")
print("Embedding dimensions:", len(query_embedding))


# ============================================================
# 6. SEARCH CHROMADB
# ============================================================

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)


# ============================================================
# 7. DISPLAY RETRIEVED RESULTS
# ============================================================

print("\n========================================")
print("RETRIEVED RESULTS")
print("========================================")

for i in range(len(results["documents"][0])):

    print("\n----------------------------------------")
    print(f"RESULT {i + 1}")
    print("----------------------------------------")

    print("\nDocument:")
    print(results["documents"][0][i][:1000])

    print("\nMetadata:")
    print(results["metadatas"][0][i])

    print("\nDistance:")
    print(results["distances"][0][i])