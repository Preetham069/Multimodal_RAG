import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

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

LLM_MODEL = "gemini-3.6-flash"


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
# 4. USER QUESTION
# ============================================================

query = "What technologies are used in the proposed system?"

print("\n========================================")
print("USER QUESTION")
print("========================================")

print(query)


# ============================================================
# 5. CREATE QUERY EMBEDDING
# ============================================================

query_result = client.models.embed_content(
    model=EMBEDDING_MODEL,
    contents=query
)

query_embedding = query_result.embeddings[0].values


# ============================================================
# 6. RETRIEVE RELEVANT DOCUMENTS
# ============================================================

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)


# ============================================================
# 7. PREPARE RETRIEVED CONTEXT
# ============================================================

context_parts = []

image_parts = []


for i in range(len(results["documents"][0])):

    document = results["documents"][0][i]

    metadata = results["metadatas"][0][i]

    item_type = metadata.get("type")

    # --------------------------------------------------------
    # PDF TEXT
    # --------------------------------------------------------

    if item_type == "pdf_text":

        source = metadata.get("source", "")
        page = metadata.get("page_label", "")

        context_parts.append(
            f"""
SOURCE: {source}
PAGE: {page}

CONTENT:
{document}
"""
        )

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    elif item_type == "image":

        image_path = metadata.get("source")

        context_parts.append(
            f"""
SOURCE IMAGE: {image_path}

The image has been retrieved as relevant visual information.
"""
        )

        # Load the image
        if os.path.exists(image_path):

            with open(image_path, "rb") as f:
                image_bytes = f.read()

            image_parts.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg"
                )
            )


# Combine text context
context = "\n".join(context_parts)


# ============================================================
# 8. CREATE RAG PROMPT
# ============================================================

prompt = f"""
You are a helpful RAG assistant.

Answer the user's question using ONLY the retrieved
information provided below.

If the retrieved information does not contain enough
information to answer the question, clearly say that
the information is not available in the retrieved
documents.

Do not invent facts.

User Question:
{query}

Retrieved Context:
{context}

Provide a clear and concise answer.
"""


# ============================================================
# 9. CREATE CONTENT FOR GEMINI
# ============================================================

contents = [prompt]

# Add retrieved images if any were found
contents.extend(image_parts)


# ============================================================
# 10. GENERATE FINAL ANSWER
# ============================================================

response = client.models.generate_content(
    model=LLM_MODEL,
    contents=contents
)


# ============================================================
# 11. DISPLAY FINAL ANSWER
# ============================================================

print("\n========================================")
print("RAG ANSWER")
print("========================================")

print(response.text)


# ============================================================
# 12. DISPLAY SOURCES
# ============================================================

print("\n========================================")
print("SOURCES USED")
print("========================================")

for i in range(len(results["metadatas"][0])):

    metadata = results["metadatas"][0][i]

    print(f"\nSource {i + 1}:")
    print(metadata)