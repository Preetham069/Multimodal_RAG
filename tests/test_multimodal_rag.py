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

query = """
What is shown in the image, and what are the main
technologies discussed in the PDF?
"""

print("\n========================================")
print("MULTIMODAL RAG TEST")
print("========================================")

print("\nUser Question:")
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
# 6. RETRIEVE PDF TEXT
# ============================================================

pdf_results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    where={
        "type": "pdf_text"
    }
)


# ============================================================
# 7. RETRIEVE IMAGE
# ============================================================

image_query = """
A photograph of Lord Ganesha sitting on an ornate
decorative throne.
"""

image_query_result = client.models.embed_content(
    model=EMBEDDING_MODEL,
    contents=image_query
)

image_query_embedding = image_query_result.embeddings[0].values


image_results = collection.query(
    query_embeddings=[image_query_embedding],
    n_results=1,
    where={
        "type": "image"
    }
)


# ============================================================
# 8. PREPARE PDF CONTEXT
# ============================================================

pdf_context = ""

for i in range(len(pdf_results["documents"][0])):

    document = pdf_results["documents"][0][i]

    metadata = pdf_results["metadatas"][0][i]

    pdf_context += f"""
PDF SOURCE:
{metadata.get("source", "")}

PAGE:
{metadata.get("page_label", "")}

CONTENT:
{document}

----------------------------------------
"""


# ============================================================
# 9. LOAD RETRIEVED IMAGE
# ============================================================

image_parts = []

image_source = ""

if len(image_results["metadatas"][0]) > 0:

    image_metadata = image_results["metadatas"][0][0]

    image_source = image_metadata.get("source", "")

    print("\nRetrieved image:")
    print(image_source)

    if os.path.exists(image_source):

        with open(image_source, "rb") as f:
            image_bytes = f.read()

        image_parts.append(
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            )
        )


# ============================================================
# 10. DISPLAY RETRIEVAL INFORMATION
# ============================================================

print("\n========================================")
print("PDF RETRIEVAL")
print("========================================")

print("PDF chunks retrieved:", len(pdf_results["documents"][0]))

for i in range(len(pdf_results["documents"][0])):

    print(
        f"PDF Result {i + 1}:",
        pdf_results["metadatas"][0][i]
    )


print("\n========================================")
print("IMAGE RETRIEVAL")
print("========================================")

if image_source:
    print("Image retrieved successfully!")
    print("Source:", image_source)
else:
    print("No image retrieved.")


# ============================================================
# 11. CREATE MULTIMODAL RAG PROMPT
# ============================================================

prompt = f"""
You are a multimodal RAG assistant.

Answer the user's question using the retrieved PDF
information and the retrieved image.

Do not invent information.

For the PDF portion, use only the retrieved PDF context.

For the image portion, directly analyze the provided image.

User Question:
{query}

================ PDF CONTEXT ================

{pdf_context}

================ IMAGE SOURCE ================

{image_source}

Now provide a clear answer with two sections:

1. IMAGE DESCRIPTION
2. TECHNOLOGIES FROM THE PDF
"""


# ============================================================
# 12. SEND TEXT + IMAGE TO GEMINI
# ============================================================

contents = [prompt]

contents.extend(image_parts)


# ============================================================
# 13. GENERATE FINAL MULTIMODAL ANSWER
# ============================================================

response = client.models.generate_content(
    model=LLM_MODEL,
    contents=contents
)


# ============================================================
# 14. DISPLAY FINAL ANSWER
# ============================================================

print("\n========================================")
print("MULTIMODAL RAG ANSWER")
print("========================================")

print(response.text)


# ============================================================
# 15. DISPLAY SOURCES
# ============================================================

print("\n========================================")
print("SOURCES")
print("========================================")

print("\nPDF Sources:")

for metadata in pdf_results["metadatas"][0]:
    print(metadata)

print("\nImage Source:")
print(image_source)