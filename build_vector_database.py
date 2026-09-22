import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import chromadb

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
# 3. CREATE CHROMADB DATABASE
# ============================================================

chroma_client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="multimodal_rag"
)


# ============================================================
# 4. LOAD PDF
# ============================================================

pdf_path = "data/pdf/IEEE conference paper.pdf"

print("\nLoading PDF...")

loader = PyPDFLoader(pdf_path)

documents = loader.load()

print("PDF pages loaded:", len(documents))


# ============================================================
# 5. SPLIT PDF INTO CHUNKS
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("PDF chunks created:", len(chunks))


# ============================================================
# 6. EMBED PDF CHUNKS
# ============================================================

print("\nCreating embeddings for PDF chunks...")

pdf_ids = []
pdf_embeddings = []
pdf_documents = []
pdf_metadatas = []

for i, chunk in enumerate(chunks):

    text = chunk.page_content

    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text
    )

    embedding = result.embeddings[0].values

    pdf_ids.append(f"pdf_chunk_{i}")

    pdf_embeddings.append(embedding)

    pdf_documents.append(text)

    pdf_metadatas.append({
        "type": "pdf_text",
        "source": pdf_path,
        "page": str(chunk.metadata.get("page", "")),
        "page_label": str(chunk.metadata.get("page_label", ""))
    })

    print(f"PDF chunk {i + 1}/{len(chunks)} embedded")


# ============================================================
# 7. ADD PDF DATA TO CHROMADB
# ============================================================

print("\nAdding PDF embeddings to ChromaDB...")

collection.add(
    ids=pdf_ids,
    embeddings=pdf_embeddings,
    documents=pdf_documents,
    metadatas=pdf_metadatas
)

print("PDF data added successfully!")


# ============================================================
# 8. LOAD IMAGE
# ============================================================

image_path = "data/images/test.jpg"

print("\nLoading image...")

with open(image_path, "rb") as f:
    image_bytes = f.read()


# ============================================================
# 9. CREATE IMAGE EMBEDDING
# ============================================================

print("Creating image embedding...")

image_result = client.models.embed_content(
    model=EMBEDDING_MODEL,
    contents=[
        types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg"
        )
    ]
)

image_embedding = image_result.embeddings[0].values


# ============================================================
# 10. ADD IMAGE TO CHROMADB
# ============================================================

collection.add(
    ids=["image_0"],
    embeddings=[image_embedding],
    documents=["Image file: test.jpg"],
    metadatas=[{
        "type": "image",
        "source": image_path
    }]
)

print("Image added successfully!")


# ============================================================
# 11. DISPLAY DATABASE INFORMATION
# ============================================================

print("\n========================================")
print("VECTOR DATABASE CREATED SUCCESSFULLY!")
print("========================================")

print("Collection name:", collection.name)
print("Total records:", collection.count())

print("\nPDF chunks stored:", len(pdf_ids))
print("Images stored: 1")
print("Total vectors:", collection.count())

print("\nEmbedding dimensions:", len(pdf_embeddings[0]))

print("\nDatabase location:")
print("chroma_db/")