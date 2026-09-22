import chromadb

from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME
)

from src.embeddings import (
    embed_text,
    embed_image
)


# ============================================================
# CHROMADB CLIENT
# ============================================================

def get_chroma_client():
    """
    Create and return a persistent ChromaDB client.
    """

    return chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )


# ============================================================
# GET COLLECTION
# ============================================================

def get_collection():
    """
    Get or create the multimodal RAG collection.
    """

    client = get_chroma_client()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    return collection


# ============================================================
# ADD PDF CHUNKS
# ============================================================

def add_pdf_chunks(collection, chunks):
    """
    Generate embeddings for PDF chunks and store them
    in ChromaDB.
    """

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for index, chunk in enumerate(chunks):

        text = chunk["text"]

        embedding = embed_text(text)

        ids.append(
            f"pdf_chunk_{index}"
        )

        embeddings.append(
            embedding
        )

        documents.append(
            text
        )

        metadatas.append({
            "type": "pdf_text",
            "source": chunk["source"],
            "page": str(chunk["page"]),
            "page_label": str(chunk["page_label"])
        })

        print(
            f"PDF chunk {index + 1}/{len(chunks)} embedded"
        )

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    print(
        f"\n{len(chunks)} PDF chunks stored successfully."
    )


# ============================================================
# ADD IMAGE
# ============================================================

def add_image(collection, image_path, image_id):
    """
    Generate an embedding for an image and store it
    in ChromaDB.
    """

    embedding = embed_image(image_path)

    collection.upsert(
        ids=[image_id],

        embeddings=[embedding],

        documents=[
            f"Image file: {image_path.name}"
        ],

        metadatas=[{
            "type": "image",
            "source": str(image_path)
        }]
    )

    print(
        f"Image stored successfully: {image_path}"
    )


# ============================================================
# DATABASE INFORMATION
# ============================================================

def get_collection_info(collection):
    """
    Return basic information about the collection.
    """

    return {
        "collection_name": collection.name,
        "total_records": collection.count()
    }