from src.embeddings import embed_text
from src.config import RETRIEVAL_TOP_K


# ============================================================
# QUERY CLASSIFICATION
# ============================================================

def classify_query(query):
    """
    Classifies the user's question into:

    image       -> image-related question
    pdf         -> PDF/document-related question
    multimodal  -> requires both image and PDF
    """

    query_lower = query.lower().strip()

    # --------------------------------------------------------
    # Image-related keywords
    # --------------------------------------------------------

    image_keywords = [
        "image",
        "picture",
        "photo",
        "shown",
        "visible",
        "visual",
        "look like",
        "looks like",
        "describe the image",
        "describe this image",
        "what do you see",
        "what is in the image",
        "what is shown"
    ]

    # --------------------------------------------------------
    # PDF/document-related keywords
    # --------------------------------------------------------

    pdf_keywords = [
        "pdf",
        "paper",
        "document",
        "page",
        "technologies",
        "technology",
        "methodology",
        "method",
        "algorithm",
        "algorithms",
        "system",
        "dataset",
        "research",
        "study",
        "according to the paper",
        "according to the document",
        "proposed system",
        "architecture",
        "model",
        "models",
        "machine learning",
        "natural language processing",
        "nlp",
        "transformer",
        "bert",
        "t5"
    ]

    has_image_intent = any(
        keyword in query_lower
        for keyword in image_keywords
    )

    has_pdf_intent = any(
        keyword in query_lower
        for keyword in pdf_keywords
    )

    # --------------------------------------------------------
    # Determine modality
    # --------------------------------------------------------

    if has_image_intent and has_pdf_intent:
        return "multimodal"

    elif has_image_intent:
        return "image"

    elif has_pdf_intent:
        return "pdf"

    else:
        # If we cannot determine the intent,
        # retrieve from both modalities.
        return "multimodal"


# ============================================================
# PDF RETRIEVAL
# ============================================================

def retrieve_pdf_documents(
    collection,
    query,
    top_k=None,
    query_embedding=None
):

    if top_k is None:
        top_k = RETRIEVAL_TOP_K

    # Retrieve more candidates than we finally need.
    candidate_k = max(top_k * 3, 12)

    # Reuse the embedding if already available.
    if query_embedding is None:
        query_embedding = embed_text(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_k,
        where={"type": "pdf_text"}
    )

    retrieved_documents = []

    if not results["documents"]:
        return retrieved_documents

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Keep track of exact duplicate chunks.
    seen_content = set()

    # Keep track of how many chunks we select from each page.
    page_counts = {}

    for i, document in enumerate(documents):

        metadata = metadatas[i]
        distance = distances[i]

        if not document or not document.strip():
            continue

        clean_content = document.strip()

        # ----------------------------------------------------
        # Remove exact duplicate content
        # ----------------------------------------------------

        content_key = " ".join(
            clean_content.lower().split()
        )

        if content_key in seen_content:
            continue

        seen_content.add(content_key)

        # ----------------------------------------------------
        # Page information
        # ----------------------------------------------------

        page = metadata.get(
            "page_label",
            "Unknown"
        )

        # ----------------------------------------------------
        # Limit repeated chunks from the same page
        # ----------------------------------------------------

        current_page_count = page_counts.get(
            page,
            0
        )

        # Maximum 2 chunks from one page
        if current_page_count >= 2:
            continue

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        retrieved_documents.append(
            {
                "content": clean_content,
                "metadata": metadata,
                "distance": distance
            }
        )

        page_counts[page] = (
            current_page_count + 1
        )

        print(
            f"Selected PDF result "
            f"{len(retrieved_documents)} | "
            f"Page: {page} | "
            f"Distance: {distance:.4f}"
        )

        # ----------------------------------------------------
        # Stop when we have enough diverse results
        # ----------------------------------------------------

        if len(retrieved_documents) >= top_k:
            break

    return retrieved_documents

# ============================================================
# IMAGE RETRIEVAL
# ============================================================

def retrieve_images(
    collection,
    query,
    top_k=1,
    query_embedding=None
):

    # Reuse an existing embedding when provided
    if query_embedding is None:
        query_embedding = embed_text(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"type": "image"}
    )

    retrieved_images = []

    if not results["metadatas"]:
        return retrieved_images

    for i, metadata in enumerate(
        results["metadatas"][0]
    ):

        distance = results["distances"][0][i]

        retrieved_images.append(
            {
                "metadata": metadata,
                "distance": distance
            }
        )

    return retrieved_images


# ============================================================
# QUERY-AWARE MULTIMODAL RETRIEVAL
# ============================================================

def retrieve_multimodal(
    collection,
    query,
    pdf_top_k=None,
    image_top_k=1
):

    # --------------------------------------------------------
    # First determine what the user is asking about
    # --------------------------------------------------------

    modality = classify_query(query)
    query_embedding = embed_text(query)

    print("\n----------------------------------------")
    print("QUERY CLASSIFICATION")
    print("----------------------------------------")
    print(f"Query: {query}")
    print(f"Detected modality: {modality}")
    print("----------------------------------------")


    # --------------------------------------------------------
    # IMAGE QUESTION
    # --------------------------------------------------------

    if modality == "image":

        print("Retrieving IMAGE only...")

        image_results = retrieve_images(
            collection=collection,
            query=query,
            top_k=image_top_k,
            query_embedding=query_embedding
        )

        return {
            "modality": "image",
            "pdf_documents": [],
            "images": image_results
        }


    # --------------------------------------------------------
    # PDF QUESTION
    # --------------------------------------------------------

    elif modality == "pdf":

        print("Retrieving PDF only...")

        pdf_results = retrieve_pdf_documents(
            collection=collection,
            query=query,
            top_k=pdf_top_k,
            query_embedding=query_embedding
        )

        return {
            "modality": "pdf",
            "pdf_documents": pdf_results,
            "images": []
        }


    # --------------------------------------------------------
    # MULTIMODAL QUESTION
    # --------------------------------------------------------

    else:

        print("Retrieving PDF + IMAGE...")

        pdf_results = retrieve_pdf_documents(
            collection=collection,
            query=query,
            top_k=pdf_top_k
        )

        image_results = retrieve_images(
            collection=collection,
            query=query,
            top_k=image_top_k
        )

        return {
            "modality": "multimodal",
            "pdf_documents": pdf_results,
            "images": image_results
        }