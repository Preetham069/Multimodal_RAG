import os
import time

from google import genai
from google.genai import types

from src.config import (
    GEMINI_API_KEY,
    LLM_MODEL
)

from src.retriever import retrieve_multimodal


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# BUILD PDF CONTEXT
# ============================================================

def build_pdf_context(pdf_documents):
    """
    Convert retrieved PDF documents into a context string.
    """

    if not pdf_documents:
        return "No PDF information was retrieved."

    context_parts = []

    for index, document in enumerate(pdf_documents):

        metadata = document["metadata"]

        source = metadata.get("source", "")
        page = metadata.get("page_label", "")

        content = document["content"]

        context_parts.append(
            f"""
--- PDF RESULT {index + 1} ---

Source:
{source}

Page:
{page}

Content:
{content}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# LOAD RETRIEVED IMAGES
# ============================================================

def load_retrieved_images(image_results):
    """
    Load the actual image files retrieved from ChromaDB.
    """

    image_parts = []

    image_sources = []

    for result in image_results:

        metadata = result["metadata"]

        image_path = metadata.get("source")

        if not image_path:
            continue

        if not os.path.exists(image_path):
            continue

        with open(image_path, "rb") as file:
            image_bytes = file.read()

        # Currently our supported image test is JPEG.
        # This can be expanded later for PNG/WebP.
        image_parts.append(
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            )
        )

        image_sources.append(image_path)

    return image_parts, image_sources


# ============================================================
# BUILD RAG PROMPT
# ============================================================

def build_prompt(query, pdf_context, image_sources):
    """
    Create a grounded prompt for Gemini.
    """

    image_information = ""

    if image_sources:

        image_information = """
The retrieved image files are also provided to you
directly as visual input.

Analyze those images only when they are relevant
to the user's question.
"""

    else:

        image_information = """
No image was retrieved for this question.
"""


    prompt = f"""
You are a multimodal Retrieval-Augmented Generation
(RAG) assistant.

Your job is to answer the user's question using the
retrieved information provided to you.

IMPORTANT RULES:

1. Use the retrieved PDF context as the source of
   information about the PDF.

2. If an image is provided, analyze the actual image
   only when relevant to the question.

3. Do not invent facts that are not supported by the
   retrieved information or visible in the provided
   image.

4. If the retrieved information is insufficient to
   answer part of the question, clearly state that
   the available sources do not contain enough
   information.

5. Keep the answer clear and concise.

6. Mention relevant PDF page numbers when discussing
   information from the PDF.

USER QUESTION:
{query}

================ RETRIEVED PDF CONTEXT ================

{pdf_context}

================ RETRIEVED IMAGE INFORMATION ================

{image_information}

Provide the final answer now.
"""

    return prompt


# ============================================================
# RUN RAG PIPELINE
# ============================================================

def answer_query(
    collection,
    query,
    pdf_top_k=5,
    image_top_k=1
):

    # ------------------------------------------------------------
    # 1. RETRIEVE RELEVANT INFORMATION
    # ------------------------------------------------------------

    retrieval_results = retrieve_multimodal(
        collection=collection,
        query=query,
        pdf_top_k=pdf_top_k,
        image_top_k=image_top_k
    )

    pdf_documents = retrieval_results["pdf_documents"]

    image_results = retrieval_results["images"]


    # ------------------------------------------------------------
    # 2. BUILD PDF CONTEXT
    # ------------------------------------------------------------

    pdf_context = build_pdf_context(
        pdf_documents
    )


    # ------------------------------------------------------------
    # 3. LOAD RETRIEVED IMAGES
    # ------------------------------------------------------------

    image_parts, image_sources = load_retrieved_images(
        image_results
    )


    # ------------------------------------------------------------
    # 4. BUILD PROMPT
    # ------------------------------------------------------------

    prompt = build_prompt(
        query=query,
        pdf_context=pdf_context,
        image_sources=image_sources
    )


    # ------------------------------------------------------------
    # 5. COMBINE TEXT + IMAGES
    # ------------------------------------------------------------

    contents = [prompt]

    contents.extend(image_parts)


    # ------------------------------------------------------------
    # 6. GENERATE ANSWER
    # ------------------------------------------------------------

    response = None

    max_retries = 3

    for attempt in range(max_retries):

        try:

            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=contents
            )

            # Gemini responded successfully
            break

        except Exception as error:

            error_text = str(error)

            # ----------------------------------------------------
            # Handle temporary Gemini 503 errors
            # ----------------------------------------------------

            if (
                "503" in error_text
                or "UNAVAILABLE" in error_text
            ):

                if attempt < max_retries - 1:

                    wait_time = 2 ** (attempt + 1)

                    print(
                        "\nGemini temporarily unavailable."
                    )

                    print(
                        f"Retrying in {wait_time} seconds..."
                    )

                    time.sleep(wait_time)

                else:

                    raise RuntimeError(
                        "Gemini is temporarily unavailable "
                        "after multiple retry attempts. "
                        "Please try again in a few moments."
                    ) from error

            else:

                # Any other error should be raised normally
                raise


    # ------------------------------------------------------------
    # 7. RETURN RESULT
    # ------------------------------------------------------------

    return {
        "answer": response.text,
        "pdf_results": pdf_documents,
        "image_results": image_results,
        "image_sources": image_sources
    }