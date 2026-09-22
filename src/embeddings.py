from google import genai
from google.genai import types

from src.config import (
    GEMINI_API_KEY,
    EMBEDDING_MODEL
)


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# TEXT EMBEDDING
# ============================================================

def embed_text(text):
    """
    Convert text into a Gemini Embedding 2 vector.
    """

    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text
    )

    return result.embeddings[0].values


# ============================================================
# IMAGE EMBEDDING
# ============================================================

def embed_image(image_path):
    """
    Convert an image into a Gemini Embedding 2 vector.
    """

    image_path = str(image_path)

    with open(image_path, "rb") as file:
        image_bytes = file.read()

    mime_type = get_image_mime_type(image_path)

    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type
            )
        ]
    )

    return result.embeddings[0].values


# ============================================================
# IMAGE MIME TYPE
# ============================================================

def get_image_mime_type(image_path):
    """
    Return the MIME type for a supported image.
    """

    image_path = str(image_path).lower()

    if image_path.endswith(".jpg") or image_path.endswith(".jpeg"):
        return "image/jpeg"

    if image_path.endswith(".png"):
        return "image/png"

    if image_path.endswith(".webp"):
        return "image/webp"

    raise ValueError(
        f"Unsupported image format: {image_path}"
    )