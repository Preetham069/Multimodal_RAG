import os
from pathlib import Path

from dotenv import load_dotenv


# ============================================================
# PROJECT ROOT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv(BASE_DIR / ".env")


# ============================================================
# GEMINI API KEY
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY was not found in the .env file."
    )


# ============================================================
# GEMINI MODELS
# ============================================================

EMBEDDING_MODEL = "gemini-embedding-2"

LLM_MODEL = "gemini-3.6-flash"


# ============================================================
# DATA PATHS
# ============================================================

DATA_DIR = BASE_DIR / "data"

PDF_DIR = DATA_DIR / "pdf"

IMAGE_DIR = DATA_DIR / "images"


# ============================================================
# VECTOR DATABASE
# ============================================================

CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "multimodal_rag"


# ============================================================
# RAG SETTINGS
# ============================================================

CHUNK_SIZE = 1000

CHUNK_OVERLAP = 200

RETRIEVAL_TOP_K = 5