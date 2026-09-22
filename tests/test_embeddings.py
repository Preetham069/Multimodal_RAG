import os

from dotenv import load_dotenv
from google import genai


# --------------------------------------------------
# 1. LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found."
    )


# --------------------------------------------------
# 2. CREATE GEMINI CLIENT
# --------------------------------------------------

client = genai.Client(
    api_key=api_key
)


# --------------------------------------------------
# 3. TEXT TO EMBEDDING
# --------------------------------------------------

text = """
Machine Learning and Natural Language Processing
are used to create adaptive coding quizzes and
provide personalized feedback to students.
"""


result = client.models.embed_content(
    model="gemini-embedding-2",
    contents=text
)


# --------------------------------------------------
# 4. GET EMBEDDING
# --------------------------------------------------

embedding = result.embeddings[0].values


# --------------------------------------------------
# 5. DISPLAY RESULTS
# --------------------------------------------------

print("\nEMBEDDING SUCCESSFUL!")
print("--------------------------")

print("Embedding type:", type(embedding))

print("Embedding dimensions:", len(embedding))

print("\nFirst 10 values:")

print(embedding[:10])