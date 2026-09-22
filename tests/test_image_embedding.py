import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY was not found in the .env file.")

# Create Gemini client
client = genai.Client(api_key=api_key)

# Path to the image
image_path = "data/images/test.jpg"

# Read the image
with open(image_path, "rb") as f:
    image_bytes = f.read()

# Create image embedding
result = client.models.embed_content(
    model="gemini-embedding-2",
    contents=[
        types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg"
        )
    ]
)

# Get the embedding vector
embedding = result.embeddings[0].values

print("\nIMAGE EMBEDDING SUCCESSFUL!")
print("--------------------------------")
print("Image:", image_path)
print("Embedding type:", type(embedding))
print("Embedding dimensions:", len(embedding))

print("\nFirst 10 values:")
print(embedding[:10])