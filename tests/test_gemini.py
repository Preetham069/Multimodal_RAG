import os

from dotenv import load_dotenv
from google import genai


# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found in the .env file."
    )

# Create Gemini client
client = genai.Client(
    api_key=api_key
)

# Send request to Gemini
response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Explain RAG in one simple sentence."
)

print("\nGemini Response:")
print(response.text)