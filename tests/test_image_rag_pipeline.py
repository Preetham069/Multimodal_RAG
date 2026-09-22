from src.vector_store import get_collection
from src.rag_pipeline import answer_query


collection = get_collection()

query = """
What is shown in the retrieved image?
Describe the visible subject and important visual details.
"""

print("\n========================================")
print("IMAGE RAG PIPELINE TEST")
print("========================================")

print("\nQuestion:")
print(query)

result = answer_query(
    collection=collection,
    query=query,
    pdf_top_k=3,
    image_top_k=1
)

print("\n========================================")
print("GENERATED ANSWER")
print("========================================")

print(result["answer"])

print("\n========================================")
print("IMAGE SOURCES")
print("========================================")

for image_result in result["image_results"]:
    print(image_result["metadata"])

print("\n========================================")
print("IMAGE RAG TEST COMPLETED!")
print("========================================")