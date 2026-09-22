from src.vector_store import get_collection
from src.retriever import retrieve_multimodal


# ============================================================
# CONNECT TO VECTOR DATABASE
# ============================================================

collection = get_collection()


# ============================================================
# TEST QUERY
# ============================================================

query = "What technologies are used in the proposed system?"

print("\n========================================")
print("RETRIEVER TEST")
print("========================================")

print("\nQuery:")
print(query)


# ============================================================
# RETRIEVE PDF + IMAGE
# ============================================================

results = retrieve_multimodal(
    collection,
    query,
    pdf_top_k=5,
    image_top_k=1
)


# ============================================================
# DISPLAY PDF RESULTS
# ============================================================

print("\n========================================")
print("PDF RESULTS")
print("========================================")

pdf_results = results["pdf_documents"]

print("Number of PDF results:", len(pdf_results))

for i, result in enumerate(pdf_results):

    print("\n----------------------------------------")
    print(f"PDF RESULT {i + 1}")
    print("----------------------------------------")

    print("Content:")
    print(result["content"][:500])

    print("\nMetadata:")
    print(result["metadata"])

    print("\nDistance:")
    print(result["distance"])


# ============================================================
# DISPLAY IMAGE RESULTS
# ============================================================

print("\n========================================")
print("IMAGE RESULTS")
print("========================================")

image_results = results["images"]

print("Number of image results:", len(image_results))

for i, result in enumerate(image_results):

    print("\n----------------------------------------")
    print(f"IMAGE RESULT {i + 1}")
    print("----------------------------------------")

    print("Metadata:")
    print(result["metadata"])

    print("\nDistance:")
    print(result["distance"])


# ============================================================
# FINAL STATUS
# ============================================================

print("\n========================================")
print("RETRIEVER TEST COMPLETED!")
print("========================================")