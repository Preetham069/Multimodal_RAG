from src.vector_store import get_collection
from src.rag_pipeline import answer_query


# ============================================================
# CONNECT TO VECTOR DATABASE
# ============================================================

collection = get_collection()


# ============================================================
# TEST QUESTION
# ============================================================

query = """
What technologies are used in the proposed system?
"""


print("\n========================================")
print("FINAL RAG PIPELINE TEST")
print("========================================")

print("\nQuestion:")
print(query)


# ============================================================
# RUN RAG
# ============================================================

result = answer_query(
    collection=collection,
    query=query,
    pdf_top_k=5,
    image_top_k=1
)


# ============================================================
# DISPLAY ANSWER
# ============================================================

print("\n========================================")
print("GENERATED ANSWER")
print("========================================")

print(result["answer"])


# ============================================================
# DISPLAY SOURCES
# ============================================================

print("\n========================================")
print("PDF SOURCES")
print("========================================")

for pdf_result in result["pdf_results"]:

    print(
        pdf_result["metadata"]
    )


print("\n========================================")
print("IMAGE SOURCES")
print("========================================")

for image_result in result["image_results"]:

    print(
        image_result["metadata"]
    )


print("\n========================================")
print("RAG PIPELINE TEST COMPLETED!")
print("========================================")