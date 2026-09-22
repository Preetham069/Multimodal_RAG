from src.embeddings import embed_text, embed_image


# ============================================================
# TEXT EMBEDDING TEST
# ============================================================

print("\n========================================")
print("TEXT EMBEDDING TEST")
print("========================================")

text = """
Machine Learning and Natural Language Processing
are used in adaptive educational systems.
"""

text_embedding = embed_text(text)

print("Text embedding successful!")
print("Dimensions:", len(text_embedding))
print("First 5 values:", text_embedding[:5])


# ============================================================
# IMAGE EMBEDDING TEST
# ============================================================

print("\n========================================")
print("IMAGE EMBEDDING TEST")
print("========================================")

image_path = "data/images/test.jpg"

image_embedding = embed_image(image_path)

print("Image embedding successful!")
print("Dimensions:", len(image_embedding))
print("First 5 values:", image_embedding[:5])


# ============================================================
# FINAL CHECK
# ============================================================

print("\n========================================")
print("EMBEDDING MODULE TEST COMPLETED!")
print("========================================")