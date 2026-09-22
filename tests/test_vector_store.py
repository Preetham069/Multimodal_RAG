from src.loaders import (
    find_pdf_files,
    load_pdf,
    chunk_pdf_pages,
    find_image_files
)

from src.vector_store import (
    get_collection,
    add_pdf_chunks,
    add_image,
    get_collection_info
)


# ============================================================
# 1. GET CHROMADB COLLECTION
# ============================================================

print("\n========================================")
print("VECTOR STORE TEST")
print("========================================")

collection = get_collection()

print("Collection:", collection.name)

print(
    "Existing records:",
    collection.count()
)


# ============================================================
# 2. FIND PDF
# ============================================================

pdf_files = find_pdf_files()

if not pdf_files:

    raise FileNotFoundError(
        "No PDF files found in the data/pdf directory."
    )


pdf_path = pdf_files[0]

print("\nPDF selected:")
print(pdf_path)


# ============================================================
# 3. LOAD PDF
# ============================================================

pages = load_pdf(pdf_path)

print(
    "PDF pages:",
    len(pages)
)


# ============================================================
# 4. CHUNK PDF
# ============================================================

chunks = chunk_pdf_pages(pages)

print(
    "PDF chunks:",
    len(chunks)
)


# ============================================================
# 5. STORE PDF CHUNKS
# ============================================================

print("\n========================================")
print("STORING PDF CHUNKS")
print("========================================")

add_pdf_chunks(
    collection,
    chunks
)


# ============================================================
# 6. FIND IMAGES
# ============================================================

image_files = find_image_files()

print("\nImages found:", len(image_files))


# ============================================================
# 7. STORE IMAGES
# ============================================================

print("\n========================================")
print("STORING IMAGES")
print("========================================")

for index, image_path in enumerate(image_files):

    image_id = f"image_{index}"

    add_image(
        collection,
        image_path,
        image_id
    )


# ============================================================
# 8. DATABASE INFORMATION
# ============================================================

info = get_collection_info(
    collection
)


print("\n========================================")
print("VECTOR STORE COMPLETE")
print("========================================")

print(
    "Collection:",
    info["collection_name"]
)

print(
    "Total records:",
    info["total_records"]
)

print("\nExpected:")
print("37 PDF chunks + 1 image = 38 records")