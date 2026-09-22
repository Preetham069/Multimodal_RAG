from src.loaders import (
    find_pdf_files,
    load_pdf,
    chunk_pdf_pages,
    find_image_files,
    load_image
)


# ============================================================
# TEST PDF
# ============================================================

print("\n========================================")
print("PDF LOADER TEST")
print("========================================")

pdf_files = find_pdf_files()

print("PDF files found:", len(pdf_files))

for pdf in pdf_files:
    print("PDF:", pdf)


if pdf_files:

    pages = load_pdf(pdf_files[0])

    print("\nPDF pages loaded:", len(pages))

    chunks = chunk_pdf_pages(pages)

    print("PDF chunks created:", len(chunks))

    print("\nFirst chunk:")
    print(chunks[0]["text"][:500])

    print("\nFirst chunk metadata:")
    print({
        "source": chunks[0]["source"],
        "page": chunks[0]["page"],
        "page_label": chunks[0]["page_label"],
        "type": chunks[0]["type"]
    })


# ============================================================
# TEST IMAGES
# ============================================================

print("\n========================================")
print("IMAGE LOADER TEST")
print("========================================")

image_files = find_image_files()

print("Images found:", len(image_files))

for image_path in image_files:

    image = load_image(image_path)

    print("\nImage:", image_path)
    print("Format:", image.format)
    print("Size:", image.size)
    print("Mode:", image.mode)


print("\n========================================")
print("LOADER TEST COMPLETED!")
print("========================================")