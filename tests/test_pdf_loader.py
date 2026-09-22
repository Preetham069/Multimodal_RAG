from langchain_community.document_loaders import PyPDFLoader


# --------------------------------------------------
# PDF FILE PATH
# --------------------------------------------------

pdf_path =  "data\\pdf\\IEEE conference paper.pdf"


# --------------------------------------------------
# CREATE PDF LOADER
# --------------------------------------------------

loader = PyPDFLoader(pdf_path)


# --------------------------------------------------
# LOAD PDF
# --------------------------------------------------

documents = loader.load()


# --------------------------------------------------
# DISPLAY INFORMATION
# --------------------------------------------------

print("\nPDF LOADING SUCCESSFUL!")
print("----------------------------")

print("Number of pages:", len(documents))


# --------------------------------------------------
# DISPLAY EACH PAGE
# --------------------------------------------------

for i, document in enumerate(documents):

    print("\n==============================")
    print(f"PAGE {i + 1}")
    print("==============================")

    print(document.page_content[:1000])

    print("\nMetadata:")
    print(document.metadata)