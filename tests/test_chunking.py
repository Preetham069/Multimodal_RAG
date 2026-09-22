from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# 1. LOAD PDF
# --------------------------------------------------

pdf_path = "data\\pdf\\IEEE conference paper.pdf"

loader = PyPDFLoader(pdf_path)

documents = loader.load()


# --------------------------------------------------
# 2. CREATE TEXT SPLITTER
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


# --------------------------------------------------
# 3. SPLIT DOCUMENTS
# --------------------------------------------------

chunks = text_splitter.split_documents(documents)


# --------------------------------------------------
# 4. DISPLAY RESULTS
# --------------------------------------------------

print("\nDOCUMENT CHUNKING SUCCESSFUL!")
print("--------------------------------")

print("Number of PDF pages:", len(documents))

print("Number of chunks:", len(chunks))


# --------------------------------------------------
# 5. DISPLAY FIRST 3 CHUNKS
# --------------------------------------------------

for i, chunk in enumerate(chunks[:3]):

    print("\n==============================")
    print(f"CHUNK {i + 1}")
    print("==============================")

    print(chunk.page_content)

    print("\nMetadata:")
    print(chunk.metadata)