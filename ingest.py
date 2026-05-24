from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


load_dotenv()

DATA_DIR = Path("data")
VECTOR_DB_DIR = "vector_db"
COLLECTION_NAME = "research_papers"


def load_pdfs():
    """
    Loads all PDF files from the data folder.
    Each page becomes one LangChain Document.
    """
    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError("No PDF files found inside the data folder.")

    all_documents = []

    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))
        documents = loader.load()

        for doc in documents:
            doc.metadata["source_file"] = pdf_file.name

        all_documents.extend(documents)

    print(f"\nLoaded {len(all_documents)} pages from {len(pdf_files)} PDFs.")
    return all_documents


def split_documents(documents):
    """
    Splits pages into smaller chunks.
    This helps retrieval because the agent searches smaller, focused pieces of text.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Split into {len(chunks)} chunks.")
    return chunks


def create_vector_database(chunks):
    """
    Creates embeddings and stores them in a persistent Chroma vector database.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR,
        collection_name=COLLECTION_NAME,
    )

    print(f"Vector database created inside: {VECTOR_DB_DIR}")
    return vector_db


def main():
    documents = load_pdfs()
    chunks = split_documents(documents)
    create_vector_database(chunks)

    print("\nIngestion complete. Your papers are now searchable.")


if __name__ == "__main__":
    main()