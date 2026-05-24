from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

VECTOR_DB_DIR = "vector_db"
COLLECTION_NAME = "research_papers"


def load_vector_database():
    """
    Loads the existing Chroma vector database from disk.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_db = Chroma(
        persist_directory=VECTOR_DB_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    return vector_db


def search_papers(vector_db, question, k=5):
    """
    Searches the vector database for the most relevant chunks.
    """
    results = vector_db.similarity_search(question, k=k)
    return results


def build_context(results):
    """
    Converts retrieved chunks into a clean context string for the LLM.
    Also keeps source file and page info.
    """
    context_parts = []

    for i, doc in enumerate(results, start=1):
        source_file = doc.metadata.get("source_file", "Unknown file")
        page = doc.metadata.get("page", "Unknown page")

        context_parts.append(
            f"""
Source {i}
File: {source_file}
Page: {page}

Content:
{doc.page_content}
"""
        )

    return "\n\n".join(context_parts)


def answer_question(question, context):
    """
    Sends the retrieved paper chunks to the LLM and asks it to answer
    only using the provided context.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template(
        """
You are a research assistant.

Answer the user's question using only the context provided below.

Rules:
- Do not make up information.
- Mention the paper/source file names when useful.
- If the context is not enough, say that the uploaded papers do not provide enough information.
- Keep the answer clear and structured.

Question:
{question}

Context:
{context}
"""
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    return response.content


def main():
    vector_db = load_vector_database()

    print("\nResearch Paper Assistant")
    print("Ask a question about your uploaded papers.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Question: ")

        if question.lower().strip() == "exit":
            print("Goodbye.")
            break

        results = search_papers(vector_db, question, k=5)
        context = build_context(results)
        answer = answer_question(question, context)

        print("\nAnswer:")
        print(answer)

        print("\nSources used:")
        for doc in results:
            source_file = doc.metadata.get("source_file", "Unknown file")
            page = doc.metadata.get("page", "Unknown page")
            print(f"- {source_file}, page {page}")

        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()