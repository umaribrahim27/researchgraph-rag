from typing import TypedDict, List, Literal
from pathlib import Path
import json
from datetime import datetime

from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END


load_dotenv()

VECTOR_DB_DIR = "vector_db"
COLLECTION_NAME = "research_papers"
MEMORY_FILE = Path("memory.json")


class ResearchAgentState(TypedDict):
    question: str
    rewritten_question: str
    intent: str
    documents: List[Document]
    source_scores: List[dict]
    context: str
    memory_context: str
    answer: str
    should_continue: bool


def load_vector_database():
    """
    Loads the existing Chroma vector database.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_db = Chroma(
        persist_directory=VECTOR_DB_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    return vector_db


vector_db = load_vector_database()


def read_memory():
    """
    Reads previous conversations from memory.json.
    """
    if not MEMORY_FILE.exists():
        MEMORY_FILE.write_text("[]", encoding="utf-8")

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as file:
            memory = json.load(file)
    except json.JSONDecodeError:
        memory = []

    return memory


def write_memory(memory):
    """
    Writes updated memory to memory.json.
    """
    with open(MEMORY_FILE, "w", encoding="utf-8") as file:
        json.dump(memory, file, indent=4)


def check_intent(state: ResearchAgentState) -> ResearchAgentState:
    """
    Checks whether the user wants to continue or quit.
    """
    question = state["question"].lower().strip()

    quit_commands = ["exit", "quit", "bye", "goodbye", "stop", "end"]

    if question in quit_commands:
        state["should_continue"] = False
        state["answer"] = "Goodbye. Research assistant closed."
    else:
        state["should_continue"] = True

    return state


def route_after_intent_check(state: ResearchAgentState) -> Literal["continue", "quit"]:
    """
    Routes the graph after checking whether the user wants to quit.
    """
    if state["should_continue"]:
        return "continue"

    return "quit"


def load_memory(state: ResearchAgentState) -> ResearchAgentState:
    """
    Loads recent memory and converts it into context for the LLM.
    """
    memory = read_memory()
    recent_memory = memory[-5:]

    if not recent_memory:
        state["memory_context"] = "No previous conversation memory yet."
        return state

    memory_parts = []

    for i, item in enumerate(recent_memory, start=1):
        memory_parts.append(
            f"""
Memory {i}
Previous question: {item.get("question", "")}
Rewritten question: {item.get("rewritten_question", "")}
Intent: {item.get("intent", "")}
Previous answer: {item.get("answer", "")}
Sources used: {", ".join(item.get("sources", []))}
"""
        )

    state["memory_context"] = "\n\n".join(memory_parts)

    return state


def rewrite_question(state: ResearchAgentState) -> ResearchAgentState:
    """
    Rewrites follow-up questions into standalone retrieval questions.
    """
    question = state["question"]
    memory_context = state["memory_context"]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template(
        """
You rewrite user questions for a research paper retrieval system.

Your job:
- If the current question is already clear, return it unchanged.
- If the current question depends on previous conversation, rewrite it into a standalone question.
- Do not answer the question.
- Only return the rewritten question.

Previous conversation memory:
{memory_context}

Current user question:
{question}

Rewritten standalone question:
"""
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "question": question,
            "memory_context": memory_context,
        }
    )

    state["rewritten_question"] = response.content.strip()

    return state


def classify_intent(state: ResearchAgentState) -> ResearchAgentState:
    """
    Classifies the user's request into one research assistant mode.
    """
    question = state["question"]
    rewritten_question = state["rewritten_question"]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template(
        """
You are an intent classifier for a research paper assistant.

Classify the user's request into exactly one of these labels:

direct_answer:
Use this when the user asks a normal question about the papers.

summary:
Use this when the user asks to summarise one paper, one method, one section, or a topic.

comparison:
Use this when the user asks to compare papers, methods, models, datasets, results, strengths, or limitations.

list_sources:
Use this when the user asks which papers/sources mention something, or asks for relevant papers only.

Return only one label. Do not explain.

Original question:
{question}

Rewritten standalone question:
{rewritten_question}

Intent:
"""
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "question": question,
            "rewritten_question": rewritten_question,
        }
    )

    intent = response.content.strip().lower()

    allowed_intents = [
        "direct_answer",
        "summary",
        "comparison",
        "list_sources",
    ]

    if intent not in allowed_intents:
        intent = "direct_answer"

    state["intent"] = intent

    return state


def route_by_intent(
    state: ResearchAgentState,
) -> Literal["direct_answer", "summary", "comparison", "list_sources"]:
    """
    Routes the graph based on the classified intent.
    """
    return state["intent"]


def retrieve_documents(state: ResearchAgentState) -> ResearchAgentState:
    """
    Retrieves relevant chunks from Chroma using the rewritten question.
    Also stores relevance scores.
    """
    question = state["rewritten_question"]

    results = vector_db.similarity_search_with_relevance_scores(question, k=5)

    documents = []
    source_scores = []

    for doc, score in results:
        documents.append(doc)

        source_file = doc.metadata.get("source_file", "Unknown file")
        page = doc.metadata.get("page", "Unknown page")

        source_scores.append(
            {
                "source_file": source_file,
                "page": page,
                "score": round(score, 4),
            }
        )

    state["documents"] = documents
    state["source_scores"] = source_scores

    return state


def build_context(state: ResearchAgentState) -> ResearchAgentState:
    """
    Builds the retrieved paper context for the LLM.
    """
    documents = state["documents"]

    context_parts = []

    for i, doc in enumerate(documents, start=1):
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

    state["context"] = "\n\n".join(context_parts)

    return state


def generate_direct_answer(state: ResearchAgentState) -> ResearchAgentState:
    """
    Generates a normal grounded answer.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template(
        """
You are a research assistant with memory.

Answer the user's question using the retrieved paper context as the main source of truth.
Use memory only to understand follow-up context, not as factual evidence.

Rules:
- Do not make up information.
- Mention source file names when useful.
- If the context is not enough, say so clearly.
- Keep the answer clear and structured.

Previous conversation memory:
{memory_context}

Original question:
{question}

Rewritten retrieval question:
{rewritten_question}

Retrieved paper context:
{context}
"""
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "question": state["question"],
            "rewritten_question": state["rewritten_question"],
            "context": state["context"],
            "memory_context": state["memory_context"],
        }
    )

    state["answer"] = response.content

    return state


def generate_summary_answer(state: ResearchAgentState) -> ResearchAgentState:
    """
    Generates a structured summary answer.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template(
        """
You are a research paper summarisation assistant.

Summarise the relevant paper content using only the retrieved context.

Structure your answer like this:

1. Main idea
2. Method / approach
3. Dataset or experimental setup, if available
4. Results, if available
5. Limitations, if available
6. Why it matters

Rules:
- Do not invent missing details.
- If dataset, results, or limitations are not available in the context, say "Not clearly provided in the retrieved context."
- Mention source file names when useful.

Previous conversation memory:
{memory_context}

Original question:
{question}

Rewritten retrieval question:
{rewritten_question}

Retrieved paper context:
{context}
"""
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "question": state["question"],
            "rewritten_question": state["rewritten_question"],
            "context": state["context"],
            "memory_context": state["memory_context"],
        }
    )

    state["answer"] = response.content

    return state


def generate_comparison_answer(state: ResearchAgentState) -> ResearchAgentState:
    """
    Generates a comparison-style answer.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template(
        """
You are a research comparison assistant.

Compare the relevant papers, methods, models, datasets, or findings using only the retrieved context.

Structure your answer like this:

1. What is being compared
2. Similarities
3. Differences
4. Strengths of each approach
5. Weaknesses or limitations
6. Best use case / practical takeaway

Rules:
- Do not make up missing information.
- If comparison is not possible from the retrieved context, say that clearly.
- Mention source file names when useful.
- Prefer bullet points or a compact table-style structure.

Previous conversation memory:
{memory_context}

Original question:
{question}

Rewritten retrieval question:
{rewritten_question}

Retrieved paper context:
{context}
"""
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "question": state["question"],
            "rewritten_question": state["rewritten_question"],
            "context": state["context"],
            "memory_context": state["memory_context"],
        }
    )

    state["answer"] = response.content

    return state


def generate_sources_answer(state: ResearchAgentState) -> ResearchAgentState:
    """
    Generates a source-list answer with relevance scores.
    """
    source_scores = state["source_scores"]
    question = state["question"]
    rewritten_question = state["rewritten_question"]

    if not source_scores:
        state["answer"] = "I could not find relevant sources for that question."
        return state

    source_lines = []

    for item in source_scores:
        source_file = item.get("source_file", "Unknown file")
        page = item.get("page", "Unknown page")
        score = item.get("score", "N/A")

        source_lines.append(
            f"- {source_file}, page {page} | relevance: {score}"
        )

    state["answer"] = (
        f"Relevant sources for your question:\n\n"
        f"Original question: {question}\n"
        f"Search query used: {rewritten_question}\n\n"
        + "\n".join(source_lines)
    )

    return state


def save_memory(state: ResearchAgentState) -> ResearchAgentState:
    """
    Saves the current interaction into local memory.
    """
    memory = read_memory()

    sources = []

    for doc in state["documents"]:
        source_file = doc.metadata.get("source_file", "Unknown file")
        page = doc.metadata.get("page", "Unknown page")
        sources.append(f"{source_file}, page {page}")

    memory_item = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": state["question"],
        "rewritten_question": state["rewritten_question"],
        "intent": state["intent"],
        "answer": state["answer"],
        "sources": sources,
        "source_scores": state["source_scores"],
    }

    memory.append(memory_item)

    memory = memory[-30:]

    write_memory(memory)

    return state


def create_graph():
    """
    Creates the full LangGraph workflow.
    """
    graph = StateGraph(ResearchAgentState)

    graph.add_node("check_intent", check_intent)
    graph.add_node("load_memory", load_memory)
    graph.add_node("rewrite_question", rewrite_question)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_documents", retrieve_documents)
    graph.add_node("build_context", build_context)

    graph.add_node("generate_direct_answer", generate_direct_answer)
    graph.add_node("generate_summary_answer", generate_summary_answer)
    graph.add_node("generate_comparison_answer", generate_comparison_answer)
    graph.add_node("generate_sources_answer", generate_sources_answer)

    graph.add_node("save_memory", save_memory)

    graph.add_edge(START, "check_intent")

    graph.add_conditional_edges(
        "check_intent",
        route_after_intent_check,
        {
            "continue": "load_memory",
            "quit": END,
        },
    )

    graph.add_edge("load_memory", "rewrite_question")
    graph.add_edge("rewrite_question", "classify_intent")
    graph.add_edge("classify_intent", "retrieve_documents")
    graph.add_edge("retrieve_documents", "build_context")

    graph.add_conditional_edges(
        "build_context",
        route_by_intent,
        {
            "direct_answer": "generate_direct_answer",
            "summary": "generate_summary_answer",
            "comparison": "generate_comparison_answer",
            "list_sources": "generate_sources_answer",
        },
    )

    graph.add_edge("generate_direct_answer", "save_memory")
    graph.add_edge("generate_summary_answer", "save_memory")
    graph.add_edge("generate_comparison_answer", "save_memory")
    graph.add_edge("generate_sources_answer", "save_memory")

    graph.add_edge("save_memory", END)

    app = graph.compile()

    return app


def print_sources(source_scores):
    """
    Prints ranked retrieved sources.
    """
    if not source_scores:
        return

    print("\nRanked sources used:")

    for item in source_scores:
        source_file = item.get("source_file", "Unknown file")
        page = item.get("page", "Unknown page")
        score = item.get("score", "N/A")

        print(f"- {source_file}, page {page} | relevance: {score}")


def main():
    """
    Runs the command-line research assistant.
    """
    app = create_graph()

    print("\nLangGraph Research Paper Assistant")
    print("Ask questions about your uploaded research papers.")
    print("Type 'exit', 'quit', or 'bye' to close.\n")

    while True:
        question = input("Question: ")

        initial_state = {
            "question": question,
            "rewritten_question": "",
            "intent": "",
            "documents": [],
            "source_scores": [],
            "context": "",
            "memory_context": "",
            "answer": "",
            "should_continue": True,
        }

        final_state = app.invoke(initial_state)

        if final_state.get("should_continue", True):
            print("\nIntent:")
            print(final_state.get("intent", "unknown"))

            print("\nRewritten retrieval question:")
            print(final_state.get("rewritten_question", ""))

        print("\nAnswer:")
        print(final_state.get("answer", ""))

        if final_state.get("should_continue", True):
            print_sources(final_state.get("source_scores", []))

        print("\n" + "-" * 80 + "\n")

        if final_state.get("should_continue") is False:
            break


if __name__ == "__main__":
    main()