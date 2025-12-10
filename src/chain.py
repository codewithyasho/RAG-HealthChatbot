"""
Simple and modern RAG chain setup using LangChain Classic (Groq + FAISS).
"""

from langchain_groq import ChatGroq
from src.prompt import medical_information_prompt
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


def create_rag_chain(vectorstore):
    """
    Create and return a complete RAG chain.
    This function:
    - Creates retriever from FAISS vectorstore
    - Initializes Groq LLM
    - Builds the retrieval chain (retriever + LLM + prompt)
    """

    print("\n🚀 Initializing RAG chain...")

    # 1️⃣ Create retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 7}
    )

    # 2️⃣ Initialize LLM (Groq)
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.2,
    )

    # 3️⃣ Define prompt template

    # PROMPT FOR MEDICAL INFORMATION / HEALTH CHATBOT
    prompt = medical_information_prompt()

    # 4️⃣ Build RAG chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)

    print("✅✅ RAG chain created successfully!\n" + "=" * 60)

    return rag_chain
