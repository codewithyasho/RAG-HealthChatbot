from src.embedding import huggingface_embeddings
from src.vectorstore import load_vectorstore
from src.chain import create_rag_chain
from dotenv import load_dotenv
load_dotenv()

vectorstore_path = "faiss_index"

embeddings = huggingface_embeddings()

vectorstore = load_vectorstore(
    embeddings=embeddings, vectorstore_path=vectorstore_path)

chain = create_rag_chain(vectorstore=vectorstore)

while True:
    query = input("\nEnter your query: ")

    response = chain.invoke({"input": query})

    print("\nHealth Chatbot:")
    print(response["answer"])
    print("=" * 60)

