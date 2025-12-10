from src.prompt import medical_information_prompt
from src.embedding import huggingface_embeddings
from src.vectorstore import load_vectorstore
from src.chain import create_rag_chain
from dotenv import load_dotenv
load_dotenv()

embeddings = huggingface_embeddings("BAAI/bge-small-en-v1.5")

vectorstore = load_vectorstore(
    embeddings=embeddings,
    vectorstore_path="faiss_index")

prompt = medical_information_prompt()

chain = create_rag_chain(vectorstore=vectorstore, prompt=prompt)

while True:
    query = input("\nEnter your query: ")

    response = chain.invoke({"input": query})

    print("\nHealth Chatbot:")
    print(response["answer"])
    print("=" * 60)
