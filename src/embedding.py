'''
generating embeddings for the chunked documents for RAG system
'''

from langchain_huggingface import HuggingFaceEmbeddings
import torch


# 3.GENERATING EMBEDDINGS FOR THE CHUNKED DOCUMENTS

# 1. HuggingFace Embeddings
def huggingface_embeddings(model_name="BAAI/bge-small-en-v1.5"):
    '''Generate embeddings for the chunked documents using HuggingFaceEmbeddings'''

    print("\n[INFO] HuggingFace Embedding model Initializing...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n[INFO] Using device: {device}")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, show_progress=True,
        model_kwargs={
            'device': device
        },
        encode_kwargs={
            'batch_size': 32,
            'normalize_embeddings': True
        }

    )

    print(f"[INFO] Model loaded successfully on {device.upper()}")

    print("=" * 50)

    return embeddings

