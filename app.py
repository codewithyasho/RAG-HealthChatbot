"""
Streamlit Health Chatbot Application
A user-friendly web interface for the RAG-based health information assistant.
"""

import streamlit as st
from src.embedding import huggingface_embeddings
from src.vectorstore import load_vectorstore
from src.chain import create_rag_chain
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Health Information Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize theme in session state
if "theme" not in st.session_state:
    st.session_state.theme = "light"


def apply_theme_css(theme):
    """Apply custom CSS based on the selected theme."""
    if theme == "dark":
        st.markdown("""
            <style>
            /* Dark Mode Styles */
            .main {
                background-color: #1a1a1a;
                color: #e0e0e0;
            }
            .stChatMessage {
                background-color: #2d2d2d;
                border-radius: 10px;
                padding: 10px;
                margin: 5px 0;
                color: #e0e0e0;
            }
            .chat-header {
                text-align: center;
                padding: 20px;
                background: linear-gradient(135deg, #434343 0%, #2d2d2d 100%);
                color: #e0e0e0;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            .warning-box {
                background-color: #3d3d1a;
                border-left: 5px solid #ffc107;
                padding: 15px;
                border-radius: 5px;
                margin: 10px 0;
                color: #e0e0e0;
            }
            .stMarkdown, .stText, p, li, h1, h2, h3, h4, h5, h6 {
                color: #e0e0e0 !important;
            }
            [data-testid="stSidebar"] {
                background-color: #2d2d2d;
            }
            .theme-toggle {
                background-color: #434343;
                border: 2px solid #667eea;
                color: #e0e0e0;
                padding: 8px 16px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 14px;
                margin-bottom: 20px;
                width: 100%;
                text-align: center;
            }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            /* Light Mode Styles */
            .main {
                background-color: #f5f7fa;
                color: #1a1a1a;
            }
            .stChatMessage {
                background-color: white;
                border-radius: 10px;
                padding: 10px;
                margin: 5px 0;
                color: #1a1a1a;
            }
            .chat-header {
                text-align: center;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            .warning-box {
                background-color: #fff3cd;
                border-left: 5px solid #ffc107;
                padding: 15px;
                border-radius: 5px;
                margin: 10px 0;
                color: #1a1a1a;
            }
            [data-testid="stSidebar"] {
                background-color: #ffffff;
            }
            .theme-toggle {
                background-color: #667eea;
                border: 2px solid #764ba2;
                color: white;
                padding: 8px 16px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 14px;
                margin-bottom: 20px;
                width: 100%;
                text-align: center;
            }
            </style>
        """, unsafe_allow_html=True)


# Apply the current theme
apply_theme_css(st.session_state.theme)


@st.cache_resource
def initialize_chatbot():
    """Initialize and cache the RAG chain to avoid reloading on every interaction."""
    try:
        with st.spinner("🔄 Loading embeddings and vectorstore..."):
            vectorstore_path = "faiss_index"

            # Check if vectorstore exists
            if not os.path.exists(os.path.join(vectorstore_path, "index.faiss")):
                return None, "FAISS index not found. Please create the vectorstore first."

            embeddings = huggingface_embeddings()

            try:
                vectorstore = load_vectorstore(
                    embeddings=embeddings,
                    vectorstore_path=vectorstore_path
                )
            except Exception as load_error:
                error_msg = str(load_error)
                if "not recognized" in error_msg or "struct faiss" in error_msg:
                    return None, "CORRUPTED_INDEX"
                raise load_error

            chain = create_rag_chain(vectorstore=vectorstore)
        return chain, None
    except Exception as e:
        return None, str(e)


def display_header():
    """Display the app header."""
    st.markdown("""
        <div class="chat-header">
            <h1>🏥 Health Information Chatbot</h1>
            <p>Get evidence-based medical information instantly</p>
        </div>
    """, unsafe_allow_html=True)


def display_disclaimer():
    """Display medical disclaimer."""
    with st.sidebar:
        st.markdown("""
            <div class="warning-box">
                <h3>⚠️ Medical Disclaimer</h3>
                <p><strong>This chatbot provides educational information only.</strong></p>
                <ul>
                    <li>Not a substitute for professional medical advice</li>
                    <li>Always consult healthcare providers</li>
                    <li>For emergencies, call your local emergency number</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)


def display_theme_toggle():
    """Display theme toggle button at the top of sidebar."""
    with st.sidebar:
        # Theme toggle button
        current_theme = st.session_state.theme
        button_label = "🌙 Switch to Dark Mode" if current_theme == "light" else "☀️ Switch to Light Mode"
        
        if st.button(button_label, use_container_width=True, key="theme_toggle"):
            # Toggle theme
            st.session_state.theme = "dark" if current_theme == "light" else "light"
            st.rerun()
        
        st.markdown("---")


def display_sidebar_info():
    """Display information in the sidebar."""
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This AI-powered health chatbot uses:
        - **RAG (Retrieval Augmented Generation)**
        - **FAISS Vector Database**
        - **HuggingFace Embeddings**
        - **Groq LLM**
        """)

        st.header("📋 How to Use")
        st.write("""
        1. Type your health-related question
        2. Click Send or press Enter
        3. Review the information provided
        4. Always consult a doctor for diagnosis
        """)

        st.header("💡 Example Questions")
        st.write("""
        - What are the symptoms of diabetes?
        - How to prevent heart disease?
        - What causes high blood pressure?
        - Treatment options for migraine
        """)

        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def main():
    """Main application function."""

    # Display header
    display_header()

    # Display theme toggle at the top of sidebar
    display_theme_toggle()

    # Display sidebar info
    display_sidebar_info()

    # Display disclaimer
    display_disclaimer()

    # Initialize chatbot
    chain, error = initialize_chatbot()

    if error:
        if error == "CORRUPTED_INDEX":
            st.error("❌ **FAISS Index is Corrupted or Incompatible**")
            st.warning("""
            **Possible causes:**
            - FAISS version mismatch
            - Index created with different FAISS version
            - Corrupted index file
            
            **Solution:**
            You need to recreate the FAISS index. Please:
            1. Delete the `faiss_index` folder
            2. Run your data ingestion script to rebuild the index
            3. Restart this app
            
            **OR** if you don't have a data ingestion script, create one to:
            - Load your medical documents
            - Create embeddings
            - Build a new FAISS index
            """)

            with st.expander("🔧 Quick Fix: Sample Index Creation Script"):
                st.code("""
# sample_create_index.py
from src.embedding import huggingface_embeddings
from src.vectorstore import create_vectorstore
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Load your documents
loader = TextLoader("your_medical_data.txt")
documents = loader.load()

# 2. Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

# 3. Create embeddings and vectorstore
embeddings = huggingface_embeddings()
vectorstore = create_vectorstore(chunks, embeddings)

print("✅ FAISS index created successfully!")
                """, language="python")
            st.stop()
        else:
            st.error(f"❌ Error initializing chatbot: {error}")
            st.info(
                "Please check your configuration and ensure all dependencies are installed.")
            st.stop()

    if chain is None:
        st.error("❌ Failed to initialize chatbot. Please check your configuration.")
        st.stop()    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask your health question here..."):

        # Validate input
        if not prompt.strip():
            st.warning("⚠️ Please enter a valid question.")
            return

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching medical information..."):
                try:
                    response = chain.invoke({"input": prompt})
                    answer = response.get(
                        "answer", "I couldn't generate a response.")

                    # Display the answer
                    st.markdown(answer)

                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:
                    error_message = f"❌ Error generating response: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message
                    })


if __name__ == "__main__":
    main()
