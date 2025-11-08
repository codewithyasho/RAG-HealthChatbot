# 🏥 Health Information Chatbot

A RAG (Retrieval Augmented Generation) based health information assistant built with LangChain, FAISS, and Streamlit. This chatbot provides evidence-based medical information in a user-friendly conversational interface.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-🦜-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## ✨ Features

- 🤖 **RAG-Powered Responses** - Uses retrieval augmented generation for accurate medical information
- 💬 **Interactive Chat Interface** - Clean Streamlit-based UI with chat history
- 🌓 **Dark/Light Mode** - Toggle between themes for comfortable viewing
- 🔍 **Semantic Search** - FAISS vector database for efficient document retrieval
- 🎯 **Structured Output** - Medical information formatted with causes, symptoms, treatments, medicines, and prevention
- ⚠️ **Safety Warnings** - Built-in medical disclaimers and uncertainty handling
- 🚀 **Fast Performance** - Cached embeddings and optimized retrieval

## 🏗️ Architecture

```
┌─────────────┐
│   User      │
│   Query     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────┐
│   Streamlit Interface       │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│   RAG Chain (LangChain)     │
│   - Groq LLM                │
│   - Custom Prompt Template  │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│   FAISS Vector Store        │
│   - HuggingFace Embeddings  │
│   - Semantic Search         │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│   Medical Knowledge Base    │
└─────────────────────────────┘
```

## 📋 Prerequisites

- Python 3.11 or higher
- Groq API key ([Get one here](https://console.groq.com/))
- Git (optional)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/codewithyasho/RAG-HealthChatbot.git
cd RAG-HealthChatbot
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Build FAISS Index

Run the rebuild script to create the vector database:

```bash
python rebuild_index.py
```

This will create a `faiss_index` folder with sample medical data.

## 🎯 Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Running CLI Version

```bash
python main.py
```

## 📁 Project Structure

```
project15-health-chatbot/
│
├── app.py                  # Streamlit web interface
├── main.py                 # CLI interface
├── rebuild_index.py        # Script to rebuild FAISS index
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create this)
├── README.md              # Project documentation
│
├── faiss_index/           # Vector database (generated)
│   └── index.faiss
│
└── src/                   # Source modules
    ├── chain.py           # RAG chain setup
    ├── embedding.py       # Embedding models
    ├── prompt.py          # Prompt templates
    └── vectorstore.py     # FAISS operations
```

## 🔧 Configuration

### Customize Embedding Model

Edit `src/embedding.py`:

```python
def huggingface_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    # Change model_name to your preferred HuggingFace model
```

### Customize LLM Settings

Edit `src/chain.py`:

```python
llm = ChatGroq(
    model="openai/gpt-oss-120b",  # Change model
    temperature=0.2,               # Adjust temperature
)
```

### Modify Prompt Template

Edit `src/prompt.py` to customize the medical information format and response style.

## 📊 Adding Your Own Medical Data

1. **Prepare your documents** (PDF, TXT, or other formats)
2. **Modify `rebuild_index.py`**:

```python
# Load your documents
from langchain_community.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader(
    "path/to/your/medical/data/",
    glob="**/*.txt",
    loader_cls=TextLoader
)
documents = loader.load()
```

3. **Run the rebuild script**:

```bash
python rebuild_index.py
```

## 💡 Example Questions

- What are the symptoms of diabetes?
- How to prevent heart disease?
- What causes high blood pressure?
- Treatment options for migraine
- What medicines are used for asthma?

## ⚠️ Medical Disclaimer

**IMPORTANT:** This chatbot provides educational information only and is NOT a substitute for professional medical advice, diagnosis, or treatment. 

- Always consult qualified healthcare providers for medical concerns
- Never disregard professional medical advice
- For emergencies, call your local emergency number immediately

## 🛠️ Technologies Used

- **[LangChain](https://python.langchain.com/)** - Framework for LLM applications
- **[Streamlit](https://streamlit.io/)** - Web interface
- **[FAISS](https://faiss.ai/)** - Vector similarity search
- **[HuggingFace](https://huggingface.co/)** - Embedding models
- **[Groq](https://groq.com/)** - Fast LLM inference
- **Python 3.11+** - Programming language

## 🐛 Troubleshooting

### FAISS Index Corrupted

```bash
# Delete old index and rebuild
rm -rf faiss_index  # Linux/Mac
# or
rmdir /s faiss_index  # Windows

python rebuild_index.py
```

### Import Errors

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### API Key Issues

- Ensure `.env` file exists in project root
- Verify GROQ_API_KEY is correctly set
- Check API key validity at [Groq Console](https://console.groq.com/)

## 📈 Future Enhancements

- [ ] Add support for multiple document formats (PDF, DOCX, CSV)
- [ ] Implement conversation memory for context-aware responses
- [ ] Add source citations for retrieved information
- [ ] Multi-language support
- [ ] Voice input/output capabilities
- [ ] Export chat history
- [ ] Admin dashboard for monitoring usage
- [ ] Integration with medical databases/APIs

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Yasho**
- GitHub: [@codewithyasho](https://github.com/codewithyasho)
- Repository: [RAG-HealthChatbot](https://github.com/codewithyasho/RAG-HealthChatbot)

## 🙏 Acknowledgments

- LangChain community for excellent documentation
- HuggingFace for open-source embedding models
- Groq for fast LLM inference
- Streamlit for the amazing UI framework

## 📞 Support

If you have any questions or issues, please:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Open an [Issue](https://github.com/codewithyasho/RAG-HealthChatbot/issues)
3. Contact via GitHub

---

⭐ **If you find this project helpful, please consider giving it a star!** ⭐
