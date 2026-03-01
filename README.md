# RAG Implementation

A Retrieval-Augmented Generation (RAG) system using LangChain, Chroma, and Ollama to build intelligent document-based Q&A applications.

## Project Structure

```
RAG Integration/
├── rag.py                 # Main RAG implementation script
├── app/                   # Application-specific code (Flask/Streamlit apps)
├── data/                  # PDF and document files for processing
├── scripts/               # Utility scripts for data processing
├── document/              # Documentation and guides
├── venv/                  # Python virtual environment
├── .gitignore             # Git ignore rules
├── .gitattributes         # Git attributes configuration
├── LICENSE                # Project license
└── README.md              # This file
```

## Key Components

### Main Script: `rag.py`
Implements a complete RAG pipeline:
- **Document Loading**: Reads PDF files using PyPDFLoader
- **Text Splitting**: Splits documents into manageable chunks (800 token chunks with 150 token overlap)
- **Embeddings**: Uses HuggingFace's MiniLM model for text embeddings
- **Vector Store**: Stores embeddings in Chroma with persistence
- **LLM Integration**: Uses Ollama's llama3 model for response generation
- **Retrieval QA**: Implements RetrievalQA chain for intelligent Q&A

## Setup Instructions

### 1. Create Virtual Environment
```bash
cd "RAG Integration"
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or: venv\Scripts\activate  # On Windows
```

### 2. Install Dependencies
```bash
pip install langchain langchain-community langchain-text-splitters
pip install chromadb
pip install sentence-transformers
pip install ollama
```

### 3. Prepare Data
- Place your PDF files in the `data/` folder
- Update the PDF path in `rag.py` if needed

### 4. Run the Application
```bash
python rag.py
```

## Dependencies

- **LangChain**: Framework for building LLM applications
- **LangChain Community**: Community integrations for loaders, embeddings, and LLMs
- **Chroma**: Vector database for storing and retrieving embeddings
- **Sentence Transformers**: For generating text embeddings
- **Ollama**: Local LLM runtime (requires Ollama installation)

## Configuration

### Embeddings Model
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- Lightweight, efficient model suitable for CPU inference

### Chunking Strategy
- **Chunk Size**: 800 tokens
- **Chunk Overlap**: 150 tokens
- **Purpose**: Preserve context across chunk boundaries

### Retriever Settings
- **Search Results**: Returns top 4 most relevant documents (k=4)

### LLM
- **Model**: Ollama's llama3
- **Type**: Local inference (no API required)

## Folder Purposes

| Folder | Purpose |
|--------|---------|
| `app/` | Streamlit/Flask web application files |
| `data/` | Input PDF documents and datasets |
| `scripts/` | Helper scripts for preprocessing, evaluation, etc. |
| `document/` | Project documentation, guides, and examples |
| `venv/` | Python virtual environment (not committed to git) |

## Usage Example

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# Load and process documents
loader = PyPDFLoader("data/sample.pdf")
documents = loader.load()

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents, embeddings)

# Create QA chain
llm = Ollama(model="llama3")
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Ask a question
response = qa.invoke({"query": "Your question here"})
print(response["result"])
```

## Requirements

- Python 3.8+
- Ollama installed and running (https://ollama.ai)
- 8GB+ RAM recommended
- GPU optional (for faster embeddings)

## Next Steps

- [ ] Add your PDF documents to `data/` folder
- [ ] Test with `rag.py` script
- [ ] Build web UI in `app/` folder (Streamlit/Flask)
- [ ] Add evaluation scripts in `scripts/` folder
- [ ] Document your findings in `document/` folder

## Git Repository

GitHub: [RAG-Implementation](https://github.com/keshavkumar143/RAG-Implementation.git)

## License

See LICENSE file for details.

---

**Last Updated**: March 1, 2026
