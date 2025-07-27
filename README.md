# LangChain Comprehensive Learning Project

This project is a comprehensive training set created to learn all the basic features of the LangChain library. Each module demonstrates a different aspect of LangChain with detailed explanations.

## 📚 Project Contents

### 1. Basic LLM Usage (`1_basic_llm.py`)
- Basic interaction with LLM models
- Different model types (GPT-3.5, GPT-4)
- Message-based chat system
- Streaming examples
- Model comparisons

### 2. Prompt Templates and Chains (`2_prompts_and_chains.py`)
- Dynamic prompt templates
- Chat prompt templates
- Few-shot prompting
- LLM Chain usage
- Sequential and Simple Sequential Chains
- Custom output parsers

### 3. Memory Management (`3_memory_management.py`)
- Conversation Buffer Memory
- Conversation Summary Memory
- Token Buffer Memory
- Window Memory
- Custom memory management techniques

### 4. Document Loading (`4_document_loading.py`)
- Text file loading
- PDF and web page loading
- Directory-based loading
- Text splitting (character, recursive, token-based)
- Document summarization

### 5. Vector Stores and Embeddings (`5_vector_stores_embeddings.py`)
- OpenAI Embeddings usage
- Chroma and FAISS vector stores
- Semantic search
- Similarity search
- Retrieval QA
- Filtering and metadata usage

### 6. Agents and Tools (`6_agents_and_tools.py`)
- Creating custom tools
- ReAct agent usage
- Conversational agent
- Multi-step problem solving
- Custom calculation tools

### 7. RAG System (`7_rag_system.py`)
- Retrieval Augmented Generation
- Comprehensive knowledge base creation
- Advanced RAG with custom prompts
- Conversational RAG
- Multi-document RAG
- RAG with source attribution

## 🚀 Installation

### 1. Requirements
```bash
# Clone the project
git clone <repo-url>
cd langchain

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\\Scripts\\activate    # Windows

# Install required libraries
pip install -r requirements.txt
```

### 2. API Key Configuration
```bash
# Copy .env.example file as .env
cp .env.example .env

# Edit .env file and add your OpenAI API key
OPENAI_API_KEY=your_openai_api_key_here
```

## 💻 Usage

### Interactive Mode (Recommended)
```bash
python main_demo.py
```
This command allows you to select and run all modules through a menu system.

### Run All Modules
```bash
python main_demo.py all
```

### Run Single Module
```bash
python main_demo.py 1    # Basic LLM
python main_demo.py 2    # Prompt Templates
python main_demo.py 3    # Memory Management
# ... etc
```

### Help
```bash
python main_demo.py help
```

### Manual Execution
You can also run each module separately:
```bash
python 1_basic_llm.py
python 2_prompts_and_chains.py
# ... etc
```

## 📖 Learning Guide

### Beginner Level
1. `1_basic_llm.py` - Get familiar with LLMs
2. `2_prompts_and_chains.py` - Learn the art of prompt writing
3. `4_document_loading.py` - Discover document processing

### Intermediate Level
4. `3_memory_management.py` - Understand memory management
5. `5_vector_stores_embeddings.py` - Explore semantic search

### Advanced Level
6. `6_agents_and_tools.py` - Create intelligent agents
7. `7_rag_system.py` - Build advanced RAG systems

## 🔍 Features

### ✅ Comprehensive Explanations
- English explanations for every line of code
- Conceptual explanations and examples
- Best practices and tips

### ✅ Practical Examples
- Real-world scenarios
- Interactive code examples
- Error handling and edge cases

### ✅ Modular Structure
- Each topic in separate modules
- Examples that can work independently
- Progressive learning structure

### ✅ Error Management
- Detailed error messages
- Solution suggestions
- Graceful error handling

## 🛠️ Technology Stack

- **LangChain**: Main framework
- **OpenAI**: LLM provider
- **Chroma**: Vector database
- **FAISS**: Alternative vector store
- **Python-dotenv**: Environment management

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- Internet connection
- 2GB+ RAM (for vector operations)

## 🔧 Troubleshooting

### API Error
```
❌ OPENAI_API_KEY environment variable not found!
```
**Solution**: Create `.env` file and add your API key.

### Import Error
```
❌ Missing libraries: langchain
```
**Solution**: Run `pip install -r requirements.txt` command.

### Memory Error
**Solution**: Use smaller chunk_size values.

## 📚 Additional Resources

- [LangChain Official Documentation](https://docs.langchain.com/)
- [OpenAI API Reference](https://platform.openai.com/docs)
- [Vector Database Guide](https://www.pinecone.io/learn/vector-database/)

## 🤝 Contributing

This project is for educational purposes. For improvements:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Send a pull request

## 📄 License

This project is published under the MIT license.

## 🙏 Acknowledgments

- To the LangChain development team
- To OpenAI for their powerful APIs
- To the open source community

---

**Happy learning! 🚀**