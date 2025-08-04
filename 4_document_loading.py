"""
LangChain Document Loading and Processing
This file demonstrates LangChain's document loading and processing features:
- Loading text files
- Loading PDF files and web pages
- Directory-based loading
- Text splitting (Text Splitting)
- Processing different document formats
"""

import os
from dotenv import load_dotenv
from langchain.document_loaders import (
    TextLoader,
    PyPDFLoader,
    WebBaseLoader,
    DirectoryLoader
)
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter
)
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

load_dotenv()

def create_sample_documents():
    """
    Create sample documents for testing
    """
    print("=== CREATING SAMPLE DOCUMENTS ===")
    
    # Create sample text file
    sample_text = """
Python Programming Language

Python is a high-level, general-purpose programming language developed by 
Guido van Rossum in 1991. Python's design philosophy emphasizes code 
readability and is particularly known for its meaningful use of whitespace.

Python Features:
- Easy to learn: Python's syntax is simple and understandable
- Cross-platform: Works on Windows, macOS, Linux
- Extensive library support: Rich standard library
- Open source: Free and source code is open
- Object-oriented: Supports OOP paradigm

Python Use Cases:
Web development, data analysis, artificial intelligence, machine learning, 
automation, scientific computing, and many other fields.

Python's popularity has grown significantly in recent years, especially 
with its use in data science and artificial intelligence fields.
"""
    
    with open("sample_text.txt", "w", encoding="utf-8") as f:
        f.write(sample_text)
    
    # Create sample HTML file
    sample_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Web Development</title>
</head>
<body>
    <h1>Modern Web Development</h1>
    <p>Web development is the process of creating websites and web applications.</p>
    
    <h2>Frontend Technologies</h2>
    <ul>
        <li>HTML - Structure</li>
        <li>CSS - Styling</li>
        <li>JavaScript - Interaction</li>
        <li>React, Vue, Angular - Frameworks</li>
    </ul>
    
    <h2>Backend Technologies</h2>
    <ul>
        <li>Node.js</li>
        <li>Python (Django, Flask)</li>
        <li>Java (Spring)</li>
        <li>PHP</li>
    </ul>
    
    <p>Full-stack developers use both frontend and backend technologies.</p>
</body>
</html>
"""
    
    with open("sample_web.html", "w", encoding="utf-8") as f:
        f.write(sample_html)
    
    print("✅ Sample files created: sample_text.txt, sample_web.html")

def text_loader_example():
    """
    Text file loading example
    """
    print("\n=== TEXT FILE LOADING ===")
    
    # Create text loader
    loader = TextLoader("sample_text.txt", encoding="utf-8")
    
    # Load document
    documents = loader.load()
    
    print(f"Number of loaded documents: {len(documents)}")
    print(f"First document content (first 200 characters):")
    print(f"{documents[0].page_content[:200]}...")
    print(f"Document metadata: {documents[0].metadata}")

def directory_loader_example():
    """
    Directory loading example - load all files in directory
    """
    print("\n=== DIRECTORY LOADER EXAMPLE ===")
    
    # Load all .txt files in directory
    loader = DirectoryLoader(".", glob="*.txt")
    documents = loader.load()
    
    print(f"Total documents in directory: {len(documents)}")
    for i, doc in enumerate(documents):
        print(f"Document {i+1}: {doc.metadata['source']}")
        print(f"Content length: {len(doc.page_content)} characters")

def web_loader_example():
    """
    Web page loading example
    """
    print("\n=== WEB PAGE LOADING ===")
    
    try:
        # Load web page
        loader = WebBaseLoader("https://python.org")
        documents = loader.load()
        
        print(f"Web page loaded: {len(documents)} document")
        print(f"Content length: {len(documents[0].page_content)} characters")
        print(f"First 300 characters:")
        print(f"{documents[0].page_content[:300]}...")
        
    except Exception as e:
        print(f"Web loading error (internet connection required): {e}")
        
        # Use local HTML file as alternative
        print("Using local HTML file...")
        with open("sample_web.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        
        # Create manual document
        doc = Document(
            page_content=html_content,
            metadata={"source": "sample_web.html"}
        )
        print(f"Local HTML document loaded: {len(doc.page_content)} characters")

def character_text_splitter_example():
    """
    Character-based text splitting example
    """
    print("\n=== CHARACTER-BASED TEXT SPLITTING ===")
    
    # Load sample document
    loader = TextLoader("sample_text.txt", encoding="utf-8")
    documents = loader.load()
    
    # Create character text splitter
    text_splitter = CharacterTextSplitter(
        chunk_size=200,      # Maximum 200 characters per chunk
        chunk_overlap=50,    # 50 character overlap between chunks
        length_function=len, # Length calculation function
        separator="\n\n"     # Paragraph separator
    )
    
    # Split document
    split_docs = text_splitter.split_documents(documents)
    
    print(f"Original document count: {len(documents)}")
    print(f"Split chunk count: {len(split_docs)}")
    
    # Show each chunk
    for i, doc in enumerate(split_docs):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Length: {len(doc.page_content)} characters")
        print(f"Content: {doc.page_content[:100]}...")

def recursive_text_splitter_example():
    """
    Recursive text splitting example
    Performs smarter splitting
    """
    print("\n=== RECURSIVE TEXT SPLITTING ===")
    
    # Long text example
    long_text = """
    Artificial Intelligence and Machine Learning

    Artificial intelligence (AI) is the simulation of human intelligence processes by machines.
    
    Machine Learning Types:
    1. Supervised Learning
    2. Unsupervised Learning  
    3. Reinforcement Learning
    
    Deep Learning uses artificial neural networks to solve complex problems.
    
    Current AI application areas:
    - Natural language processing
    - Computer vision
    - Autonomous vehicles
    - Medicine and healthcare
    - Finance
    """
    
    # Create recursive text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=30,
        separators=["\n\n", "\n", " ", ""]  # Separators in priority order
    )
    
    # Split text
    chunks = text_splitter.split_text(long_text)
    
    print(f"Total chunk count: {len(chunks)}")
    
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Length: {len(chunk)} characters")
        print(f"Content: {chunk.strip()}")

def token_text_splitter_example():
    """
    Token-based text splitting example
    Considers LLM token limits
    """
    print("\n=== TOKEN-BASED TEXT SPLITTING ===")
    
    # Load sample document
    loader = TextLoader("sample_text.txt", encoding="utf-8")
    documents = loader.load()
    
    # Create token text splitter
    text_splitter = TokenTextSplitter(
        chunk_size=100,      # Maximum 100 tokens per chunk
        chunk_overlap=20     # 20 token overlap between chunks
    )
    
    # Split documents
    split_docs = text_splitter.split_documents(documents)
    
    print(f"Token-based split chunk count: {len(split_docs)}")
    
    for i, doc in enumerate(split_docs):
        print(f"\n--- Token Chunk {i+1} ---")
        # Estimate token count (approximate)
        estimated_tokens = len(doc.page_content.split()) * 1.3
        print(f"Estimated token count: {estimated_tokens:.0f}")
        print(f"Content: {doc.page_content[:150]}...")

def document_summarization_example():
    """
    Document summarization example
    Summarizes loaded documents
    """
    print("\n=== DOCUMENT SUMMARIZATION ===")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    
    # Load and split document
    loader = TextLoader("sample_text.txt", encoding="utf-8")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    split_docs = text_splitter.split_documents(documents)
    
    # Load summarization chain
    summarize_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",  # map_reduce for long documents
        verbose=True
    )
    
    try:
        # Summarize document
        summary = summarize_chain.run(split_docs)
        
        print(f"Original document length: {sum(len(doc.page_content) for doc in split_docs)} characters")
        print(f"Summary length: {len(summary)} characters")
        print(f"\nSummary:")
        print(summary)
        
    except Exception as e:
        print(f"Summarization error: {e}")

def cleanup_files():
    """
    Clean up test files
    """
    try:
        os.remove("sample_text.txt")
        os.remove("sample_web.html")
        print("\n✅ Test files cleaned up")
    except:
        pass

def main():
    """
    Main function - run all document loading examples
    """
    print("LangChain Document Loading Examples Starting...\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not found!")
        return
    
    try:
        # Create sample files
        create_sample_documents()
        
        # Document loading examples
        text_loader_example()
        directory_loader_example()
        web_loader_example()
        
        # Text splitting examples
        character_text_splitter_example()
        recursive_text_splitter_example()
        token_text_splitter_example()
        
        # Document processing example
        document_summarization_example()
        
        print("\n✅ All document loading examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
    
    finally:
        # Clean up files
        cleanup_files()

if __name__ == "__main__":
    main()