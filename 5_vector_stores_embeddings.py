"""
LangChain Vector Stores and Embeddings
This file demonstrates LangChain's vector store and embedding features:
- Creating embeddings
- Using vector stores (Chroma, FAISS)
- Semantic search
- Similarity search
- Document search with vector stores
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma, FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
import tempfile
import shutil

load_dotenv()

def create_sample_knowledge_base():
    """
    Create sample knowledge base documents
    """
    print("=== CREATING SAMPLE KNOWLEDGE BASE ===")
    
    # Sample documents on various topics
    documents_data = [
        {
            "content": """
            Python Programming Fundamentals
            
            Python is an easy-to-learn and powerful programming language. 
            Variables, lists, dictionaries, and functions are Python's basic building blocks.
            
            Basic Data Types:
            - int: Integers
            - float: Decimal numbers  
            - str: Strings
            - bool: True/False values
            
            Indentation is very important in Python and defines code blocks.
            """,
            "metadata": {"topic": "python_basics", "difficulty": "beginner"}
        },
        {
            "content": """
            Web Development and Frameworks
            
            Modern web development includes both frontend and backend technologies.
            
            Frontend: HTML, CSS, JavaScript
            Backend: Python (Django, Flask), Node.js, PHP
            
            Responsive design, mobile-first approach, and performance optimization
            are critical topics in today's web development.
            
            Progressive Web Apps (PWA) and Single Page Applications (SPA) 
            are popular trends.
            """,
            "metadata": {"topic": "web_development", "difficulty": "intermediate"}
        },
        {
            "content": """
            Artificial Intelligence and Machine Learning
            
            Artificial intelligence is the simulation of human-like thinking capabilities by machines.
            
            Machine Learning Types:
            1. Supervised Learning: Learning with labeled data
            2. Unsupervised Learning: Learning with unlabeled data
            3. Reinforcement Learning: Learning through reward-penalty system
            
            Deep learning solves complex problems using artificial neural networks.
            TensorFlow, PyTorch, and Scikit-learn are popular libraries.
            """,
            "metadata": {"topic": "ai_ml", "difficulty": "advanced"}
        },
        {
            "content": """
            Data Science and Analysis
            
            Data science is the art of extracting meaningful insights from large datasets.
            
            Data Science Process:
            1. Data collection
            2. Data cleaning
            3. Exploratory data analysis (EDA)
            4. Model building
            5. Results interpretation
            
            Pandas, NumPy, Matplotlib, and Seaborn are the most commonly used tools.
            Statistics knowledge is fundamental for data scientists.
            """,
            "metadata": {"topic": "data_science", "difficulty": "intermediate"}
        },
        {
            "content": """
            Mobile Application Development
            
            Mobile applications are developed for iOS and Android platforms.
            
            Native Development:
            - iOS: Swift, Objective-C
            - Android: Java, Kotlin
            
            Cross-platform Development:
            - React Native
            - Flutter
            - Xamarin
            
            Mobile-first design, user experience (UX), and performance 
            are critical factors in mobile development.
            """,
            "metadata": {"topic": "mobile_development", "difficulty": "intermediate"}
        }
    ]
    
    # Create document objects
    documents = []
    for doc_data in documents_data:
        doc = Document(
            page_content=doc_data["content"],
            metadata=doc_data["metadata"]
        )
        documents.append(doc)
    
    print(f"✅ {len(documents)} sample documents created")
    return documents

def embedding_basics_example():
    """
    Basic usage of embeddings
    """
    print("\n=== EMBEDDING BASICS ===")
    
    # Create OpenAI embeddings
    embeddings = OpenAIEmbeddings()
    
    # Sample texts
    texts = [
        "Python is an amazing programming language",
        "Web development is very fun", 
        "Machine learning is the technology of the future",
        "JavaScript is used for frontend development"
    ]
    
    # Convert texts to embeddings
    text_embeddings = embeddings.embed_documents(texts)
    
    print(f"Total text count: {len(texts)}")
    print(f"Each embedding dimension: {len(text_embeddings[0])}")
    print(f"First embedding's first 5 values: {text_embeddings[0][:5]}")
    
    # Embedding for a single query
    query = "Python programming"
    query_embedding = embeddings.embed_query(query)
    print(f"Query embedding dimension: {len(query_embedding)}")
    print(f"Query embedding's first 5 values: {query_embedding[:5]}")

def chroma_vector_store_example():
    """
    Chroma vector store usage
    """
    print("\n=== CHROMA VECTOR STORE ===")
    
    # Get sample documents
    documents = create_sample_knowledge_base()
    
    # Create embeddings
    embeddings = OpenAIEmbeddings()
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create Chroma vector store
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=temp_dir
        )
        
        print(f"✅ Chroma vector store created: {len(documents)} documents")
        
        # Perform similarity search
        query = "I want to learn Python"
        similar_docs = vectorstore.similarity_search(query, k=2)
        
        print(f"\nQuery: '{query}'")
        print(f"Found similar documents: {len(similar_docs)}")
        
        for i, doc in enumerate(similar_docs, 1):
            print(f"\n--- Similar Document {i} ---")
            print(f"Topic: {doc.metadata.get('topic', 'N/A')}")
            print(f"Content: {doc.page_content[:200]}...")
        
        # Search with scores
        scored_docs = vectorstore.similarity_search_with_score(query, k=3)
        print(f"\n--- SEARCH WITH SCORES ---")
        for doc, score in scored_docs:
            print(f"Score: {score:.4f} | Topic: {doc.metadata.get('topic', 'N/A')}")
    
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

def faiss_vector_store_example():
    """
    FAISS vector store usage
    """
    print("\n=== FAISS VECTOR STORE ===")
    
    # Get sample documents
    documents = create_sample_knowledge_base()
    
    # Create embeddings
    embeddings = OpenAIEmbeddings()
    
    # Create FAISS vector store
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )
    
    print(f"✅ FAISS vector store created: {len(documents)} documents")
    
    # Different search types
    queries = [
        "what is web development",
        "artificial intelligence technologies", 
        "how to do data analysis",
        "I want to develop mobile apps"
    ]
    
    for query in queries:
        print(f"\n--- Query: '{query}' ---")
        
        # Find most similar document
        similar_docs = vectorstore.similarity_search(query, k=1)
        
        if similar_docs:
            doc = similar_docs[0]
            print(f"Most similar topic: {doc.metadata.get('topic', 'N/A')}")
            print(f"Difficulty: {doc.metadata.get('difficulty', 'N/A')}")
            print(f"Content summary: {doc.page_content[:150]}...")
    
    # Save vector store to file
    try:
        vectorstore.save_local("faiss_index")
        print(f"\n✅ FAISS index saved to file")
        
        # Load saved index
        loaded_vectorstore = FAISS.load_local("faiss_index", embeddings)
        test_result = loaded_vectorstore.similarity_search("Python", k=1)
        print(f"✅ Saved index loaded successfully")
        
        # Clean up files
        import glob
        for file in glob.glob("faiss_index*"):
            os.remove(file)
            
    except Exception as e:
        print(f"FAISS save/load error: {e}")

def semantic_search_example():
    """
    Semantic search example
    """
    print("\n=== SEMANTIC SEARCH EXAMPLE ===")
    
    # More detailed sample documents
    detailed_docs = [
        Document(
            page_content="You can create lists in Python using list comprehension. Example: [x*2 for x in range(10)]",
            metadata={"category": "python", "type": "tutorial", "level": "intermediate"}
        ),
        Document(
            page_content="You can develop modern web applications using React.js. It offers a component-based approach.",
            metadata={"category": "web", "type": "framework", "level": "intermediate"}
        ),
        Document(
            page_content="Machine learning algorithms automatically learn from datasets and make predictions.",
            metadata={"category": "ai", "type": "concept", "level": "advanced"}
        ),
        Document(
            page_content="SQL is the standard language for querying databases. SELECT, INSERT, UPDATE commands are basic.",
            metadata={"category": "database", "type": "language", "level": "beginner"}
        ),
        Document(
            page_content="Docker container technology allows you to easily package and distribute your applications.",
            metadata={"category": "devops", "type": "tool", "level": "intermediate"}
        )
    ]
    
    # Create vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(detailed_docs, embeddings)
    
    # Semantic queries
    semantic_queries = [
        ("list creation", "Python data structures"),
        ("frontend development", "Web interface technologies"),
        ("automatic learning", "Artificial intelligence algorithms"),
        ("data querying", "Database operations"),
        ("application deployment", "DevOps tools")
    ]
    
    for query, description in semantic_queries:
        print(f"\n--- {description} ---")
        print(f"Query: '{query}'")
        
        # Perform semantic search
        results = vectorstore.similarity_search_with_score(query, k=2)
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"\nResult {i} (Score: {score:.4f}):")
            print(f"Category: {doc.metadata.get('category', 'N/A')}")
            print(f"Type: {doc.metadata.get('type', 'N/A')}")
            print(f"Content: {doc.page_content}")

def retrieval_qa_example():
    """
    Retrieval QA (Information Retrieval + Question Answering) example
    Combining vector store with QA chain
    """
    print("\n=== RETRIEVAL QA EXAMPLE ===")
    
    # Create documents and vector store
    documents = create_sample_knowledge_base()
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embeddings)
    
    # Create LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    
    # Create Retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Documents are combined and given to LLM
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        verbose=True,
        return_source_documents=True
    )
    
    # Ask questions
    questions = [
        "What are Python's basic data types?",
        "What technologies are used in web development?",
        "What are the types of machine learning?",
        "How does the data science process work?",
        "What approaches are there for mobile app development?"
    ]
    
    for question in questions:
        print(f"\n{'='*50}")
        print(f"QUESTION: {question}")
        print('='*50)
        
        try:
            # Answer the question
            result = qa_chain({"query": question})
            
            print(f"\nANSWER: {result['result']}")
            
            # Show source documents
            print(f"\nSOURCE DOCUMENTS:")
            for i, doc in enumerate(result['source_documents'], 1):
                print(f"\nSource {i}:")
                print(f"Topic: {doc.metadata.get('topic', 'N/A')}")
                print(f"Difficulty: {doc.metadata.get('difficulty', 'N/A')}")
                print(f"Content summary: {doc.page_content[:100]}...")
                
        except Exception as e:
            print(f"Error: {e}")

def vector_store_filtering_example():
    """
    Vector store filtering example
    Metadata-based filtering
    """
    print("\n=== VECTOR STORE FILTERING ===")
    
    # Create documents
    documents = create_sample_knowledge_base()
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embeddings)
    
    # Apply different filters
    filters_and_queries = [
        {
            "filter": {"difficulty": "beginner"},
            "query": "I want to learn programming",
            "description": "Beginner level only"
        },
        {
            "filter": {"difficulty": "intermediate"},
            "query": "web technologies",
            "description": "Intermediate level topics"
        },
        {
            "filter": {"topic": "ai_ml"},
            "query": "learning algorithms",
            "description": "AI/ML topics only"
        }
    ]
    
    for filter_config in filters_and_queries:
        print(f"\n--- {filter_config['description']} ---")
        print(f"Filter: {filter_config['filter']}")
        print(f"Query: '{filter_config['query']}'")
        
        # Filtered search
        results = vectorstore.similarity_search(
            filter_config['query'],
            k=3,
            filter=filter_config['filter']
        )
        
        print(f"Found results: {len(results)}")
        for i, doc in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Topic: {doc.metadata.get('topic', 'N/A')}")
            print(f"Difficulty: {doc.metadata.get('difficulty', 'N/A')}")
            print(f"Content: {doc.page_content[:100]}...")

def main():
    """
    Main function - run all vector store and embedding examples
    """
    print("LangChain Vector Stores and Embeddings Examples Starting...\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not found!")
        return
    
    try:
        # Embedding examples
        embedding_basics_example()
        chroma_vector_store_example()
        faiss_vector_store_example()
        semantic_search_example()
        retrieval_qa_example()
        vector_store_filtering_example()
        
        print("\n✅ All vector store and embedding examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")

if __name__ == "__main__":
    main()