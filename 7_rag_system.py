"""
LangChain RAG (Retrieval Augmented Generation) System
This file comprehensively demonstrates LangChain's RAG features:
- Document loading and preprocessing
- Vector store creation
- Retrieval-based QA
- Advanced RAG techniques
- Multi-document RAG
- Conversational RAG
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import tempfile
import shutil

load_dotenv()

def create_comprehensive_knowledge_base():
    """
    Create comprehensive knowledge base
    Detailed documents on various topics
    """
    print("=== CREATING COMPREHENSIVE KNOWLEDGE BASE ===")
    
    # Detailed documents on various technology topics
    knowledge_docs = [
        {
            "title": "Python Web Frameworks",
            "content": """
            Python Web Frameworks
            
            Django:
            Django is a free and open-source web framework based on Python. It uses the Model-View-Template (MVT) architecture.
            
            Django Features:
            - ORM (Object-Relational Mapping) system
            - Automatic admin panel generation
            - URL routing system
            - Template system
            - Security features (CSRF, XSS protection)
            - Internationalization support
            
            Django Use Cases:
            - E-commerce websites
            - Content management systems
            - Social media platforms
            - Enterprise web applications
            
            Flask:
            Flask is a minimalist web framework for Python. It's known as a micro-framework.
            
            Flask Features:
            - Simple and easy to learn
            - Flexible structure
            - Uses Werkzeug WSGI toolkit
            - Jinja2 template engine
            - Blueprint support
            - Extension system
            
            Flask vs Django:
            - Django is more comprehensive, Flask is more flexible
            - Django is ideal for large projects, Flask for small-medium projects
            - Django offers more built-in features
            """,
            "category": "web_development",
            "language": "python",
            "difficulty": "intermediate"
        },
        {
            "title": "Machine Learning Algorithms",
            "content": """
            Machine Learning Algorithms
            
            Supervised Learning:
            
            1. Linear Regression:
            - Used to predict continuous values
            - Looks for linear relationships between dependent and independent variables
            - Evaluated with Mean Squared Error (MSE)
            - Not prone to overfitting
            
            2. Logistic Regression:
            - Used for binary and multiclass classification
            - Uses sigmoid function
            - Returns probability values
            - Ideal for linearly separable data
            
            3. Decision Trees:
            - Used for both regression and classification
            - Easy to interpret
            - Provides feature importance
            - Prone to overfitting
            
            4. Random Forest:
            - Ensemble of multiple decision trees
            - Reduces overfitting
            - Provides feature importance
            - Gives robust and accurate results
            
            5. Support Vector Machines (SVM):
            - Used for classification and regression
            - Can solve non-linear problems with kernel trick
            - Good for high-dimensional data
            - Sensitive to outliers
            
            Unsupervised Learning:
            
            1. K-Means Clustering:
            - Divides data into K clusters
            - Centroid-based clustering
            - Good for spherical clusters
            - K value must be predetermined
            
            2. Hierarchical Clustering:
            - Creates tree-like cluster structure
            - Has agglomerative and divisive types
            - Can be visualized with dendrograms
            - K value is not predetermined
            
            3. DBSCAN:
            - Density-based clustering
            - Can find arbitrarily shaped clusters
            - Can perform outlier detection
            - Robust to noise
            """,
            "category": "machine_learning",
            "language": "general",
            "difficulty": "advanced"
        },
        {
            "title": "React.js and Modern Frontend",
            "content": """
            React.js and Modern Frontend Development
            
            React.js Fundamentals:
            React is a JavaScript library developed by Facebook. It provides UI development with a component-based structure.
            
            React Core Concepts:
            
            1. Components:
            - Functional components (modern approach)
            - Class components (legacy)
            - Data transfer with props
            - State management
            
            2. JSX (JavaScript Extension):
            - HTML-like syntax
            - JavaScript expressions usage
            - Component rendering
            
            3. Virtual DOM:
            - Memory representation of real DOM
            - Provides performance optimization
            - Updates through diffing algorithm
            
            4. Hooks:
            - useState: state management
            - useEffect: side effects
            - useContext: context API
            - useReducer: complex state logic
            - Custom hooks: reusable logic
            
            React Ecosystem:
            
            1. State Management:
            - Redux: predictable state container
            - MobX: reactive state management
            - Zustand: lightweight alternative
            - Context API: built-in solution
            
            2. Routing:
            - React Router: declarative routing
            - Next.js Router: file-based routing
            - Reach Router: merged with React Router
            
            3. Styling:
            - CSS Modules: scoped CSS
            - Styled-components: CSS-in-JS
            - Emotion: CSS-in-JS library
            - Tailwind CSS: utility-first CSS
            
            4. Testing:
            - Jest: JavaScript testing framework
            - React Testing Library: testing utilities
            - Enzyme: JavaScript testing utility
            
            Modern Frontend Trends:
            - JAMstack architecture
            - Server-side rendering (SSR)
            - Static site generation (SSG)
            - Progressive Web Apps (PWA)
            - Micro-frontends
            - TypeScript adoption
            """,
            "category": "web_development",
            "language": "javascript",
            "difficulty": "intermediate"
        },
        {
            "title": "Database Management and SQL",
            "content": """
            Database Management and SQL
            
            Relational Databases:
            
            Basic Concepts:
            - Table: Data storage structure
            - Row: Single record
            - Column: Data field
            - Primary Key: Unique identifier
            - Foreign Key: Relationship key
            - Index: Performance enhancing structure
            
            SQL Commands:
            
            1. DDL (Data Definition Language):
            - CREATE: Create table/database
            - ALTER: Modify structure
            - DROP: Delete
            - TRUNCATE: Clear data
            
            2. DML (Data Manipulation Language):
            - SELECT: Query data
            - INSERT: Add data
            - UPDATE: Modify data
            - DELETE: Remove data
            
            3. DCL (Data Control Language):
            - GRANT: Give permissions
            - REVOKE: Remove permissions
            
            Advanced SQL Topics:
            
            1. Joins:
            - INNER JOIN: Matching records
            - LEFT JOIN: All records from left table
            - RIGHT JOIN: All records from right table
            - FULL OUTER JOIN: All records
            
            2. Aggregate Functions:
            - COUNT(): Counting
            - SUM(): Addition
            - AVG(): Average
            - MIN()/MAX(): Minimum/Maximum
            - GROUP BY: Grouping
            - HAVING: Filtering for groups
            
            3. Window Functions:
            - ROW_NUMBER(): Row number
            - RANK()/DENSE_RANK(): Ranking
            - LAG()/LEAD(): Previous/next value
            - PARTITION BY: Partitioning
            
            NoSQL Databases:
            
            1. Document Databases:
            - MongoDB: JSON-like documents
            - CouchDB: HTTP-based API
            - Amazon DocumentDB
            
            2. Key-Value Stores:
            - Redis: In-memory data structure
            - Amazon DynamoDB
            - Apache Cassandra
            
            3. Graph Databases:
            - Neo4j: Native graph database
            - Amazon Neptune
            - ArangoDB: Multi-model
            
            Database Design:
            - Normalization: Reduce data redundancy
            - Denormalization: Performance optimization
            - Entity-Relationship Diagrams
            - ACID properties: Atomicity, Consistency, Isolation, Durability
            """,
            "category": "database",
            "language": "sql",
            "difficulty": "intermediate"
        },
        {
            "title": "Cloud Computing and DevOps",
            "content": """
            Cloud Computing and DevOps
            
            Cloud Computing Models:
            
            1. Service Models:
            - IaaS (Infrastructure as a Service): AWS EC2, Google Compute Engine
            - PaaS (Platform as a Service): Heroku, Google App Engine
            - SaaS (Software as a Service): Gmail, Salesforce
            
            2. Deployment Models:
            - Public Cloud: Amazon AWS, Microsoft Azure, Google Cloud
            - Private Cloud: On-premises, dedicated resources
            - Hybrid Cloud: Public + Private combination
            - Multi-cloud: Multiple cloud providers
            
            Major Cloud Providers:
            
            1. Amazon Web Services (AWS):
            - EC2: Elastic Compute Cloud
            - S3: Simple Storage Service
            - RDS: Relational Database Service
            - Lambda: Serverless computing
            - CloudFormation: Infrastructure as Code
            
            2. Microsoft Azure:
            - Virtual Machines
            - Blob Storage
            - SQL Database
            - Azure Functions
            - ARM Templates
            
            3. Google Cloud Platform (GCP):
            - Compute Engine
            - Cloud Storage
            - Cloud SQL
            - Cloud Functions
            - Deployment Manager
            
            DevOps Practices:
            
            1. Continuous Integration/Continuous Deployment (CI/CD):
            - Version Control: Git, GitHub, GitLab
            - Build Automation: Jenkins, GitHub Actions, GitLab CI
            - Testing Automation: Unit, Integration, E2E tests
            - Deployment Automation: Blue-green, Rolling deployments
            
            2. Infrastructure as Code (IaC):
            - Terraform: Multi-cloud provisioning
            - Ansible: Configuration management
            - Puppet: Infrastructure automation
            - Chef: Infrastructure automation
            
            3. Containerization:
            - Docker: Container platform
            - Kubernetes: Container orchestration
            - Docker Compose: Multi-container applications
            - Container registries: Docker Hub, ECR, GCR
            
            4. Monitoring and Logging:
            - Prometheus: Monitoring system
            - Grafana: Metrics visualization
            - ELK Stack (Elasticsearch, Logstash, Kibana)
            - APM tools: New Relic, Datadog
            
            DevOps Culture:
            - Collaboration between Dev and Ops
            - Automation of repetitive tasks
            - Continuous improvement
            - Fail fast, recover quickly
            - Infrastructure reliability
            - Security integration (DevSecOps)
            """,
            "category": "devops",
            "language": "general",
            "difficulty": "advanced"
        }
    ]
    
    # Create Document objects
    documents = []
    for doc_data in knowledge_docs:
        doc = Document(
            page_content=f"# {doc_data['title']}\n\n{doc_data['content']}",
            metadata={
                "title": doc_data["title"],
                "category": doc_data["category"],
                "language": doc_data["language"],
                "difficulty": doc_data["difficulty"]
            }
        )
        documents.append(doc)
    
    print(f"✅ {len(documents)} comprehensive documents created")
    return documents

def basic_rag_example():
    """
    Basic RAG system example
    """
    print("\n=== BASIC RAG SYSTEM ===")
    
    # Create documents and LLM
    documents = create_comprehensive_knowledge_base()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    embeddings = OpenAIEmbeddings()
    
    # Split documents with text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    split_docs = text_splitter.split_documents(documents)
    
    print(f"Total document chunks: {len(split_docs)}")
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings
    )
    
    # Create RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        verbose=True,
        return_source_documents=True
    )
    
    # Test questions
    rag_questions = [
        "What are the main differences between Django and Flask?",
        "What are supervised learning algorithms in machine learning?",
        "What are hooks in React and why are they used?",
        "What are the JOIN types in SQL and how are they used?",
        "What are the differences between IaaS, PaaS and SaaS in cloud computing?"
    ]
    
    for question in rag_questions:
        print(f"\n{'='*60}")
        print(f"QUESTION: {question}")
        print('='*60)
        
        try:
            result = qa_chain({"query": question})
            
            print(f"\nANSWER:")
            print(result['result'])
            
            print(f"\nSOURCES USED:")
            for i, doc in enumerate(result['source_documents'], 1):
                print(f"\nSource {i}:")
                print(f"Title: {doc.metadata.get('title', 'N/A')}")
                print(f"Category: {doc.metadata.get('category', 'N/A')}")
                print(f"Content summary: {doc.page_content[:150]}...")
                
        except Exception as e:
            print(f"Error: {e}")

def advanced_rag_with_custom_prompt():
    """
    Advanced RAG with custom prompt
    """
    print("\n=== ADVANCED RAG WITH CUSTOM PROMPT ===")
    
    # Create documents and vector store
    documents = create_comprehensive_knowledge_base()
    embeddings = OpenAIEmbeddings()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    split_docs = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(split_docs, embeddings)
    
    # Custom prompt template
    custom_prompt = PromptTemplate(
        template="""You are a technology expert. Answer the question using the context information below.
If the answer is not in the context information, say "This information is not available in the provided sources".
Always respond in English and explain technical terms.

Context Information:
{context}

Question: {question}

Detailed Answer:""",
        input_variables=["context", "question"]
    )
    
    # Create LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    
    # Create custom chain
    qa_chain = load_qa_chain(
        llm=llm,
        chain_type="stuff",
        prompt=custom_prompt,
        verbose=True
    )
    
    # Advanced questions
    advanced_questions = [
        "How is state management done in React functional components and what is the difference from class components?",
        "What is overfitting in machine learning and how is it prevented?",
        "What are window functions in SQL and what is the difference from aggregate functions?",
        "How is a CI/CD pipeline set up in DevOps and what are its benefits?"
    ]
    
    for question in advanced_questions:
        print(f"\n{'='*60}")
        print(f"ADVANCED QUESTION: {question}")
        print('='*60)
        
        try:
            # Retrieve relevant documents
            relevant_docs = vectorstore.similarity_search(question, k=4)
            
            # Answer with custom chain
            response = qa_chain.run(
                input_documents=relevant_docs,
                question=question
            )
            
            print(f"\nDETAILED ANSWER:")
            print(response)
            
        except Exception as e:
            print(f"Error: {e}")

def conversational_rag_example():
    """
    Conversational RAG system
    RAG that remembers previous conversation
    """
    print("\n=== CONVERSATIONAL RAG SYSTEM ===")
    
    # Create documents and vector store
    documents = create_comprehensive_knowledge_base()
    embeddings = OpenAIEmbeddings()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(split_docs, embeddings)
    
    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # Create LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    
    # Create conversational RAG chain
    conv_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        verbose=True,
        return_source_documents=True
    )
    
    # Conversation flow simulation
    conversation_flow = [
        "Which frameworks do you recommend for Python web development?",
        "Which of these is more suitable for beginners?",
        "How do you create a simple web application with Flask?",
        "What should I pay attention to if I want to switch to Django?",
        "What were the advantages of Flask you mentioned earlier?",
        "How is machine learning used in web development?"
    ]
    
    for i, question in enumerate(conversation_flow, 1):
        print(f"\n{'='*50}")
        print(f"CONVERSATION STEP {i}: {question}")
        print('='*50)
        
        try:
            # Ask question and get answer
            result = conv_chain({"question": question})
            
            print(f"\nANSWER:")
            print(result['answer'])
            
            print(f"\nCONVERSATION HISTORY:")
            for msg in memory.chat_memory.messages[-4:]:  # Last 2 question-answer pairs
                if hasattr(msg, 'content'):
                    print(f"{type(msg).__name__}: {msg.content[:100]}...")
                    
        except Exception as e:
            print(f"Error: {e}")

def multi_document_rag_with_filtering():
    """
    Multi-document RAG system with filtering
    """
    print("\n=== MULTI-DOCUMENT RAG WITH FILTERING ===")
    
    # Create documents
    documents = create_comprehensive_knowledge_base()
    embeddings = OpenAIEmbeddings()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    split_docs = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(split_docs, embeddings)
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    
    # Different filters and questions
    filtered_queries = [
        {
            "filter": {"category": "web_development"},
            "question": "What technologies are used in modern web development?",
            "description": "Only web development documents"
        },
        {
            "filter": {"difficulty": "advanced"},
            "question": "What are the advanced level technologies?",
            "description": "Only advanced level documents"
        },
        {
            "filter": {"language": "python"},
            "question": "What can be done with Python?",
            "description": "Only Python related documents"
        },
        {
            "filter": {"category": "machine_learning"},
            "question": "How to choose machine learning algorithms?", 
            "description": "Only ML documents"
        }
    ]
    
    for query_config in filtered_queries:
        print(f"\n{'='*60}")
        print(f"FILTER: {query_config['description']}")
        print(f"QUESTION: {query_config['question']}")
        print('='*60)
        
        try:
            # Get filtered documents
            filtered_docs = vectorstore.similarity_search(
                query_config['question'],
                k=4,
                filter=query_config['filter']
            )
            
            print(f"Number of filtered documents found: {len(filtered_docs)}")
            
            if filtered_docs:
                # Answer with QA chain
                qa_chain = load_qa_chain(llm, chain_type="stuff")
                response = qa_chain.run(
                    input_documents=filtered_docs,
                    question=query_config['question']
                )
                
                print(f"\nFILTERED ANSWER:")
                print(response)
                
                print(f"\nDOCUMENTS USED:")
                for i, doc in enumerate(filtered_docs, 1):
                    print(f"{i}. {doc.metadata.get('title', 'N/A')} "
                          f"({doc.metadata.get('category', 'N/A')})")
            else:
                print("No documents found matching this filter.")
                
        except Exception as e:
            print(f"Error: {e}")

def rag_with_source_citation():
    """
    RAG system with source citation
    """
    print("\n=== RAG WITH SOURCE CITATION ===")
    
    # Documents and vector store
    documents = create_comprehensive_knowledge_base()
    embeddings = OpenAIEmbeddings()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(split_docs, embeddings)
    
    # Custom prompt for source citation
    citation_prompt = PromptTemplate(
        template="""Answer the question using the context information below.
Don't forget to cite sources for each piece of information you use in your answer.
Source format: [Document Title]

Context Information:
{context}

Question: {question}

Answer with source citations:""",
        input_variables=["context", "question"]
    )
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=citation_prompt)
    
    # Questions requiring source citation
    citation_questions = [
        "Can you compare web frameworks?",
        "What are the use cases of machine learning algorithms?",
        "What are the cloud computing service models?"
    ]
    
    for question in citation_questions:
        print(f"\n{'='*60}")
        print(f"QUESTION WITH CITATION: {question}")
        print('='*60)
        
        try:
            relevant_docs = vectorstore.similarity_search(question, k=4)
            
            # Enrich context with source information
            enriched_context = ""
            for i, doc in enumerate(relevant_docs, 1):
                title = doc.metadata.get('title', f'Document {i}')
                enriched_context += f"\n--- {title} ---\n{doc.page_content}\n"
            
            response = qa_chain.run(
                context=enriched_context,
                question=question
            )
            
            print(f"\nANSWER WITH CITATIONS:")
            print(response)
            
            print(f"\nAVAILABLE SOURCE DOCUMENTS:")
            for i, doc in enumerate(relevant_docs, 1):
                print(f"{i}. {doc.metadata.get('title', 'N/A')}")
                
        except Exception as e:
            print(f"Error: {e}")

def main():
    """
    Main function - run all RAG examples
    """
    print("LangChain RAG (Retrieval Augmented Generation) Examples Starting...\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not found!")
        return
    
    try:
        # Run RAG examples
        basic_rag_example()
        advanced_rag_with_custom_prompt()
        conversational_rag_example()
        multi_document_rag_with_filtering()
        rag_with_source_citation()
        
        print("\n✅ All RAG examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")

if __name__ == "__main__":
    main()