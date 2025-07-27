"""
LangChain RAG (Retrieval Augmented Generation) Sistemi
Bu dosya LangChain'in RAG özelliklerini kapsamlı şekilde gösterir:
- Document loading ve preprocessing
- Vector store oluşturma
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
    Kapsamlı bilgi tabanı oluştur
    Farklı konularda detaylı belgeler
    """
    print("=== KAPSAMLI BİLGİ TABANI OLUŞTURULUYOR ===")
    
    # Çeşitli teknoloji konularında detaylı belgeler
    knowledge_docs = [
        {
            "title": "Python Web Framework'leri",
            "content": """
            Python Web Framework'leri
            
            Django:
            Django, Python tabanlı ücretsiz ve açık kaynak web framework'üdür. Model-View-Template (MVT) mimarisini kullanır.
            
            Django Özellikleri:
            - ORM (Object-Relational Mapping) sistemi
            - Admin paneli otomatik oluşturma
            - URL routing sistemi
            - Template sistemi
            - Güvenlik özellikleri (CSRF, XSS koruması)
            - İnternasyonalizasyon desteği
            
            Django Kullanım Alanları:
            - E-ticaret siteleri
            - İçerik yönetim sistemleri
            - Sosyal medya platformları
            - Kurumsal web uygulamaları
            
            Flask:
            Flask, Python için minimalist web framework'üdür. Micro-framework olarak bilinir.
            
            Flask Özellikleri:
            - Basit ve öğrenmesi kolay
            - Esnek yapı
            - Werkzeug WSGI toolkit kullanır
            - Jinja2 template engine
            - Blueprint desteği
            - Extension sistemi
            
            Flask vs Django:
            - Django daha kapsamlı, Flask daha esnek
            - Django büyük projeler için, Flask küçük-orta projeler için ideal
            - Django daha fazla built-in özellik sunar
            """,
            "category": "web_development",
            "language": "python",
            "difficulty": "intermediate"
        },
        {
            "title": "Makine Öğrenmesi Algoritmaları",
            "content": """
            Makine Öğrenmesi Algoritmaları
            
            Supervised Learning (Gözetimli Öğrenme):
            
            1. Linear Regression (Doğrusal Regresyon):
            - Sürekli değerleri tahmin etmek için kullanılır
            - Bağımlı ve bağımsız değişkenler arasında doğrusal ilişki arar
            - Mean Squared Error (MSE) ile değerlendirilir
            - Overfitting'e eğilimli değildir
            
            2. Logistic Regression (Lojistik Regresyon):
            - Binary ve multiclass classification için kullanılır
            - Sigmoid fonksiyonu kullanır
            - Probability değerleri döndürür
            - Linear separable veri için idealdir
            
            3. Decision Trees (Karar Ağaçları):
            - Hem regression hem classification için kullanılır
            - Interpret edilmesi kolaydır
            - Feature importance sağlar
            - Overfitting'e eğilimlidir
            
            4. Random Forest:
            - Multiple decision tree'lerin ensemble'ı
            - Overfitting'i azaltır
            - Feature importance sağlar
            - Robust ve accurate sonuçlar verir
            
            5. Support Vector Machines (SVM):
            - Classification ve regression için kullanılır
            - Kernel trick ile non-linear problems çözebilir
            - High-dimensional data için iyidir
            - Outlier'lara duyarlıdır
            
            Unsupervised Learning (Gözetimsiz Öğrenme):
            
            1. K-Means Clustering:
            - Veriyi K adet cluster'a böler
            - Centroid tabanlı clustering
            - Spherical cluster'lar için iyidir
            - K değeri önceden belirlenmeli
            
            2. Hierarchical Clustering:
            - Tree-like cluster yapısı oluşturur
            - Agglomerative ve divisive türleri var
            - Dendrogram ile görselleştirilebilir
            - K değeri önceden belirlenmez
            
            3. DBSCAN:
            - Density-based clustering
            - Arbitrary şekilli cluster'ları bulabilir
            - Outlier detection yapabilir
            - Noise'a robust'tur
            """,
            "category": "machine_learning",
            "language": "general",
            "difficulty": "advanced"
        },
        {
            "title": "React.js ve Modern Frontend",
            "content": """
            React.js ve Modern Frontend Geliştirme
            
            React.js Temelleri:
            React, Facebook tarafından geliştirilen JavaScript library'sidir. Component-based yapısı ile UI geliştirme sağlar.
            
            React Temel Kavramları:
            
            1. Components:
            - Functional components (modern yaklaşım)
            - Class components (legacy)
            - Props ile veri aktarımı
            - State yönetimi
            
            2. JSX (JavaScript Extension):
            - HTML benzeri syntax
            - JavaScript expressions kullanımı
            - Component rendering
            
            3. Virtual DOM:
            - Real DOM'un memory'deki representation'ı
            - Performans optimizasyonu sağlar
            - Diffing algorithm ile güncellemeler
            
            4. Hooks:
            - useState: state yönetimi
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
            "title": "Veri Tabanı Yönetimi ve SQL",
            "content": """
            Veri Tabanı Yönetimi ve SQL
            
            İlişkisel Veri Tabanları:
            
            Temel Kavramlar:
            - Table (Tablo): Veri saklama yapısı
            - Row (Satır): Tek bir kayıt
            - Column (Sütun): Veri alanı
            - Primary Key: Benzersiz tanımlayıcı
            - Foreign Key: İlişki kurucu anahtar
            - Index: Performans artırıcı yapı
            
            SQL Komutları:
            
            1. DDL (Data Definition Language):
            - CREATE: Tablo/veritabanı oluşturma
            - ALTER: Yapı değiştirme
            - DROP: Silme
            - TRUNCATE: Veri temizleme
            
            2. DML (Data Manipulation Language):
            - SELECT: Veri sorgulama
            - INSERT: Veri ekleme
            - UPDATE: Veri güncelleme
            - DELETE: Veri silme
            
            3. DCL (Data Control Language):
            - GRANT: Yetki verme
            - REVOKE: Yetki alma
            
            Advanced SQL Konuları:
            
            1. Joins:
            - INNER JOIN: Eşleşen kayıtlar
            - LEFT JOIN: Sol tablodaki tüm kayıtlar
            - RIGHT JOIN: Sağ tablodaki tüm kayıtlar
            - FULL OUTER JOIN: Tüm kayıtlar
            
            2. Aggregate Functions:
            - COUNT(): Sayma
            - SUM(): Toplama
            - AVG(): Ortalama
            - MIN()/MAX(): Minimum/Maximum
            - GROUP BY: Gruplama
            - HAVING: Gruplar için filtreleme
            
            3. Window Functions:
            - ROW_NUMBER(): Satır numarası
            - RANK()/DENSE_RANK(): Sıralama
            - LAG()/LEAD(): Önceki/sonraki değer
            - PARTITION BY: Bölümleme
            
            NoSQL Veri Tabanları:
            
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
            
            Veri Tabanı Tasarımı:
            - Normalization: Veri tekrarını azaltma
            - Denormalization: Performans optimizasyonu
            - Entity-Relationship Diagrams
            - ACID properties: Atomicity, Consistency, Isolation, Durability
            """,
            "category": "database",
            "language": "sql",
            "difficulty": "intermediate"
        },
        {
            "title": "Cloud Computing ve DevOps",
            "content": """
            Cloud Computing ve DevOps
            
            Cloud Computing Modelleri:
            
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
            
            4. Monitoring ve Logging:
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
    
    # Document objelerini oluştur
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
    
    print(f"✅ {len(documents)} kapsamlı belge oluşturuldu")
    return documents

def basic_rag_example():
    """
    Temel RAG sistemi örneği
    """
    print("\n=== TEMEL RAG SİSTEMİ ===")
    
    # Belgeler ve LLM'i oluştur
    documents = create_comprehensive_knowledge_base()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    embeddings = OpenAIEmbeddings()
    
    # Text splitter ile belgeleri böl
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    split_docs = text_splitter.split_documents(documents)
    
    print(f"Toplam belge parçası: {len(split_docs)}")
    
    # Vector store oluştur
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings
    )
    
    # RAG chain oluştur
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        verbose=True,
        return_source_documents=True
    )
    
    # Test soruları
    rag_questions = [
        "Django ve Flask arasındaki temel farklar nelerdir?",
        "Makine öğrenmesinde supervised learning algoritmaları nelerdir?",
        "React'ta hooks nedir ve neden kullanılır?",
        "SQL'de JOIN türleri nelerdir ve nasıl kullanılır?",
        "Cloud computing'de IaaS, PaaS ve SaaS arasındaki farklar nedir?"
    ]
    
    for question in rag_questions:
        print(f"\n{'='*60}")
        print(f"SORU: {question}")
        print('='*60)
        
        try:
            result = qa_chain({"query": question})
            
            print(f"\nCEVAP:")
            print(result['result'])
            
            print(f"\nKULLANILAN KAYNAKLAR:")
            for i, doc in enumerate(result['source_documents'], 1):
                print(f"\nKaynak {i}:")
                print(f"Başlık: {doc.metadata.get('title', 'N/A')}")
                print(f"Kategori: {doc.metadata.get('category', 'N/A')}")
                print(f"İçerik özeti: {doc.page_content[:150]}...")
                
        except Exception as e:
            print(f"Hata: {e}")

def advanced_rag_with_custom_prompt():
    """
    Özel prompt ile gelişmiş RAG
    """
    print("\n=== ÖZEL PROMPT İLE GELİŞMİŞ RAG ===")
    
    # Belgeler ve vector store oluştur
    documents = create_comprehensive_knowledge_base()
    embeddings = OpenAIEmbeddings()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    split_docs = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(split_docs, embeddings)
    
    # Özel prompt template
    custom_prompt = PromptTemplate(
        template="""Sen bir teknoloji uzmanısın. Aşağıdaki bağlam bilgilerini kullanarak soruyu yanıtla.
Eğer bağlam bilgilerinde cevap yoksa, "Bu bilgi mevcut kaynaklarda bulunmuyor" de.
Her zaman Türkçe yanıt ver ve teknik terimleri açıkla.

Bağlam Bilgileri:
{context}

Soru: {question}

Detaylı Yanıt:""",
        input_variables=["context", "question"]
    )
    
    # LLM oluştur
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    
    # Custom chain oluştur
    qa_chain = load_qa_chain(
        llm=llm,
        chain_type="stuff",
        prompt=custom_prompt,
        verbose=True
    )
    
    # Gelişmiş sorular
    advanced_questions = [
        "React'ta functional component'lerde state yönetimi nasıl yapılır ve class component'lerden farkı nedir?",
        "Makine öğrenmesinde overfitting nedir ve nasıl önlenir?",
        "SQL'de window function'lar nedir ve aggregate function'lardan farkı nedir?",
        "DevOps'ta CI/CD pipeline'ı nasıl kurulur ve faydaları nelerdir?"
    ]
    
    for question in advanced_questions:
        print(f"\n{'='*60}")
        print(f"GELİŞMİŞ SORU: {question}")
        print('='*60)
        
        try:
            # İlgili belgeleri getir
            relevant_docs = vectorstore.similarity_search(question, k=4)
            
            # Custom chain ile yanıtla
            response = qa_chain.run(
                input_documents=relevant_docs,
                question=question
            )
            
            print(f"\nDETAYLI YANIT:")
            print(response)
            
        except Exception as e:
            print(f"Hata: {e}")

def conversational_rag_example():
    """
    Konuşmalı RAG sistemi
    Geçmiş konuşmayı hatırlayarak RAG
    """
    print("\n=== KONUŞMALI RAG SİSTEMİ ===")
    
    # Belgeler ve vector store oluştur
    documents = create_comprehensive_knowledge_base()
    embeddings = OpenAIEmbeddings()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(split_docs, embeddings)
    
    # Memory oluştur
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # LLM oluştur
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    
    # Conversational RAG chain oluştur
    conv_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        verbose=True,
        return_source_documents=True
    )
    
    # Konuşma akışı simülasyonu
    conversation_flow = [
        "Python web geliştirme için hangi framework'leri önerirsin?",
        "Bunlardan hangisi yeni başlayanlar için daha uygun?",
        "Flask ile basit bir web uygulaması nasıl yapılır?",
        "Django'ya geçiş yapmak istesem neye dikkat etmeliyim?",
        "Az önce bahsettiğin Flask'ın avantajları nelerdi?",
        "Web geliştirmede makine öğrenmesi nasıl kullanılır?"
    ]
    
    for i, question in enumerate(conversation_flow, 1):
        print(f"\n{'='*50}")
        print(f"KONUŞMA ADIMI {i}: {question}")
        print('='*50)
        
        try:
            # Soruyu sor ve yanıtı al
            result = conv_chain({"question": question})
            
            print(f"\nYANIT:")
            print(result['answer'])
            
            print(f"\nGEÇMİŞ KONUŞMA:")
            for msg in memory.chat_memory.messages[-4:]:  # Son 2 soru-cevap çifti
                if hasattr(msg, 'content'):
                    print(f"{type(msg).__name__}: {msg.content[:100]}...")
                    
        except Exception as e:
            print(f"Hata: {e}")

def multi_document_rag_with_filtering():
    """
    Çoklu belge RAG sistemi filtreleme ile
    """
    print("\n=== FİLTRELEME İLE ÇOKLİ BELGE RAG ===")
    
    # Belgeler oluştur
    documents = create_comprehensive_knowledge_base()
    embeddings = OpenAIEmbeddings()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    split_docs = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(split_docs, embeddings)
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    
    # Farklı filtreler ve sorular
    filtered_queries = [
        {
            "filter": {"category": "web_development"},
            "question": "Modern web geliştirmede hangi teknolojiler kullanılır?",
            "description": "Sadece web geliştirme belgeleri"
        },
        {
            "filter": {"difficulty": "advanced"},
            "question": "İleri seviye teknolojiler hangileridir?",
            "description": "Sadece ileri seviye belgeler"
        },
        {
            "filter": {"language": "python"},
            "question": "Python ile neler yapılabilir?",
            "description": "Sadece Python ile ilgili belgeler"
        },
        {
            "filter": {"category": "machine_learning"},
            "question": "Makine öğrenmesi algoritmaları nasıl seçilir?", 
            "description": "Sadece ML belgeleri"
        }
    ]
    
    for query_config in filtered_queries:
        print(f"\n{'='*60}")
        print(f"FİLTRE: {query_config['description']}")
        print(f"SORU: {query_config['question']}")
        print('='*60)
        
        try:
            # Filtrelenmiş belgeler getir
            filtered_docs = vectorstore.similarity_search(
                query_config['question'],
                k=4,
                filter=query_config['filter']
            )
            
            print(f"Bulunan filtrelenmiş belge sayısı: {len(filtered_docs)}")
            
            if filtered_docs:
                # QA chain ile yanıtla
                qa_chain = load_qa_chain(llm, chain_type="stuff")
                response = qa_chain.run(
                    input_documents=filtered_docs,
                    question=query_config['question']
                )
                
                print(f"\nFİLTRELENMİŞ YANIT:")
                print(response)
                
                print(f"\nKULLANILAN BELGELER:")
                for i, doc in enumerate(filtered_docs, 1):
                    print(f"{i}. {doc.metadata.get('title', 'N/A')} "
                          f"({doc.metadata.get('category', 'N/A')})")
            else:
                print("Bu filtre ile eşleşen belge bulunamadı.")
                
        except Exception as e:
            print(f"Hata: {e}")

def rag_with_source_citation():
    """
    Kaynak atıfı ile RAG sistemi
    """
    print("\n=== KAYNAK ATIFI İLE RAG ===")
    
    # Belgeler ve vector store
    documents = create_comprehensive_knowledge_base()
    embeddings = OpenAIEmbeddings()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(split_docs, embeddings)
    
    # Kaynak atıfı için özel prompt
    citation_prompt = PromptTemplate(
        template="""Aşağıdaki bağlam bilgilerini kullanarak soruyu yanıtla.
Yanıtında kullandığın her bilgi için kaynak belirtmeyi unutma.
Kaynak format: [Belge Başlığı]

Bağlam Bilgileri:
{context}

Soru: {question}

Kaynak atıflı yanıt:""",
        input_variables=["context", "question"]
    )
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=citation_prompt)
    
    # Kaynak atıfı gerektiren sorular
    citation_questions = [
        "Web framework'leri karşılaştırır mısın?",
        "Makine öğrenmesi algoritmalarının kullanım alanları neler?",
        "Cloud computing servis modelleri nelerdir?"
    ]
    
    for question in citation_questions:
        print(f"\n{'='*60}")
        print(f"KAYNAK ATIFLI SORU: {question}")
        print('='*60)
        
        try:
            relevant_docs = vectorstore.similarity_search(question, k=4)
            
            # Context'i kaynak bilgileri ile zenginleştir
            enriched_context = ""
            for i, doc in enumerate(relevant_docs, 1):
                title = doc.metadata.get('title', f'Belge {i}')
                enriched_context += f"\n--- {title} ---\n{doc.page_content}\n"
            
            response = qa_chain.run(
                context=enriched_context,
                question=question
            )
            
            print(f"\nKAYNAK ATIFLI YANIT:")
            print(response)
            
            print(f"\nMEVCUT KAYNAK BELGELER:")
            for i, doc in enumerate(relevant_docs, 1):
                print(f"{i}. {doc.metadata.get('title', 'N/A')}")
                
        except Exception as e:
            print(f"Hata: {e}")

def main():
    """
    Ana fonksiyon - tüm RAG örneklerini çalıştır
    """
    print("LangChain RAG (Retrieval Augmented Generation) Örnekleri Başlıyor...\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("HATA: OPENAI_API_KEY environment variable bulunamadı!")
        return
    
    try:
        # RAG örneklerini çalıştır
        basic_rag_example()
        advanced_rag_with_custom_prompt()
        conversational_rag_example()
        multi_document_rag_with_filtering()
        rag_with_source_citation()
        
        print("\n✅ Tüm RAG örnekleri başarıyla tamamlandı!")
        
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")

if __name__ == "__main__":
    main()