"""
LangChain Vector Stores ve Embeddings
Bu dosya LangChain'in vector store ve embedding özelliklerini gösterir:
- Embeddings oluşturma
- Vector store kullanımı (Chroma, FAISS)
- Semantic search (anlamsal arama)
- Similarity search (benzerlik araması)
- Vector store ile belge arama
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
    Örnek bilgi tabanı belgeleri oluştur
    """
    print("=== ÖRNEK BİLGİ TABANI OLUŞTURULUYOR ===")
    
    # Çeşitli konularda örnek belgeler
    documents_data = [
        {
            "content": """
            Python Programlama Temelleri
            
            Python, kolay öğrenilen ve güçlü bir programlama dilidir. 
            Değişkenler, listeler, sözlükler ve fonksiyonlar Python'un temel yapı taşlarıdır.
            
            Temel Veri Tipleri:
            - int: Tam sayılar
            - float: Ondalıklı sayılar  
            - str: Metinler
            - bool: Doğru/Yanlış değerleri
            
            Python'da girintiler çok önemlidir ve kod bloklarını belirler.
            """,
            "metadata": {"topic": "python_basics", "difficulty": "beginner"}
        },
        {
            "content": """
            Web Geliştirme ve Framework'ler
            
            Modern web geliştirme frontend ve backend teknolojilerini içerir.
            
            Frontend: HTML, CSS, JavaScript
            Backend: Python (Django, Flask), Node.js, PHP
            
            Responsive tasarım, mobile-first yaklaşım ve performans optimizasyonu
            günümüz web geliştirmede kritik konulardır.
            
            Progressive Web Apps (PWA) ve Single Page Applications (SPA) 
            popüler trend'lerdir.
            """,
            "metadata": {"topic": "web_development", "difficulty": "intermediate"}
        },
        {
            "content": """
            Yapay Zeka ve Makine Öğrenmesi
            
            Yapay zeka, makinelerin insan benzeri düşünme yeteneklerini simüle etmesidir.
            
            Makine Öğrenmesi Türleri:
            1. Supervised Learning: Etiketli verilerle öğrenme
            2. Unsupervised Learning: Etiketsiz verilerle öğrenme
            3. Reinforcement Learning: Ödül-ceza sistemiyle öğrenme
            
            Derin öğrenme, yapay sinir ağları kullanarak karmaşık problemleri çözer.
            TensorFlow, PyTorch ve Scikit-learn popüler kütüphanelerdir.
            """,
            "metadata": {"topic": "ai_ml", "difficulty": "advanced"}
        },
        {
            "content": """
            Veri Bilimi ve Analizi
            
            Veri bilimi, büyük veri setlerinden anlamlı bilgiler çıkarma sanatıdır.
            
            Veri Bilimi Süreci:
            1. Veri toplama
            2. Veri temizleme
            3. Keşifsel veri analizi (EDA)
            4. Model oluşturma
            5. Sonuçları yorumlama
            
            Pandas, NumPy, Matplotlib ve Seaborn en çok kullanılan araçlardır.
            İstatistik bilgisi veri bilimci için temeldir.
            """,
            "metadata": {"topic": "data_science", "difficulty": "intermediate"}
        },
        {
            "content": """
            Mobil Uygulama Geliştirme
            
            Mobil uygulamalar iOS ve Android platformları için geliştirilir.
            
            Native Geliştirme:
            - iOS: Swift, Objective-C
            - Android: Java, Kotlin
            
            Cross-platform Geliştirme:
            - React Native
            - Flutter
            - Xamarin
            
            Mobile-first tasarım, kullanıcı deneyimi (UX) ve performans 
            mobil geliştirmede kritik faktörlerdir.
            """,
            "metadata": {"topic": "mobile_development", "difficulty": "intermediate"}
        }
    ]
    
    # Dokümanları oluştur
    documents = []
    for doc_data in documents_data:
        doc = Document(
            page_content=doc_data["content"],
            metadata=doc_data["metadata"]
        )
        documents.append(doc)
    
    print(f"✅ {len(documents)} örnek belge oluşturuldu")
    return documents

def embedding_basics_example():
    """
    Embedding'lerin temel kullanımı
    """
    print("\n=== EMBEDDİNG TEMELLERİ ===")
    
    # OpenAI embeddings oluştur
    embeddings = OpenAIEmbeddings()
    
    # Örnek metinler
    texts = [
        "Python harika bir programlama dilidir",
        "Web geliştirme çok eğlencelidir", 
        "Makine öğrenmesi geleceğin teknolojisidir",
        "JavaScript frontend için kullanılır"
    ]
    
    # Metinleri embedding'e çevir
    text_embeddings = embeddings.embed_documents(texts)
    
    print(f"Toplam metin sayısı: {len(texts)}")
    print(f"Her embedding'in boyutu: {len(text_embeddings[0])}")
    print(f"İlk embedding'in ilk 5 değeri: {text_embeddings[0][:5]}")
    
    # Tek bir sorgu için embedding
    query = "Python programlama"
    query_embedding = embeddings.embed_query(query)
    print(f"Sorgu embedding boyutu: {len(query_embedding)}")
    print(f"Sorgu embedding'in ilk 5 değeri: {query_embedding[:5]}")

def chroma_vector_store_example():
    """
    Chroma vector store kullanımı
    """
    print("\n=== CHROMA VECTOR STORE ===")
    
    # Örnek belgeleri al
    documents = create_sample_knowledge_base()
    
    # Embeddings oluştur
    embeddings = OpenAIEmbeddings()
    
    # Geçici dizin oluştur
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Chroma vector store oluştur
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=temp_dir
        )
        
        print(f"✅ Chroma vector store oluşturuldu: {len(documents)} belge")
        
        # Benzerlik araması yap
        query = "Python öğrenmek istiyorum"
        similar_docs = vectorstore.similarity_search(query, k=2)
        
        print(f"\nSorgu: '{query}'")
        print(f"Bulunan benzer belgeler: {len(similar_docs)}")
        
        for i, doc in enumerate(similar_docs, 1):
            print(f"\n--- Benzer Belge {i} ---")
            print(f"Konu: {doc.metadata.get('topic', 'N/A')}")
            print(f"İçerik: {doc.page_content[:200]}...")
        
        # Skor ile birlikte arama
        scored_docs = vectorstore.similarity_search_with_score(query, k=3)
        print(f"\n--- SKOR İLE ARAMA ---")
        for doc, score in scored_docs:
            print(f"Skor: {score:.4f} | Konu: {doc.metadata.get('topic', 'N/A')}")
    
    finally:
        # Geçici dizini temizle
        shutil.rmtree(temp_dir, ignore_errors=True)

def faiss_vector_store_example():
    """
    FAISS vector store kullanımı
    """
    print("\n=== FAISS VECTOR STORE ===")
    
    # Örnek belgeleri al
    documents = create_sample_knowledge_base()
    
    # Embeddings oluştur
    embeddings = OpenAIEmbeddings()
    
    # FAISS vector store oluştur
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )
    
    print(f"✅ FAISS vector store oluşturuldu: {len(documents)} belge")
    
    # Farklı arama türleri
    queries = [
        "web geliştirme nedir",
        "yapay zeka teknolojileri", 
        "veri analizi nasıl yapılır",
        "mobil app geliştirmek istiyorum"
    ]
    
    for query in queries:
        print(f"\n--- Sorgu: '{query}' ---")
        
        # En benzer belgeyi bul
        similar_docs = vectorstore.similarity_search(query, k=1)
        
        if similar_docs:
            doc = similar_docs[0]
            print(f"En benzer konu: {doc.metadata.get('topic', 'N/A')}")
            print(f"Zorluk: {doc.metadata.get('difficulty', 'N/A')}")
            print(f"İçerik özeti: {doc.page_content[:150]}...")
    
    # Vector store'u dosyaya kaydet
    try:
        vectorstore.save_local("faiss_index")
        print(f"\n✅ FAISS index dosyaya kaydedildi")
        
        # Kaydedilen index'i yükle
        loaded_vectorstore = FAISS.load_local("faiss_index", embeddings)
        test_result = loaded_vectorstore.similarity_search("Python", k=1)
        print(f"✅ Kaydedilen index başarıyla yüklendi")
        
        # Dosyaları temizle
        import glob
        for file in glob.glob("faiss_index*"):
            os.remove(file)
            
    except Exception as e:
        print(f"FAISS kaydetme/yükleme hatası: {e}")

def semantic_search_example():
    """
    Anlamsal arama örneği
    """
    print("\n=== ANLAMSAl ARAMA ÖRNEĞİ ===")
    
    # Daha detaylı örnek belgeler
    detailed_docs = [
        Document(
            page_content="Python'da liste comprehension kullanarak listeler oluşturabilirsiniz. Örnek: [x*2 for x in range(10)]",
            metadata={"category": "python", "type": "tutorial", "level": "intermediate"}
        ),
        Document(
            page_content="React.js kullanarak modern web uygulamaları geliştirebilirsiniz. Component tabanlı yaklaşım sunar.",
            metadata={"category": "web", "type": "framework", "level": "intermediate"}
        ),
        Document(
            page_content="Machine learning algoritmaları veri setlerinden otomatik olarak öğrenir ve tahminler yapar.",
            metadata={"category": "ai", "type": "concept", "level": "advanced"}
        ),
        Document(
            page_content="SQL veri tabanlarından veri sorgulamak için kullanılan standart dildir. SELECT, INSERT, UPDATE komutları temeldir.",
            metadata={"category": "database", "type": "language", "level": "beginner"}
        ),
        Document(
            page_content="Docker konteyner teknolojisi ile uygulamalarınızı kolayca paketleyip dağıtabilirsiniz.",
            metadata={"category": "devops", "type": "tool", "level": "intermediate"}
        )
    ]
    
    # Vector store oluştur
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(detailed_docs, embeddings)
    
    # Anlamsal sorular
    semantic_queries = [
        ("liste oluşturma", "Python'da veri yapıları"),
        ("frontend geliştirme", "Web arayüz teknolojileri"),
        ("otomatik öğrenme", "Yapay zeka algoritmaları"),
        ("veri sorgulama", "Veritabanı işlemleri"),
        ("uygulama dağıtımı", "DevOps araçları")
    ]
    
    for query, description in semantic_queries:
        print(f"\n--- {description} ---")
        print(f"Sorgu: '{query}'")
        
        # Anlamsal arama yap
        results = vectorstore.similarity_search_with_score(query, k=2)
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"\nSonuç {i} (Skor: {score:.4f}):")
            print(f"Kategori: {doc.metadata.get('category', 'N/A')}")
            print(f"Tip: {doc.metadata.get('type', 'N/A')}")
            print(f"İçerik: {doc.page_content}")

def retrieval_qa_example():
    """
    Retrieval QA (Bilgi Getirme + Soru Cevaplama) örneği
    Vector store ile QA chain birleştirme
    """
    print("\n=== RETRİEVAL QA ÖRNEĞİ ===")
    
    # Belgeler ve vector store oluştur
    documents = create_sample_knowledge_base()
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embeddings)
    
    # LLM oluştur
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    
    # Retrieval QA chain oluştur
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Belgeler birleştirilerek LLM'e verilir
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        verbose=True,
        return_source_documents=True
    )
    
    # Sorular sor
    questions = [
        "Python'un temel veri tipleri nelerdir?",
        "Web geliştirmede hangi teknolojiler kullanılır?",
        "Makine öğrenmesi türleri nelerdir?",
        "Veri bilimi süreci nasıl işler?",
        "Mobil uygulama geliştirmede hangi yaklaşımlar var?"
    ]
    
    for question in questions:
        print(f"\n{'='*50}")
        print(f"SORU: {question}")
        print('='*50)
        
        try:
            # Soruyu yanıtla
            result = qa_chain({"query": question})
            
            print(f"\nCEVAP: {result['result']}")
            
            # Kaynak belgeleri göster
            print(f"\nKAYNAK BELGELER:")
            for i, doc in enumerate(result['source_documents'], 1):
                print(f"\nKaynak {i}:")
                print(f"Konu: {doc.metadata.get('topic', 'N/A')}")
                print(f"Zorluk: {doc.metadata.get('difficulty', 'N/A')}")
                print(f"İçerik özeti: {doc.page_content[:100]}...")
                
        except Exception as e:
            print(f"Hata: {e}")

def vector_store_filtering_example():
    """
    Vector store'da filtreleme örneği
    Metadata tabanlı filtreleme
    """
    print("\n=== VECTOR STORE FİLTRELEME ===")
    
    # Belgeler oluştur
    documents = create_sample_knowledge_base()
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embeddings)
    
    # Farklı filtreler uygula
    filters_and_queries = [
        {
            "filter": {"difficulty": "beginner"},
            "query": "programlama öğrenmek istiyorum",
            "description": "Sadece başlangıç seviyesi"
        },
        {
            "filter": {"difficulty": "intermediate"},
            "query": "web teknolojileri",
            "description": "Orta seviye konular"
        },
        {
            "filter": {"topic": "ai_ml"},
            "query": "öğrenme algoritmaları",
            "description": "Sadece AI/ML konuları"
        }
    ]
    
    for filter_config in filters_and_queries:
        print(f"\n--- {filter_config['description']} ---")
        print(f"Filtre: {filter_config['filter']}")
        print(f"Sorgu: '{filter_config['query']}'")
        
        # Filtrelenmiş arama
        results = vectorstore.similarity_search(
            filter_config['query'],
            k=3,
            filter=filter_config['filter']
        )
        
        print(f"Bulunan sonuç sayısı: {len(results)}")
        for i, doc in enumerate(results, 1):
            print(f"\nSonuç {i}:")
            print(f"Konu: {doc.metadata.get('topic', 'N/A')}")
            print(f"Zorluk: {doc.metadata.get('difficulty', 'N/A')}")
            print(f"İçerik: {doc.page_content[:100]}...")

def main():
    """
    Ana fonksiyon - tüm vector store ve embedding örneklerini çalıştır
    """
    print("LangChain Vector Stores ve Embeddings Örnekleri Başlıyor...\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("HATA: OPENAI_API_KEY environment variable bulunamadı!")
        return
    
    try:
        # Embedding örnekleri
        embedding_basics_example()
        chroma_vector_store_example()
        faiss_vector_store_example()
        semantic_search_example()
        retrieval_qa_example()
        vector_store_filtering_example()
        
        print("\n✅ Tüm vector store ve embedding örnekleri başarıyla tamamlandı!")
        
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")

if __name__ == "__main__":
    main()