"""
LangChain Document Loading ve Processing
Bu dosya LangChain'in belge yükleme ve işleme özelliklerini gösterir:
- Text dosyası yükleme
- PDF dosyası yükleme
- Web sayfası yükleme
- Belgeleri parçalara ayırma (Text Splitting)
- Farklı belge formatlarını işleme
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
    Test için örnek belgeler oluştur
    """
    print("=== ÖRNEK BELGELER OLUŞTURULUYOR ===")
    
    # Örnek metin dosyası oluştur
    sample_text = """
Python Programlama Dili

Python, 1991 yılında Guido van Rossum tarafından geliştirilmiş, yüksek seviyeli, 
genel amaçlı bir programlama dilidir. Python'un tasarım felsefesi, kod okunabilirliğini 
vurgular ve özellikle girintilerin anlamlı olarak kullanıldığı sözdizimi ile bilinir.

Python Özellikleri:
- Kolay öğrenim: Python'un sözdizimi sade ve anlaşılırdır
- Çok platformlu: Windows, macOS, Linux'ta çalışır
- Geniş kütüphane desteği: Standart kütüphane çok zengindir
- Açık kaynak: Ücretsiz ve kaynak kodu açıktır
- Nesne yönelimli: OOP paradigmasını destekler

Python Kullanım Alanları:
Web geliştirme, veri analizi, yapay zeka, makine öğrenmesi, otomasyon,
bilimsel hesaplama ve daha birçok alanda kullanılır.

Python'un popülaritesi son yıllarda özellikle veri bilimi ve yapay zeka 
alanlarındaki kullanımı ile büyük artış göstermiştir.
"""
    
    with open("sample_text.txt", "w", encoding="utf-8") as f:
        f.write(sample_text)
    
    # Örnek HTML dosyası oluştur
    sample_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Web Geliştirme</title>
</head>
<body>
    <h1>Modern Web Geliştirme</h1>
    <p>Web geliştirme, web siteleri ve web uygulamaları oluşturma sürecidir.</p>
    
    <h2>Frontend Teknolojileri</h2>
    <ul>
        <li>HTML - Yapı</li>
        <li>CSS - Stil</li>
        <li>JavaScript - Etkileşim</li>
        <li>React, Vue, Angular - Framework'ler</li>
    </ul>
    
    <h2>Backend Teknolojileri</h2>
    <ul>
        <li>Node.js</li>
        <li>Python (Django, Flask)</li>
        <li>Java (Spring)</li>
        <li>PHP</li>
    </ul>
    
    <p>Fullstack geliştiriciler hem frontend hem backend teknolojilerini kullanır.</p>
</body>
</html>
"""
    
    with open("sample_web.html", "w", encoding="utf-8") as f:
        f.write(sample_html)
    
    print("✅ Örnek dosyalar oluşturuldu: sample_text.txt, sample_web.html")

def text_loader_example():
    """
    Text dosyası yükleme örneği
    """
    print("\n=== TEXT DOSYASI YÜKLEME ===")
    
    # Text loader oluştur
    loader = TextLoader("sample_text.txt", encoding="utf-8")
    
    # Belgeyi yükle
    documents = loader.load()
    
    print(f"Yüklenen belge sayısı: {len(documents)}")
    print(f"İlk belgenin içeriği (ilk 200 karakter):")
    print(f"{documents[0].page_content[:200]}...")
    print(f"Belge metadata'sı: {documents[0].metadata}")

def directory_loader_example():
    """
    Dizin içindeki tüm dosyaları yükleme örneği
    """
    print("\n=== DİZİN LOADER ÖRNEĞI ===")
    
    # Dizindeki tüm .txt dosyalarını yükle
    loader = DirectoryLoader(".", glob="*.txt")
    documents = loader.load()
    
    print(f"Dizindeki toplam belge sayısı: {len(documents)}")
    for i, doc in enumerate(documents):
        print(f"Belge {i+1}: {doc.metadata['source']}")
        print(f"İçerik uzunluğu: {len(doc.page_content)} karakter")

def web_loader_example():
    """
    Web sayfası yükleme örneği
    """
    print("\n=== WEB SAYFA YÜKLEME ===")
    
    try:
        # Web sayfasını yükle
        loader = WebBaseLoader("https://python.org")
        documents = loader.load()
        
        print(f"Web sayfası yüklendi: {len(documents)} belge")
        print(f"İçerik uzunluğu: {len(documents[0].page_content)} karakter")
        print(f"İlk 300 karakter:")
        print(f"{documents[0].page_content[:300]}...")
        
    except Exception as e:
        print(f"Web yükleme hatası (internet bağlantısı gerekli): {e}")
        
        # Local HTML dosyasını alternatif olarak kullan
        print("Local HTML dosyası kullanılıyor...")
        with open("sample_web.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        
        # Manual document oluştur
        doc = Document(
            page_content=html_content,
            metadata={"source": "sample_web.html"}
        )
        print(f"Local HTML belgesi yüklendi: {len(doc.page_content)} karakter")

def character_text_splitter_example():
    """
    Karakter tabanlı metin bölme örneği
    """
    print("\n=== KARAKTER TABANLI METİN BÖLME ===")
    
    # Örnek belgeyi yükle
    loader = TextLoader("sample_text.txt", encoding="utf-8")
    documents = loader.load()
    
    # Character text splitter oluştur
    text_splitter = CharacterTextSplitter(
        chunk_size=200,      # Her parça maksimum 200 karakter
        chunk_overlap=50,    # Parçalar arası 50 karakter örtüşme
        length_function=len, # Uzunluk hesaplama fonksiyonu
        separator="\n\n"     # Paragraf ayırıcısı
    )
    
    # Belgeyi böl
    split_docs = text_splitter.split_documents(documents)
    
    print(f"Orijinal belge sayısı: {len(documents)}")
    print(f"Bölünmüş parça sayısı: {len(split_docs)}")
    
    # Her parçayı göster
    for i, doc in enumerate(split_docs):
        print(f"\n--- Parça {i+1} ---")
        print(f"Uzunluk: {len(doc.page_content)} karakter")
        print(f"İçerik: {doc.page_content[:100]}...")

def recursive_text_splitter_example():
    """
    Recursive (özyinelemeli) metin bölme örneği
    Daha akıllı bölme yapar
    """
    print("\n=== RECURSİVE METİN BÖLME ===")
    
    # Uzun metin örneği
    long_text = """
    Yapay Zeka ve Makine Öğrenmesi

    Yapay zeka (AI), makinelerin insan benzeri düşünme ve öğrenme yeteneklerini simüle etmesidir.
    
    Makine Öğrenmesi Türleri:
    1. Gözetimli Öğrenme (Supervised Learning)
    2. Gözetimsiz Öğrenme (Unsupervised Learning)  
    3. Pekiştirmeli Öğrenme (Reinforcement Learning)
    
    Derin Öğrenme (Deep Learning) ise yapay sinir ağlarını kullanarak karmaşık problemleri çözer.
    
    Günümüzde AI kullanım alanları:
    - Doğal dil işleme
    - Bilgisayarlı görü
    - Otonom araçlar
    - Tıp ve sağlık
    - Finans
    """
    
    # Recursive text splitter oluştur
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=30,
        separators=["\n\n", "\n", " ", ""]  # Öncelik sırası ile ayırıcılar
    )
    
    # Metni böl
    chunks = text_splitter.split_text(long_text)
    
    print(f"Toplam parça sayısı: {len(chunks)}")
    
    for i, chunk in enumerate(chunks):
        print(f"\n--- Parça {i+1} ---")
        print(f"Uzunluk: {len(chunk)} karakter")
        print(f"İçerik: {chunk.strip()}")

def token_text_splitter_example():
    """
    Token tabanlı metin bölme örneği
    LLM token limitlerini dikkate alır
    """
    print("\n=== TOKEN TABANLI METİN BÖLME ===")
    
    # Örnek belgeyi yükle
    loader = TextLoader("sample_text.txt", encoding="utf-8")
    documents = loader.load()
    
    # Token text splitter oluştur
    text_splitter = TokenTextSplitter(
        chunk_size=100,      # Her parça maksimum 100 token
        chunk_overlap=20     # Parçalar arası 20 token örtüşme
    )
    
    # Belgeleri böl
    split_docs = text_splitter.split_documents(documents)
    
    print(f"Token tabanlı bölünmüş parça sayısı: {len(split_docs)}")
    
    for i, doc in enumerate(split_docs):
        print(f"\n--- Token Parçası {i+1} ---")
        # Token sayısını tahmin et (yaklaşık)
        estimated_tokens = len(doc.page_content.split()) * 1.3
        print(f"Tahmini token sayısı: {estimated_tokens:.0f}")
        print(f"İçerik: {doc.page_content[:150]}...")

def document_summarization_example():
    """
    Belge özetleme örneği
    Yüklenen belgeleri özetler
    """
    print("\n=== BELGE ÖZETLEME ===")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    
    # Belgeyi yükle ve böl
    loader = TextLoader("sample_text.txt", encoding="utf-8")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    split_docs = text_splitter.split_documents(documents)
    
    # Özetleme chain'i yükle
    summarize_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",  # Uzun belgeler için map_reduce
        verbose=True
    )
    
    try:
        # Belgeyi özetle
        summary = summarize_chain.run(split_docs)
        
        print(f"Orijinal belge uzunluğu: {sum(len(doc.page_content) for doc in split_docs)} karakter")
        print(f"Özet uzunluğu: {len(summary)} karakter")
        print(f"\nÖzet:")
        print(summary)
        
    except Exception as e:
        print(f"Özetleme hatası: {e}")

def cleanup_files():
    """
    Test dosyalarını temizle
    """
    try:
        os.remove("sample_text.txt")
        os.remove("sample_web.html")
        print("\n✅ Test dosyaları temizlendi")
    except:
        pass

def main():
    """
    Ana fonksiyon - tüm belge yükleme örneklerini çalıştır
    """
    print("LangChain Document Loading Örnekleri Başlıyor...\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("HATA: OPENAI_API_KEY environment variable bulunamadı!")
        return
    
    try:
        # Örnek dosyaları oluştur
        create_sample_documents()
        
        # Belge yükleme örnekleri
        text_loader_example()
        directory_loader_example()
        web_loader_example()
        
        # Metin bölme örnekleri
        character_text_splitter_example()
        recursive_text_splitter_example()
        token_text_splitter_example()
        
        # Belge işleme örneği
        document_summarization_example()
        
        print("\n✅ Tüm document loading örnekleri başarıyla tamamlandı!")
        
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")
    
    finally:
        # Dosyaları temizle
        cleanup_files()

if __name__ == "__main__":
    main()