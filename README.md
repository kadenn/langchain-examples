# LangChain Kapsamlı Öğrenme Projesi

Bu proje, LangChain kütüphanesinin tüm temel özelliklerini öğrenmek için oluşturulmuş kapsamlı bir eğitim setidir. Her modül, LangChain'in farklı bir yönünü detaylı açıklamalarla birlikte gösterir.

## 📚 Proje İçeriği

### 1. Temel LLM Kullanımı (`1_basic_llm.py`)
- LLM modelleriyle temel etkileşim
- Farklı model türleri (GPT-3.5, GPT-4)
- Mesaj tabanlı chat sistemi
- Streaming (akış) örnekleri
- Model karşılaştırmaları

### 2. Prompt Templates ve Chains (`2_prompts_and_chains.py`)
- Dinamik prompt template'leri
- Chat prompt template'leri
- Few-shot prompting
- LLM Chain kullanımı
- Sequential ve Simple Sequential Chains
- Özel output parser'lar

### 3. Memory Management (`3_memory_management.py`)
- Conversation Buffer Memory
- Conversation Summary Memory
- Token Buffer Memory
- Window Memory
- Özel hafıza yönetimi teknikleri

### 4. Document Loading (`4_document_loading.py`)
- Text dosyası yükleme
- PDF ve web sayfası yükleme
- Dizin bazlı yükleme
- Text splitting (karakter, recursive, token bazlı)
- Belge özetleme

### 5. Vector Stores ve Embeddings (`5_vector_stores_embeddings.py`)
- OpenAI Embeddings kullanımı
- Chroma ve FAISS vector store'ları
- Semantic search (anlamsal arama)
- Similarity search
- Retrieval QA
- Filtreleme ve metadata kullanımı

### 6. Agents ve Tools (`6_agents_and_tools.py`)
- Özel araçlar (tools) oluşturma
- ReAct agent kullanımı
- Conversational agent
- Multi-step problem solving
- Özel hesaplama araçları

### 7. RAG Sistemi (`7_rag_system.py`)
- Retrieval Augmented Generation
- Kapsamlı bilgi tabanı oluşturma
- Özel prompt'larla gelişmiş RAG
- Conversational RAG
- Multi-document RAG
- Kaynak atıfı ile RAG

## 🚀 Kurulum

### 1. Gereksinimler
```bash
# Projeyi klonlayın
git clone <repo-url>
cd langchain

# Sanal ortam oluşturun (önerilen)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\\Scripts\\activate    # Windows

# Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt
```

### 2. API Anahtarı Yapılandırması
```bash
# .env.example dosyasını .env olarak kopyalayın
cp .env.example .env

# .env dosyasını düzenleyip OpenAI API anahtarınızı ekleyin
OPENAI_API_KEY=your_openai_api_key_here
```

## 💻 Kullanım

### Etkileşimli Mod (Önerilen)
```bash
python main_demo.py
```
Bu komut menü sistemi ile tüm modülleri seçerek çalıştırmanızı sağlar.

### Tüm Modülleri Çalıştırma
```bash
python main_demo.py all
```

### Tek Modül Çalıştırma
```bash
python main_demo.py 1    # Temel LLM
python main_demo.py 2    # Prompt Templates
python main_demo.py 3    # Memory Management
# ... vb
```

### Yardım
```bash
python main_demo.py help
```

### Manuel Çalıştırma
Her modülü ayrı ayrı da çalıştırabilirsiniz:
```bash
python 1_basic_llm.py
python 2_prompts_and_chains.py
# ... vb
```

## 📖 Öğrenme Rehberi

### Başlangıç Seviyesi
1. `1_basic_llm.py` - LLM'lerle tanışın
2. `2_prompts_and_chains.py` - Prompt yazma sanatını öğrenin
3. `4_document_loading.py` - Belge işlemeyi keşfedin

### Orta Seviye
4. `3_memory_management.py` - Hafıza yönetimini anlayın
5. `5_vector_stores_embeddings.py` - Anlamsal aramayı keşfedin

### İleri Seviye
6. `6_agents_and_tools.py` - Akıllı aracılar oluşturun
7. `7_rag_system.py` - Gelişmiş RAG sistemleri yapın

## 🔍 Özellikler

### ✅ Kapsamlı Açıklamalar
- Her kod satırı için Türkçe açıklamalar
- Kavramsal açıklamalar ve örnekler
- Best practice'ler ve ipuçları

### ✅ Pratik Örnekler
- Gerçek dünya senaryoları
- Interaktif kod örnekleri
- Hata yönetimi ve edge case'ler

### ✅ Modüler Yapı
- Her konu ayrı modülde
- Bağımsız çalışabilen örnekler
- Progressif öğrenme yapısı

### ✅ Hata Yönetimi
- Detaylı hata mesajları
- Çözüm önerileri
- Graceful error handling

## 🛠️ Teknoloji Stack'i

- **LangChain**: Ana framework
- **OpenAI**: LLM sağlayıcısı
- **Chroma**: Vector veritabanı
- **FAISS**: Alternatif vector store
- **Python-dotenv**: Environment yönetimi

## 📋 Gereksinimler

- Python 3.8+
- OpenAI API anahtarı
- İnternet bağlantısı
- 2GB+ RAM (vector işlemleri için)

## 🔧 Sorun Giderme

### API Hatası
```
❌ OPENAI_API_KEY environment variable bulunamadı!
```
**Çözüm**: `.env` dosyasını oluşturun ve API anahtarınızı ekleyin.

### Import Hatası
```
❌ Eksik kütüphaneler: langchain
```
**Çözüm**: `pip install -r requirements.txt` komutunu çalıştırın.

### Memory Hatası
**Çözüm**: Daha küçük chunk_size değerleri kullanın.

## 📚 Ek Kaynaklar

- [LangChain Resmi Dökümantasyonu](https://docs.langchain.com/)
- [OpenAI API Referansı](https://platform.openai.com/docs)
- [Vector Database Rehberi](https://www.pinecone.io/learn/vector-database/)

## 🤝 Katkıda Bulunma

Bu proje eğitim amaçlıdır. Geliştirmeler için:

1. Fork yapın
2. Feature branch oluşturun
3. Değişikliklerinizi commit edin
4. Pull request gönderin

## 📄 Lisans

Bu proje MIT lisansı altında yayınlanmıştır.

## 🙏 Teşekkürler

- LangChain geliştirici ekibine
- OpenAI'ya güçlü API'leri için
- Açık kaynak topluluğuna

---

**İyi öğrenmeler! 🚀**