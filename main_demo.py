"""
LangChain Kapsamlı Demo Script
Bu ana script tüm LangChain özelliklerini sırayla çalıştırır ve menü sistemi sağlar.
Kullanıcı istediği modülü seçerek çalıştırabilir.
"""

import os
import sys
from dotenv import load_dotenv

# Tüm modülleri import et
try:
    import importlib.util
    
    # Modül dosyalarını kontrol et
    modules = {
        "1": ("Temel LLM Kullanımı", "1_basic_llm.py"),
        "2": ("Prompt Templates ve Chains", "2_prompts_and_chains.py"),
        "3": ("Memory Management", "3_memory_management.py"),
        "4": ("Document Loading", "4_document_loading.py"),
        "5": ("Vector Stores ve Embeddings", "5_vector_stores_embeddings.py"),
        "6": ("Agents ve Tools", "6_agents_and_tools.py"),
        "7": ("RAG Sistemi", "7_rag_system.py")
    }
    
except ImportError as e:
    print(f"Import hatası: {e}")
    sys.exit(1)

def check_environment():
    """
    Çevre değişkenlerini ve gereksinimleri kontrol et
    """
    print("=== ÇEVRE KONTROLLERI ===")
    
    # .env dosyasını yükle
    load_dotenv()
    
    # API anahtarı kontrolü
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ HATA: OPENAI_API_KEY environment variable bulunamadı!")
        print("\nÇözüm adımları:")
        print("1. .env.example dosyasını .env olarak kopyalayın")
        print("2. .env dosyasına OpenAI API anahtarınızı ekleyin")
        print("3. Örnek: OPENAI_API_KEY=your_api_key_here")
        return False
    
    # Temel kütüphaneleri kontrol et
    required_modules = [
        'langchain',
        'langchain_openai',
        'openai',
        'chromadb',
        'faiss',
        'dotenv'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module.replace('-', '_'))
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"❌ Eksik kütüphaneler: {', '.join(missing_modules)}")
        print("\nKurulum için çalıştırın:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ Tüm gereksinimler karşılandı")
    return True

def display_menu():
    """
    Ana menüyü göster
    """
    print("\n" + "="*60)
    print("           LANGCHAIN KAPSAMLI DEMO PROGRAMI")
    print("="*60)
    print("\nMevcut Modüller:")
    print("-" * 40)
    
    for key, (name, filename) in modules.items():
        # Dosya varlığını kontrol et
        if os.path.exists(filename):
            status = "✅"
        else:
            status = "❌"
        print(f"{key}. {name} {status}")
    
    print("\nDiğer Seçenekler:")
    print("-" * 40)
    print("a. Tüm modülleri sırayla çalıştır")
    print("h. Yardım ve kullanım bilgileri")
    print("q. Çıkış")
    print("\n" + "="*60)

def show_help():
    """
    Yardım bilgilerini göster
    """
    help_text = """
    LANGCHAIN DEMO PROGRAMI YARDIMI
    ================================
    
    Bu program LangChain kütüphanesinin tüm temel özelliklerini gösterir:
    
    📚 Modül Açıklamaları:
    
    1. Temel LLM Kullanımı:
       - LLM modellerini kullanma
       - Farklı model türleri
       - Streaming ve temel etkileşim
    
    2. Prompt Templates ve Chains:
       - Dinamik prompt oluşturma
       - Template kullanımı
       - Chain'leri birbirine bağlama
    
    3. Memory Management:
       - Konuşma geçmişi saklama
       - Farklı hafıza türleri
       - Context yönetimi
    
    4. Document Loading:
       - Belge yükleme ve işleme
       - Text splitting
       - Farklı format desteği
    
    5. Vector Stores ve Embeddings:
       - Anlamsal arama
       - Vector veritabanları
       - Embedding oluşturma
    
    6. Agents ve Tools:
       - Akıllı aracılar
       - Özel araçlar oluşturma
       - Çoklu adım problem çözme
    
    7. RAG Sistemi:
       - Retrieval Augmented Generation
       - Belgelerle desteklenmiş QA
       - Gelişmiş arama teknikleri
    
    🔧 Kurulum Gereksinimleri:
    - Python 3.8+
    - pip install -r requirements.txt
    - OpenAI API anahtarı (.env dosyasında)
    
    💡 Kullanım İpuçları:
    - Her modül bağımsız çalışır
    - Ayrıntılı açıklamalar kod içinde
    - Hata durumunda API anahtarını kontrol edin
    
    🐛 Sorun Giderme:
    - API hatası: OPENAI_API_KEY kontrol edin
    - Import hatası: requirements.txt'i yükleyin
    - Ağ hatası: İnternet bağlantınızı kontrol edin
    """
    print(help_text)

def run_module(module_key):
    """
    Belirli bir modülü çalıştır
    """
    if module_key not in modules:
        print(f"❌ Geçersiz modül seçimi: {module_key}")
        return False
    
    name, filename = modules[module_key]
    
    if not os.path.exists(filename):
        print(f"❌ Modül dosyası bulunamadı: {filename}")
        return False
    
    print(f"\n🚀 {name} modülü çalıştırılıyor...")
    print("-" * 50)
    
    try:
        # Modülü dinamik olarak yükle ve çalıştır
        spec = importlib.util.spec_from_file_location("module", filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # main fonksiyonunu çalıştır
        if hasattr(module, 'main'):
            module.main()
        else:
            print(f"❌ {filename} dosyasında main() fonksiyonu bulunamadı")
            return False
            
        print(f"\n✅ {name} modülü başarıyla tamamlandı!")
        return True
        
    except KeyboardInterrupt:
        print(f"\n⚠️  {name} modülü kullanıcı tarafından durduruldu")
        return False
    except Exception as e:
        print(f"\n❌ {name} modülünde hata oluştu: {str(e)}")
        return False

def run_all_modules():
    """
    Tüm modülleri sırayla çalıştır
    """
    print("\n🎯 TÜM MODÜLLER ÇALIŞTIRILACAK")
    print("="*50)
    
    response = input("Devam etmek istiyor musunuz? (y/N): ")
    if response.lower() not in ['y', 'yes', 'evet', 'e']:
        print("❌ İşlem iptal edildi")
        return
    
    results = {}
    start_time = __import__('time').time()
    
    for key in sorted(modules.keys()):
        name, filename = modules[key]
        print(f"\n📦 {key}/{len(modules)}: {name}")
        print("=" * 60)
        
        success = run_module(key)
        results[key] = {
            'name': name,
            'success': success
        }
        
        if success:
            print(f"✅ {name} - BAŞARILI")
        else:
            print(f"❌ {name} - BAŞARISIZ")
        
        # Modüller arası kısa bekleme
        print("\n⏳ Sonraki modüle geçiliyor...")
        __import__('time').sleep(2)
    
    # Özet rapor
    end_time = __import__('time').time()
    total_time = end_time - start_time
    
    print("\n" + "="*60)
    print("           TAMAMLANMA RAPORU")
    print("="*60)
    
    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)
    
    print(f"📊 Toplam Modül: {total}")
    print(f"✅ Başarılı: {successful}")
    print(f"❌ Başarısız: {total - successful}")
    print(f"⏱️  Toplam Süre: {total_time:.1f} saniye")
    
    print(f"\n📋 Detay Rapor:")
    for key, result in results.items():
        status = "✅ BAŞARILI" if result['success'] else "❌ BAŞARISIZ"
        print(f"  {key}. {result['name']}: {status}")
    
    if successful == total:
        print(f"\n🎉 TÜM MODÜLLER BAŞARIYLA TAMAMLANDI!")
    else:
        print(f"\n⚠️  {total - successful} modülde sorun yaşandı")

def interactive_mode():
    """
    Etkileşimli mod - kullanıcıdan sürekli girdi al
    """
    print("\n🔄 ETKİLEŞİMLİ MOD BAŞLATILDI")
    print("Çıkmak için 'q' yazın")
    
    while True:
        try:
            display_menu()
            choice = input("\nSeçiminizi yapın: ").strip().lower()
            
            if choice == 'q':
                print("\n👋 Güle güle!")
                break
            elif choice == 'h':
                show_help()
                input("\nDevam etmek için Enter'a basın...")
            elif choice == 'a':
                run_all_modules()
                input("\nDevam etmek için Enter'a basın...")
            elif choice in modules:
                run_module(choice)
                input("\nDevam etmek için Enter'a basın...")
            else:
                print(f"❌ Geçersiz seçim: {choice}")
                input("Devam etmek için Enter'a basın...")
                
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Program kullanıcı tarafından durduruldu")
            break
        except EOFError:
            print(f"\n\n👋 Program sonlandırıldı")
            break

def main():
    """
    Ana fonksiyon
    """
    print("🚀 LangChain Demo Programı Başlatılıyor...")
    
    # Çevre kontrolü
    if not check_environment():
        print("\n❌ Çevre kontrolleri başarısız. Program sonlandırılıyor.")
        sys.exit(1)
    
    # Komut satırı argümanlarını kontrol et
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg == 'all':
            run_all_modules()
        elif arg == 'help' or arg == 'h':
            show_help()
        elif arg in modules:
            run_module(arg)
        else:
            print(f"❌ Geçersiz argüman: {arg}")
            print("Kullanım: python main_demo.py [1-7|all|help]")
            sys.exit(1)
    else:
        # Etkileşimli mod
        interactive_mode()

if __name__ == "__main__":
    main()