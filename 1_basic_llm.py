"""
LangChain Temel LLM Kullanımı
Bu dosya LangChain'in en temel özelliklerini gösterir:
- LLM'lerle nasıl etkileşim kurulur
- Farklı LLM modelleri nasıl kullanılır
- Temel prompt gönderme işlemleri
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# .env dosyasından environment değişkenlerini yükle
load_dotenv()

def basic_llm_example():
    """
    En temel LLM kullanımı örneği
    Bir LLM modeli oluşturup basit bir soru soruyoruz
    """
    print("=== TEMEL LLM KULLANIMI ===")
    
    # OpenAI Chat modeli oluştur
    # temperature: 0 = deterministik, 1 = yaratıcı
    # max_tokens: maksimum token sayısı (cevap uzunluğu)
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=500
    )
    
    # Basit bir soru sor
    response = llm.invoke("Türkiye'nin başkenti neresidir?")
    print(f"Soru: Türkiye'nin başkenti neresidir?")
    print(f"Cevap: {response.content}\n")
    
    return llm

def chat_with_messages():
    """
    Mesaj tabanlı chat örneği
    Sistem mesajı, kullanıcı mesajı ve AI mesajı ile etkileşim
    """
    print("=== MESAJ TABANLI CHAT ===")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
    
    # Farklı mesaj türleri ile bir konuşma oluştur
    messages = [
        # Sistem mesajı: AI'nin davranışını belirler
        SystemMessage(content="Sen bir Python programlama uzmanısın. Kısa ve net cevaplar ver."),
        
        # Kullanıcı mesajı
        HumanMessage(content="Python'da liste ve tuple arasındaki fark nedir?"),
    ]
    
    response = llm.invoke(messages)
    print(f"Sistem rolü: Python uzmanı")
    print(f"Kullanıcı: Python'da liste ve tuple arasındaki fark nedir?")
    print(f"AI: {response.content}\n")
    
    # Konuşmaya devam et
    messages.append(AIMessage(content=response.content))
    messages.append(HumanMessage(content="Bir örnek verebilir misin?"))
    
    response2 = llm.invoke(messages)
    print(f"Kullanıcı: Bir örnek verebilir misin?")
    print(f"AI: {response2.content}\n")

def different_models_comparison():
    """
    Farklı LLM modellerini karşılaştırma
    Aynı soruyu farklı modellere sorarak cevapları karşılaştırıyoruz
    """
    print("=== FARKLI MODEL KARŞILAŞTIRMASI ===")
    
    # Farklı modeller oluştur
    models = {
        "GPT-3.5-Turbo": ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3),
        "GPT-4": ChatOpenAI(model="gpt-4", temperature=0.3),
    }
    
    question = "Yapay zeka nedir? 50 kelime ile açıkla."
    
    for model_name, model in models.items():
        try:
            response = model.invoke(question)
            print(f"{model_name} Cevabı:")
            print(f"{response.content}\n")
            print("-" * 50)
        except Exception as e:
            print(f"{model_name} için hata: {e}\n")

def streaming_example():
    """
    Streaming (akış) örneği
    Cevabın parça parça gelmesini sağlar (gerçek zamanlı görünüm)
    """
    print("=== STREAMING ÖRNEĞI ===")
    
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        streaming=True
    )
    
    print("Soru: Bilgisayar programlamanın tarihçesi hakkında kısa bilgi ver")
    print("Cevap (streaming): ", end="")
    
    # Stream ile cevabı parça parça al
    for chunk in llm.stream("Bilgisayar programlamanın tarihçesi hakkında kısa bilgi ver"):
        print(chunk.content, end="", flush=True)
    
    print("\n")

def main():
    """
    Ana fonksiyon - tüm örnekleri çalıştır
    """
    print("LangChain Temel LLM Örnekleri Başlıyor...\n")
    
    # API anahtarının varlığını kontrol et
    if not os.getenv("OPENAI_API_KEY"):
        print("HATA: OPENAI_API_KEY environment variable bulunamadı!")
        print("Lütfen .env dosyasını oluşturun ve API anahtarınızı ekleyin.")
        return
    
    try:
        # Temel örnekleri çalıştır
        basic_llm_example()
        chat_with_messages()
        different_models_comparison()
        streaming_example()
        
        print("✅ Tüm temel LLM örnekleri başarıyla tamamlandı!")
        
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")
        print("API anahtarınızı ve internet bağlantınızı kontrol edin.")

if __name__ == "__main__":
    main()