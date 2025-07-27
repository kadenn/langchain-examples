"""
LangChain Memory Management (Hafıza Yönetimi)
Bu dosya LangChain'in hafıza yönetimi özelliklerini gösterir:
- Conversation Buffer Memory
- Conversation Summary Memory
- Conversation Token Buffer Memory
- Conversation Window Memory
- Vector Store Memory
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationTokenBufferMemory,
    ConversationSummaryBufferMemory,
    ConversationBufferWindowMemory
)
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

load_dotenv()

def conversation_buffer_memory_example():
    """
    Conversation Buffer Memory örneği
    Tüm konuşma geçmişini olduğu gibi saklar
    """
    print("=== CONVERSATION BUFFER MEMORY ===")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    # Buffer memory oluştur
    memory = ConversationBufferMemory()
    
    # Conversation chain oluştur
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True  # Hafızada ne olduğunu göster
    )
    
    # İlk soru
    response1 = conversation.predict(input="Merhaba, benim adım Ali. Senin adın ne?")
    print(f"AI: {response1}\n")
    
    # İkinci soru (hafızada ilk soruyu hatırlayacak)
    response2 = conversation.predict(input="Benim adımı hatırlıyor musun?")
    print(f"AI: {response2}\n")
    
    # Üçüncü soru
    response3 = conversation.predict(input="Python hakkında ne biliyorsun?")
    print(f"AI: {response3}\n")
    
    # Hafızadaki tüm konuşmayı göster
    print("Hafızadaki Konuşma:")
    print(memory.buffer)
    print("-" * 50)

def conversation_summary_memory_example():
    """
    Conversation Summary Memory örneği
    Konuşma geçmişini özetleyerek saklar (uzun konuşmalar için ideal)
    """
    print("=== CONVERSATION SUMMARY MEMORY ===")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    
    # Summary memory oluştur
    memory = ConversationSummaryMemory(llm=llm)
    
    # Conversation chain oluştur
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    # Uzun bir konuşma simülasyonu
    questions = [
        "Merhaba, ben bir yazılım geliştiriciyim ve Python öğreniyorum.",
        "Python'da en çok kullanılan veri yapıları nelerdir?",
        "Liste ve sözlük arasındaki fark nedir?",
        "Bana basit bir Python projesi önerebilir misin?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n--- Soru {i} ---")
        response = conversation.predict(input=question)
        print(f"Kullanıcı: {question}")
        print(f"AI: {response}")
    
    # Özetlenmiş hafızayı göster
    print(f"\nÖzetlenmiş Hafıza:\n{memory.buffer}")
    print("-" * 50)

def conversation_token_buffer_memory_example():
    """
    Conversation Token Buffer Memory örneği
    Belirli token limitine göre hafızayı yönetir
    """
    print("=== CONVERSATION TOKEN BUFFER MEMORY ===")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
    
    # Token buffer memory oluştur (maksimum 100 token)
    memory = ConversationTokenBufferMemory(
        llm=llm,
        max_token_limit=100  # Çok düşük limit test için
    )
    
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    # Çok fazla metin içeren sorular sor
    long_questions = [
        "Merhaba, ben uzun bir süre boyunca programlama ile ilgileniyorum ve şu anda Python öğrenme sürecindeyim.",
        "Web geliştirme alanında Django ve Flask framework'leri hakkında detaylı bilgi verebilir misin?",
        "Makine öğrenmesi ve yapay zeka alanında Python'un rolü nedir ve hangi kütüphaneler kullanılır?",
        "İlk sorumda kendimden bahsetmiştim, beni hatırlıyor musun?"  # Bu soru hafıza limitini test edecek
    ]
    
    for i, question in enumerate(long_questions, 1):
        print(f"\n--- Soru {i} ---")
        response = conversation.predict(input=question)
        print(f"AI: {response}")
        
        # Hafızadaki token sayısını göster
        print(f"Hafızadaki token sayısı: {memory.llm.get_num_tokens(memory.buffer)}")
    
    print("-" * 50)

def conversation_summary_buffer_memory_example():
    """
    Conversation Summary Buffer Memory örneği
    Hem buffer hem de summary kullanarak hybrid yaklaşım
    """
    print("=== CONVERSATION SUMMARY BUFFER MEMORY ===")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4)
    
    # Summary buffer memory oluştur
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=150  # Bu limite ulaşınca özet çıkarır
    )
    
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    # Progressif olarak uzayan konuşma
    conversation_flow = [
        "Merhaba, ben Ayşe.",
        "İstanbul'da yaşıyorum ve bilgisayar mühendisiyim.",
        "Şu anda bir e-ticaret sitesi geliştiriyorum.",
        "Python ve Django kullanıyorum bu proje için.",
        "Veritabanı olarak da PostgreSQL tercih ettim.",
        "Benim adımı ve mesleğimi hatırlıyor musun?"
    ]
    
    for i, message in enumerate(conversation_flow, 1):
        print(f"\n--- Mesaj {i} ---")
        response = conversation.predict(input=message)
        print(f"Kullanıcı: {message}")
        print(f"AI: {response}")
        
        # Hafıza durumunu göster
        print(f"Buffer: {memory.chat_memory.messages}")
        if hasattr(memory, 'moving_summary_buffer') and memory.moving_summary_buffer:
            print(f"Summary: {memory.moving_summary_buffer}")
    
    print("-" * 50)

def conversation_window_memory_example():
    """
    Conversation Window Memory örneği
    Sadece son K mesajı hatırlar (sliding window)
    """
    print("=== CONVERSATION WINDOW MEMORY ===")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6)
    
    # Window memory oluştur (sadece son 2 etkileşimi hatırla)
    memory = ConversationBufferWindowMemory(k=2)
    
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    # Çok sayıda soru sor
    questions = [
        "Benim adım Mehmet.",
        "30 yaşındayım.",
        "İzmir'de yaşıyorum.", 
        "Doktor olarak çalışıyorum.",
        "Kitap okumayı seviyorum.",
        "Benim adımı hatırlıyor musun?",  # İlk mesaj unutulmuş olmalı
        "Yaşımı hatırlıyor musun?",      # Bu da unutulmuş olmalı
        "Mesleğimi hatırlıyor musun?"    # Bu hatırlanmalı
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n--- Soru {i} ---")
        response = conversation.predict(input=question)
        print(f"Kullanıcı: {question}")
        print(f"AI: {response}")
        
        # Window'daki mesaj sayısını göster
        print(f"Hafızadaki mesaj sayısı: {len(memory.chat_memory.messages)}")
    
    print("-" * 50)

def custom_memory_with_specific_info():
    """
    Özel hafıza yönetimi örneği
    Belirli bilgileri saklamak için özel prompt template kullanma
    """
    print("=== ÖZEL HAFIZA YÖNETİMİ ===")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
    
    # Özel prompt template ile hafıza
    template = """Sen yardımcı bir asistansın. Kullanıcı hakkında önemli bilgileri hatırlarsın.

Kullanıcı Bilgileri:
{history}

Kullanıcı: {input}
Asistan:"""
    
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=template
    )
    
    memory = ConversationBufferMemory()
    
    conversation = ConversationChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=True
    )
    
    # Kişisel bilgi toplama konuşması
    personal_questions = [
        "Merhaba, ben Elif. 25 yaşındayım ve grafik tasarımcıyım.",
        "Ankara'da yaşıyorum ve kedi beslerim.",
        "Boş zamanlarımda resim yapmayı ve müzik dinlemeyi severim.",
        "Favori müzik türüm jazz'dır.",
        "Benim hakkımda hatırladığın şeyleri söyleyebilir misin?"
    ]
    
    for question in personal_questions:
        response = conversation.predict(input=question)
        print(f"Kullanıcı: {question}")
        print(f"Asistan: {response}\n")
    
    print("-" * 50)

def main():
    """
    Ana fonksiyon - tüm hafıza yönetimi örneklerini çalıştır
    """
    print("LangChain Memory Management Örnekleri Başlıyor...\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("HATA: OPENAI_API_KEY environment variable bulunamadı!")
        return
    
    try:
        # Farklı hafıza türlerini test et
        conversation_buffer_memory_example()
        conversation_summary_memory_example()
        conversation_token_buffer_memory_example()
        conversation_summary_buffer_memory_example()
        conversation_window_memory_example()
        custom_memory_with_specific_info()
        
        print("✅ Tüm memory management örnekleri başarıyla tamamlandı!")
        
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")

if __name__ == "__main__":
    main()