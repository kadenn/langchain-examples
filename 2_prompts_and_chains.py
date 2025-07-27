"""
LangChain Prompt Templates ve Chains
Bu dosya LangChain'in prompt yönetimi ve zincir (chain) özelliklerini gösterir:
- Prompt template'leri nasıl oluşturulur
- Dinamik prompt'lar nasıl yapılır
- Chain'ler nasıl kullanılır
- Sequential chain'ler nasıl oluşturulur
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.schema import BaseOutputParser

load_dotenv()

class CommaSeparatedListOutputParser(BaseOutputParser):
    """
    Özel output parser örneği
    LLM'den gelen cevabı virgülle ayrılmış liste olarak parse eder
    """
    
    def parse(self, text: str):
        """Metni virgülle ayrılmış listeye çevir"""
        return text.strip().split(", ")

def basic_prompt_template():
    """
    Temel prompt template kullanımı
    Değişkenli prompt'lar oluşturma
    """
    print("=== TEMEL PROMPT TEMPLATE ===")
    
    # Basit prompt template oluştur
    template = """
    Sen bir {role} uzmanısın.
    {topic} hakkında {style} bir açıklama yap.
    Açıklama maksimum {max_words} kelime olsun.
    """
    
    prompt = PromptTemplate(
        input_variables=["role", "topic", "style", "max_words"],
        template=template
    )
    
    # Template'i farklı değerlerle kullan
    formatted_prompt = prompt.format(
        role="yazılım geliştirici",
        topic="Python programlama",
        style="basit ve anlaşılır",
        max_words="100"
    )
    
    print("Oluşturulan Prompt:")
    print(formatted_prompt)
    print("-" * 50)

def chat_prompt_template():
    """
    Chat için özel prompt template
    Sistem ve kullanıcı mesajlarını template olarak kullanma
    """
    print("=== CHAT PROMPT TEMPLATE ===")
    
    # Chat prompt template oluştur
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "Sen bir {expertise} uzmanısın. Her zaman {tone} bir şekilde cevap ver."),
        ("human", "{user_question}")
    ])
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    # Template'i formatla ve LLM'e gönder
    messages = chat_template.format_messages(
        expertise="beslenme",
        tone="dostane ve yardımsever",
        user_question="Sağlıklı bir kahvaltı için ne önerirsin?"
    )
    
    response = llm.invoke(messages)
    print("Soru: Sağlıklı bir kahvaltı için ne önerirsin?")
    print(f"Uzman Cevabı: {response.content}\n")

def few_shot_prompt_example():
    """
    Few-shot prompting örneği
    LLM'e örnek giriş-çıkış çiftleri vererek öğretme
    """
    print("=== FEW-SHOT PROMPT ÖRNEĞI ===")
    
    # Örnekler tanımla
    examples = [
        {
            "input": "Python'da liste oluştur",
            "output": "my_list = [1, 2, 3, 4, 5]"
        },
        {
            "input": "Python'da sözlük oluştur", 
            "output": "my_dict = {'anahtar': 'değer', 'isim': 'Ali'}"
        },
        {
            "input": "Python'da döngü yaz",
            "output": "for i in range(5):\n    print(i)"
        }
    ]
    
    # Her örnek için template
    example_template = """
    Görev: {input}
    Kod: {output}
    """
    
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template=example_template
    )
    
    # Few-shot prompt template oluştur
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="Sen bir Python kod asistanısın. Verilen görevler için kısa kod örnekleri yaz:\n",
        suffix="\nGörev: {input}\nKod:",
        input_variables=["input"]
    )
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    
    # Yeni görev ver
    formatted_prompt = few_shot_prompt.format(input="Python'da dosya oku")
    response = llm.invoke(formatted_prompt)
    
    print("Few-shot örneklerle eğitilmiş LLM'den yeni görev:")
    print("Görev: Python'da dosya oku")
    print(f"Kod: {response.content}\n")

def basic_chain_example():
    """
    Temel LLM Chain kullanımı
    Prompt template + LLM + Output parsing
    """
    print("=== TEMEL CHAIN ÖRNEĞI ===")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    # Prompt template oluştur
    prompt = PromptTemplate(
        input_variables=["product"],
        template="Şu ürün için 5 yaratıcı slogan öner: {product}. Her sloganı virgülle ayır."
    )
    
    # Output parser oluştur
    parser = CommaSeparatedListOutputParser()
    
    # Chain oluştur (prompt + llm + parser)
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        output_parser=parser
    )
    
    # Chain'i çalıştır
    result = chain.run(product="akıllı telefon")
    
    print("Ürün: akıllı telefon")
    print("Oluşturulan sloganlar:")
    for i, slogan in enumerate(result, 1):
        print(f"{i}. {slogan}")
    print()

def sequential_chain_example():
    """
    Sequential Chain örneği
    Bir chain'in çıktısını diğerine girdi olarak verme
    """
    print("=== SEQUENTIAL CHAIN ÖRNEĞI ===")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    # İlk chain: Konu özeti oluştur
    first_prompt = PromptTemplate(
        input_variables=["topic"],
        template="'{topic}' konusu hakkında 100 kelimelik bir özet yaz."
    )
    first_chain = LLMChain(llm=llm, prompt=first_prompt, output_key="summary")
    
    # İkinci chain: Özeti tweet'e çevir
    second_prompt = PromptTemplate(
        input_variables=["summary"],
        template="Bu özeti 280 karakterlik bir tweet'e çevir:\n{summary}"
    )
    second_chain = LLMChain(llm=llm, prompt=second_prompt, output_key="tweet")
    
    # Sequential chain oluştur
    overall_chain = SequentialChain(
        chains=[first_chain, second_chain],
        input_variables=["topic"],
        output_variables=["summary", "tweet"],
        verbose=True  # Adımları göster
    )
    
    # Chain'i çalıştır
    result = overall_chain({"topic": "Yapay Zeka ve Gelecek"})
    
    print(f"Konu: Yapay Zeka ve Gelecek")
    print(f"\nÖzet:\n{result['summary']}")
    print(f"\nTweet:\n{result['tweet']}\n")

def simple_sequential_chain_example():
    """
    Simple Sequential Chain örneği
    Daha basit ardışık işlem zinciri
    """
    print("=== SIMPLE SEQUENTIAL CHAIN ÖRNEĞI ===")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8)
    
    # İlk chain: Hikaye başlangıcı
    story_prompt = PromptTemplate(
        input_variables=["character"],
        template="{character} karakteri ile başlayan 3 cümlelik bir hikaye başlangıcı yaz."
    )
    story_chain = LLMChain(llm=llm, prompt=story_prompt)
    
    # İkinci chain: Hikayeyi devam ettir
    continue_prompt = PromptTemplate(
        input_variables=["story_beginning"],
        template="Bu hikaye başlangıcını 2 cümle ile devam ettir:\n{story_beginning}"
    )
    continue_chain = LLMChain(llm=llm, prompt=continue_prompt)
    
    # Simple sequential chain (sadece bir çıktı değişkeni)
    overall_chain = SimpleSequentialChain(
        chains=[story_chain, continue_chain],
        verbose=True
    )
    
    # Chain'i çalıştır
    story = overall_chain.run("genç bir mühendis")
    
    print(f"Karakter: genç bir mühendis")
    print(f"\nTamamlanan Hikaye:\n{story}\n")

def main():
    """
    Ana fonksiyon - tüm prompt ve chain örneklerini çalıştır
    """
    print("LangChain Prompt ve Chain Örnekleri Başlıyor...\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("HATA: OPENAI_API_KEY environment variable bulunamadı!")
        return
    
    try:
        # Prompt örnekleri
        basic_prompt_template()
        chat_prompt_template()
        few_shot_prompt_example()
        
        # Chain örnekleri
        basic_chain_example()
        sequential_chain_example()
        simple_sequential_chain_example()
        
        print("✅ Tüm prompt ve chain örnekleri başarıyla tamamlandı!")
        
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")

if __name__ == "__main__":
    main()