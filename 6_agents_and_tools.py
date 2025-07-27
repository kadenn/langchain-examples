"""
LangChain Agents ve Tools
Bu dosya LangChain'in agent ve tool özelliklerini gösterir:
- Özel araçlar (tools) oluşturma
- Agent türleri ve kullanımları
- Tool'ları birlikte kullanma
- ReAct agent
- Conversation agent
- Agent ile çoklu adım problem çözme
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, create_react_agent
from langchain.tools import Tool, DuckDuckGoSearchRun
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain import hub
import requests
import json
import math
from datetime import datetime

load_dotenv()

# Örnek özel araçlar oluşturalım

def calculator_tool(expression: str) -> str:
    """
    Matematiksel hesaplama aracı
    Basit matematiksel ifadeleri değerlendirir
    """
    try:
        # Güvenlik için sadece temel operatörlere izin ver
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Hata: Sadece sayılar ve temel operatörler (+, -, *, /, (), .) kullanılabilir"
        
        # eval kullanımı riskli ama bu örnekte kontrollü şekilde kullanıyoruz
        result = eval(expression)
        return f"Sonuç: {result}"
    except Exception as e:
        return f"Hesaplama hatası: {str(e)}"

def weather_tool(city: str) -> str:
    """
    Hava durumu aracı (mock - gerçek API kullanımı için API key gerekir)
    """
    # Örnek mock hava durumu verisi
    mock_weather_data = {
        "istanbul": "İstanbul: 22°C, Parçalı bulutlu, Nem: %65",
        "ankara": "Ankara: 18°C, Açık, Nem: %45", 
        "izmir": "İzmir: 25°C, Güneşli, Nem: %70",
        "antalya": "Antalya: 28°C, Güneşli, Nem: %60"
    }
    
    city_lower = city.lower()
    if city_lower in mock_weather_data:
        return mock_weather_data[city_lower]
    else:
        return f"{city} için hava durumu bilgisi bulunamadı. Mevcut şehirler: İstanbul, Ankara, İzmir, Antalya"

def datetime_tool(format_type: str = "full") -> str:
    """
    Tarih ve saat bilgisi aracı
    """
    now = datetime.now()
    
    if format_type == "date":
        return now.strftime("%Y-%m-%d")
    elif format_type == "time":
        return now.strftime("%H:%M:%S")
    elif format_type == "full":
        return now.strftime("%Y-%m-%d %H:%M:%S")
    else:
        return now.strftime("%Y-%m-%d %H:%M:%S")

def text_analysis_tool(text: str) -> str:
    """
    Metin analiz aracı
    Temel metin istatistiklerini verir
    """
    words = text.split()
    characters = len(text)
    characters_no_spaces = len(text.replace(" ", ""))
    sentences = text.count('.') + text.count('!') + text.count('?')
    
    return f"""
    Metin Analizi:
    - Kelime sayısı: {len(words)}
    - Karakter sayısı: {characters}
    - Boşluksuz karakter sayısı: {characters_no_spaces}
    - Cümle sayısı: {sentences}
    - Ortalama kelime uzunluğu: {sum(len(word) for word in words) / len(words):.2f}
    """

def currency_converter_tool(amount_and_currencies: str) -> str:
    """
    Para birimi çevirici (mock veri)
    Format: "100 USD to TRY" 
    """
    # Mock döviz kurları
    exchange_rates = {
        ("USD", "TRY"): 27.5,
        ("EUR", "TRY"): 30.2,
        ("GBP", "TRY"): 35.1,
        ("USD", "EUR"): 0.91,
        ("EUR", "USD"): 1.10,
        ("TRY", "USD"): 0.036
    }
    
    try:
        parts = amount_and_currencies.split()
        if "to" not in parts:
            return "Format: 'miktar FROM_CURRENCY to TO_CURRENCY' (örnek: '100 USD to TRY')"
        
        amount = float(parts[0])
        from_currency = parts[1].upper()
        to_currency = parts[3].upper()
        
        rate_key = (from_currency, to_currency)
        if rate_key in exchange_rates:
            result = amount * exchange_rates[rate_key]
            return f"{amount} {from_currency} = {result:.2f} {to_currency}"
        else:
            return f"Döviz çifti {from_currency}-{to_currency} için kur bulunamadı"
    
    except Exception as e:
        return f"Hata: {str(e)}. Format: 'miktar FROM_CURRENCY to TO_CURRENCY'"

def basic_agent_example():
    """
    Temel agent kullanımı
    """
    print("=== TEMEL AGENT KULLANIMI ===")
    
    # LLM oluştur
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    
    # Araçları tanımla
    tools = [
        Tool(
            name="Calculator",
            func=calculator_tool,
            description="Matematiksel hesaplamalar için kullanın. Girdi: matematiksel ifade (örnek: '2+2*3')"
        ),
        Tool(
            name="Weather",
            func=weather_tool,
            description="Hava durumu bilgisi için kullanın. Girdi: şehir adı (İstanbul, Ankara, İzmir, Antalya)"
        ),
        Tool(
            name="DateTime",
            func=datetime_tool,
            description="Tarih ve saat bilgisi için kullanın. Girdi: 'full', 'date', veya 'time'"
        )
    ]
    
    # Agent'ı initialize et
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    
    # Test soruları
    test_queries = [
        "15 * 24 + 100 işlemini hesapla",
        "İstanbul'un hava durumu nasıl?",
        "Şu anki tarih ve saat nedir?",
        "250 + 150 çıkar sonra 5 ile çarp"
    ]
    
    for query in test_queries:
        print(f"\n--- Sorgu: {query} ---")
        try:
            response = agent.run(query)
            print(f"Agent Cevabı: {response}")
        except Exception as e:
            print(f"Hata: {e}")
        print("-" * 50)

def react_agent_example():
    """
    ReAct (Reasoning + Acting) agent örneği
    Daha gelişmiş düşünme ve hareket etme kombinasyonu
    """
    print("\n=== REACT AGENT ÖRNEĞİ ===")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    
    # Daha karmaşık araçlar
    tools = [
        Tool(
            name="Calculator",
            func=calculator_tool,
            description="Matematiksel hesaplamalar için kullanın"
        ),
        Tool(
            name="TextAnalysis",
            func=text_analysis_tool,
            description="Metin analizi için kullanın. Girdi: analiz edilecek metin"
        ),
        Tool(
            name="CurrencyConverter", 
            func=currency_converter_tool,
            description="Para birimi çevirisi için kullanın. Format: 'miktar FROM to TO' (örnek: '100 USD to TRY')"
        )
    ]
    
    # ReAct prompt template'ini al
    try:
        prompt = hub.pull("hwchase17/react")
    except:
        # Hub'dan alınamazsa basit bir prompt oluştur
        from langchain.prompts import PromptTemplate
        prompt = PromptTemplate(
            input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
            template="""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
        )
    
    # ReAct agent oluştur
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10
    )
    
    # Karmaşık sorular
    complex_queries = [
        "Bu metni analiz et: 'Python harika bir programlama dilidir. Öğrenmesi kolay ve çok güçlüdür.' Sonra kelime sayısını 5 ile çarp.",
        "100 USD'yi TRY'ye çevir, sonra sonucu 2'ye böl",
        "Şu hesaplamayı yap: (25 * 4) + (30 / 2) - 10, sonra sonucu 'Bu sayı çok büyük' metninin kelime sayısı ile çarp"
    ]
    
    for query in complex_queries:
        print(f"\n{'='*60}")
        print(f"Karmaşık Sorgu: {query}")
        print('='*60)
        
        try:
            response = agent_executor.invoke({"input": query})
            print(f"\nFinal Answer: {response['output']}")
        except Exception as e:
            print(f"Hata: {e}")

def conversational_agent_example():
    """
    Memory ile conversational agent
    Konuşma geçmişini hatırlayan agent
    """
    print("\n=== CONVERSATİONAL AGENT ===")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
    
    # Memory oluştur
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Araçlar
    tools = [
        Tool(
            name="Calculator",
            func=calculator_tool,
            description="Matematiksel hesaplamalar için kullanın"
        ),
        Tool(
            name="Weather",
            func=weather_tool,
            description="Hava durumu bilgisi için kullanın"
        ),
        Tool(
            name="DateTime",
            func=datetime_tool,
            description="Tarih ve saat bilgisi için kullanın"
        )
    ]
    
    # Conversational agent oluştur
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True
    )
    
    # Conversation simülasyonu
    conversation_flow = [
        "Merhaba, benim adım Ali. 25 yaşındayım.",
        "İstanbul'un hava durumu nasıl?",
        "10 * 5 + 20 hesapla",
        "Benim adımı hatırlıyor musun?",
        "Az önce hesapladığın sonuca 15 ekle",
        "Şu anki saat kaç?"
    ]
    
    for message in conversation_flow:
        print(f"\n--- Kullanıcı: {message} ---")
        try:
            response = agent.run(message)
            print(f"Agent: {response}")
        except Exception as e:
            print(f"Hata: {e}")
        print("-" * 40)

def custom_tool_agent_example():
    """
    Özel araçlarla agent örneği
    """
    print("\n=== ÖZEL ARAÇLARLA AGENT ===")
    
    def fibonacci_tool(n: str) -> str:
        """Fibonacci dizisi hesaplama aracı"""
        try:
            num = int(n)
            if num < 0:
                return "Negatif sayılar için Fibonacci hesaplanamaz"
            elif num <= 1:
                return str(num)
            
            a, b = 0, 1
            for _ in range(2, num + 1):
                a, b = b, a + b
            return f"{num}. Fibonacci sayısı: {b}"
        except Exception as e:
            return f"Hata: {str(e)}"
    
    def prime_check_tool(n: str) -> str:
        """Asal sayı kontrolü aracı"""
        try:
            num = int(n)
            if num < 2:
                return f"{num} asal sayı değildir"
            
            for i in range(2, int(math.sqrt(num)) + 1):
                if num % i == 0:
                    return f"{num} asal sayı değildir ({i} ile bölünebilir)"
            
            return f"{num} asal sayıdır"
        except Exception as e:
            return f"Hata: {str(e)}"
    
    def word_reverse_tool(text: str) -> str:
        """Metin ters çevirme aracı"""
        return f"Ters çevrilmiş: {text[::-1]}"
    
    # Custom tools
    custom_tools = [
        Tool(
            name="Fibonacci",
            func=fibonacci_tool,
            description="N. Fibonacci sayısını hesaplar. Girdi: pozitif tam sayı"
        ),
        Tool(
            name="PrimeCheck",
            func=prime_check_tool,
            description="Bir sayının asal olup olmadığını kontrol eder. Girdi: pozitif tam sayı"
        ),
        Tool(
            name="ReverseText",
            func=word_reverse_tool,
            description="Verilen metni ters çevirir. Girdi: herhangi bir metin"
        ),
        Tool(
            name="Calculator",
            func=calculator_tool,
            description="Matematiksel hesaplamalar için kullanın"
        )
    ]
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    
    agent = initialize_agent(
        tools=custom_tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    
    # Özel araçları test et
    custom_queries = [
        "10. Fibonacci sayısını hesapla",
        "17 sayısı asal mı?",
        "'LangChain' kelimesini ters çevir",
        "5. Fibonacci sayısını bul, sonra bu sayının asal olup olmadığını kontrol et",
        "100 sayısını 7'ye böl, sonra sonucun asal olup olmadığına bak"
    ]
    
    for query in custom_queries:
        print(f"\n--- Özel Araç Sorgusu: {query} ---")
        try:
            response = agent.run(query)
            print(f"Agent Cevabı: {response}")
        except Exception as e:
            print(f"Hata: {e}")
        print("-" * 50)

def multi_step_problem_solving():
    """
    Çok adımlı problem çözme örneği
    """
    print("\n=== ÇOK ADIMLI PROBLEM ÇÖZME ===")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    
    # Tüm araçları birleştir
    all_tools = [
        Tool(
            name="Calculator",
            func=calculator_tool,
            description="Matematiksel hesaplamalar için kullanın"
        ),
        Tool(
            name="TextAnalysis",
            func=text_analysis_tool,
            description="Metin analizi için kullanın"
        ),
        Tool(
            name="CurrencyConverter",
            func=currency_converter_tool,
            description="Para birimi çevirisi için kullanın"
        ),
        Tool(
            name="DateTime",
            func=datetime_tool,
            description="Tarih ve saat bilgisi için kullanın"
        )
    ]
    
    agent = initialize_agent(
        tools=all_tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=15
    )
    
    # Çok karmaşık problem
    complex_problem = """
    Şu adımları takip et:
    1. Bugünün tarihini al
    2. 'Bugün LangChain öğreniyorum ve çok eğleniyorum!' metnini analiz et
    3. Bu metnin kelime sayısını 25 ile çarp
    4. Çıkan sonucu 100 USD'ye ekle ve toplam miktarı TRY'ye çevir
    5. Final sonucu söyle
    """
    
    print(f"Karmaşık Problem: {complex_problem}")
    print("-" * 80)
    
    try:
        response = agent.run(complex_problem)
        print(f"\nFINAL SONUÇ: {response}")
    except Exception as e:
        print(f"Hata: {e}")

def main():
    """
    Ana fonksiyon - tüm agent ve tool örneklerini çalıştır
    """
    print("LangChain Agents ve Tools Örnekleri Başlıyor...\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("HATA: OPENAI_API_KEY environment variable bulunamadı!")
        return
    
    try:
        # Agent örneklerini çalıştır
        basic_agent_example()
        react_agent_example()
        conversational_agent_example()
        custom_tool_agent_example()
        multi_step_problem_solving()
        
        print("\n✅ Tüm agent ve tool örnekleri başarıyla tamamlandı!")
        
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")

if __name__ == "__main__":
    main()