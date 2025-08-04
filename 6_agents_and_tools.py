"""
LangChain Agents and Tools
This file demonstrates LangChain's agent and tool features:
- Creating custom tools
- Agent types and usage
- Using tools together
- ReAct agent
- Conversation agent
- Multi-step problem solving with agents
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

# Let's create sample custom tools

def calculator_tool(expression: str) -> str:
    """
    Mathematical calculation tool
    Evaluates simple mathematical expressions
    """
    try:
        # For security, only allow basic operators
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Only numbers and basic operators (+, -, *, /, (), .) are allowed"
        
        # Using eval is risky but we're using it in a controlled way in this example
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"

def weather_tool(city: str) -> str:
    """
    Weather tool (mock - requires API key for real API usage)
    """
    # Sample mock weather data
    mock_weather_data = {
        "new york": "New York: 22°C, Partly cloudy, Humidity: 65%",
        "london": "London: 18°C, Clear, Humidity: 45%", 
        "tokyo": "Tokyo: 25°C, Sunny, Humidity: 70%",
        "paris": "Paris: 28°C, Sunny, Humidity: 60%"
    }
    
    city_lower = city.lower()
    if city_lower in mock_weather_data:
        return mock_weather_data[city_lower]
    else:
        return f"Weather information not found for {city}. Available cities: New York, London, Tokyo, Paris"

def datetime_tool(format_type: str = "full") -> str:
    """
    Date and time information tool
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
    Text analysis tool
    Provides basic text statistics
    """
    words = text.split()
    characters = len(text)
    characters_no_spaces = len(text.replace(" ", ""))
    sentences = text.count('.') + text.count('!') + text.count('?')
    
    return f"""
    Text Analysis:
    - Word count: {len(words)}
    - Character count: {characters}
    - Characters without spaces: {characters_no_spaces}
    - Sentence count: {sentences}
    - Average word length: {sum(len(word) for word in words) / len(words):.2f}
    """

def currency_converter_tool(amount_and_currencies: str) -> str:
    """
    Currency converter (mock data)
    Format: "100 USD to EUR" 
    """
    # Mock exchange rates
    exchange_rates = {
        ("USD", "EUR"): 0.85,
        ("EUR", "USD"): 1.18,
        ("USD", "GBP"): 0.73,
        ("GBP", "USD"): 1.37,
        ("EUR", "GBP"): 0.86,
        ("GBP", "EUR"): 1.16
    }
    
    try:
        parts = amount_and_currencies.split()
        if "to" not in parts:
            return "Format: 'amount FROM_CURRENCY to TO_CURRENCY' (example: '100 USD to EUR')"
        
        amount = float(parts[0])
        from_currency = parts[1].upper()
        to_currency = parts[3].upper()
        
        rate_key = (from_currency, to_currency)
        if rate_key in exchange_rates:
            result = amount * exchange_rates[rate_key]
            return f"{amount} {from_currency} = {result:.2f} {to_currency}"
        else:
            return f"Exchange rate not found for {from_currency}-{to_currency} pair"
    
    except Exception as e:
        return f"Error: {str(e)}. Format: 'amount FROM_CURRENCY to TO_CURRENCY'"

def basic_agent_example():
    """
    Basic agent usage
    """
    print("=== BASIC AGENT USAGE ===")
    
    # Create LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    
    # Define tools
    tools = [
        Tool(
            name="Calculator",
            func=calculator_tool,
            description="Use for mathematical calculations. Input: mathematical expression (example: '2+2*3')"
        ),
        Tool(
            name="Weather",
            func=weather_tool,
            description="Use for weather information. Input: city name (New York, London, Tokyo, Paris)"
        ),
        Tool(
            name="DateTime",
            func=datetime_tool,
            description="Use for date and time information. Input: 'full', 'date', or 'time'"
        )
    ]
    
    # Initialize agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    
    # Test queries
    test_queries = [
        "Calculate 15 * 24 + 100",
        "What's the weather like in London?",
        "What's the current date and time?",
        "Calculate 250 + 150 then multiply by 5"
    ]
    
    for query in test_queries:
        print(f"\n--- Query: {query} ---")
        try:
            response = agent.run(query)
            print(f"Agent Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 50)

def react_agent_example():
    """
    ReAct (Reasoning + Acting) agent example
    Advanced combination of reasoning and acting
    """
    print("\n=== REACT AGENT EXAMPLE ===")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    
    # More complex tools
    tools = [
        Tool(
            name="Calculator",
            func=calculator_tool,
            description="Use for mathematical calculations"
        ),
        Tool(
            name="TextAnalysis",
            func=text_analysis_tool,
            description="Use for text analysis. Input: text to analyze"
        ),
        Tool(
            name="CurrencyConverter", 
            func=currency_converter_tool,
            description="Use for currency conversion. Format: 'amount FROM to TO' (example: '100 USD to EUR')"
        )
    ]
    
    # Get ReAct prompt template
    try:
        prompt = hub.pull("hwchase17/react")
    except:
        # Create simple prompt if can't get from hub
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
    
    # Create ReAct agent
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10
    )
    
    # Complex queries
    complex_queries = [
        "Analyze this text: 'Python is a great programming language. It's easy to learn and very powerful.' Then multiply the word count by 5.",
        "Convert 100 USD to EUR, then divide the result by 2",
        "Calculate: (25 * 4) + (30 / 2) - 10, then multiply the result by the word count of 'This number is very big'"
    ]
    
    for query in complex_queries:
        print(f"\n{'='*60}")
        print(f"Complex Query: {query}")
        print('='*60)
        
        try:
            response = agent_executor.invoke({"input": query})
            print(f"\nFinal Answer: {response['output']}")
        except Exception as e:
            print(f"Error: {e}")

def conversational_agent_example():
    """
    Conversational agent with memory
    Agent that remembers conversation history
    """
    print("\n=== CONVERSATIONAL AGENT ===")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
    
    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Tools
    tools = [
        Tool(
            name="Calculator",
            func=calculator_tool,
            description="Use for mathematical calculations"
        ),
        Tool(
            name="Weather",
            func=weather_tool,
            description="Use for weather information"
        ),
        Tool(
            name="DateTime",
            func=datetime_tool,
            description="Use for date and time information"
        )
    ]
    
    # Create conversational agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True
    )
    
    # Conversation simulation
    conversation_flow = [
        "Hello, my name is Alex. I'm 25 years old.",
        "What's the weather like in New York?",
        "Calculate 10 * 5 + 20",
        "Do you remember my name?",
        "Add 15 to the result you just calculated",
        "What's the current time?"
    ]
    
    for message in conversation_flow:
        print(f"\n--- User: {message} ---")
        try:
            response = agent.run(message)
            print(f"Agent: {response}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 40)

def custom_tool_agent_example():
    """
    Agent example with custom tools
    """
    print("\n=== AGENT WITH CUSTOM TOOLS ===")
    
    def fibonacci_tool(n: str) -> str:
        """Fibonacci sequence calculation tool"""
        try:
            num = int(n)
            if num < 0:
                return "Fibonacci cannot be calculated for negative numbers"
            elif num <= 1:
                return str(num)
            
            a, b = 0, 1
            for _ in range(2, num + 1):
                a, b = b, a + b
            return f"{num}th Fibonacci number: {b}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def prime_check_tool(n: str) -> str:
        """Prime number check tool"""
        try:
            num = int(n)
            if num < 2:
                return f"{num} is not a prime number"
            
            for i in range(2, int(math.sqrt(num)) + 1):
                if num % i == 0:
                    return f"{num} is not a prime number (divisible by {i})"
            
            return f"{num} is a prime number"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def word_reverse_tool(text: str) -> str:
        """Text reversal tool"""
        return f"Reversed: {text[::-1]}"
    
    # Custom tools
    custom_tools = [
        Tool(
            name="Fibonacci",
            func=fibonacci_tool,
            description="Calculates the Nth Fibonacci number. Input: positive integer"
        ),
        Tool(
            name="PrimeCheck",
            func=prime_check_tool,
            description="Checks if a number is prime. Input: positive integer"
        ),
        Tool(
            name="ReverseText",
            func=word_reverse_tool,
            description="Reverses given text. Input: any text"
        ),
        Tool(
            name="Calculator",
            func=calculator_tool,
            description="Use for mathematical calculations"
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
    
    # Test custom tools
    custom_queries = [
        "Calculate the 10th Fibonacci number",
        "Is 17 a prime number?",
        "Reverse the word 'LangChain'",
        "Find the 5th Fibonacci number, then check if it's prime",
        "Divide 100 by 7, then check if the result is prime"
    ]
    
    for query in custom_queries:
        print(f"\n--- Custom Tool Query: {query} ---")
        try:
            response = agent.run(query)
            print(f"Agent Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 50)

def multi_step_problem_solving():
    """
    Multi-step problem solving example
    """
    print("\n=== MULTI-STEP PROBLEM SOLVING ===")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    
    # Combine all tools
    all_tools = [
        Tool(
            name="Calculator",
            func=calculator_tool,
            description="Use for mathematical calculations"
        ),
        Tool(
            name="TextAnalysis",
            func=text_analysis_tool,
            description="Use for text analysis"
        ),
        Tool(
            name="CurrencyConverter",
            func=currency_converter_tool,
            description="Use for currency conversion"
        ),
        Tool(
            name="DateTime",
            func=datetime_tool,
            description="Use for date and time information"
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
    
    # Very complex problem
    complex_problem = """
    Follow these steps:
    1. Get today's date
    2. Analyze the text 'Today I am learning LangChain and having so much fun!'
    3. Multiply the word count of this text by 25
    4. Add the result to 100 USD and convert the total amount to EUR
    5. Tell me the final result
    """
    
    print(f"Complex Problem: {complex_problem}")
    print("-" * 80)
    
    try:
        response = agent.run(complex_problem)
        print(f"\nFINAL RESULT: {response}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    """
    Main function - run all agent and tool examples
    """
    print("LangChain Agents and Tools Examples Starting...\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not found!")
        return
    
    try:
        # Run agent examples
        basic_agent_example()
        react_agent_example()
        conversational_agent_example()
        custom_tool_agent_example()
        multi_step_problem_solving()
        
        print("\n✅ All agent and tool examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")

if __name__ == "__main__":
    main()