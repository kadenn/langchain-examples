"""
LangChain Basic LLM Usage
This file demonstrates the most basic features of LangChain:
- How to interact with LLMs
- How to use different LLM models
- Basic prompt sending operations
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# Load environment variables from .env file
load_dotenv()

def basic_llm_example():
    """
    Most basic LLM usage example
    Create an LLM model and ask a simple question
    """
    print("=== BASIC LLM USAGE ===")
    
    # Create OpenAI Chat model
    # temperature: 0 = deterministic, 1 = creative
    # max_tokens: maximum token count (response length)
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=500
    )
    
    # Ask a simple question
    response = llm.invoke("What is the capital of Turkey?")
    print(f"Question: What is the capital of Turkey?")
    print(f"Answer: {response.content}\n")
    
    return llm

def chat_with_messages():
    """
    Message-based chat example
    Interaction with system message, user message, and AI message
    """
    print("=== MESSAGE-BASED CHAT ===")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
    
    # Create a conversation with different message types
    messages = [
        # System message: determines AI's behavior
        SystemMessage(content="You are a Python programming expert. Give short and clear answers."),
        
        # User message
        HumanMessage(content="What is the difference between lists and tuples in Python?"),
    ]
    
    response = llm.invoke(messages)
    print(f"System role: Python expert")
    print(f"User: What is the difference between lists and tuples in Python?")
    print(f"AI: {response.content}\n")
    
    # Continue the conversation
    messages.append(AIMessage(content=response.content))
    messages.append(HumanMessage(content="Can you give me an example?"))
    
    response2 = llm.invoke(messages)
    print(f"User: Can you give me an example?")
    print(f"AI: {response2.content}\n")

def different_models_comparison():
    """
    Comparing different LLM models
    Ask the same question to different models and compare answers
    """
    print("=== DIFFERENT MODEL COMPARISON ===")
    
    # Create different models
    models = {
        "GPT-3.5-Turbo": ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3),
        "GPT-4": ChatOpenAI(model="gpt-4", temperature=0.3),
    }
    
    question = "What is artificial intelligence? Explain in 50 words."
    
    for model_name, model in models.items():
        try:
            response = model.invoke(question)
            print(f"{model_name} Answer:")
            print(f"{response.content}\n")
            print("-" * 50)
        except Exception as e:
            print(f"Error for {model_name}: {e}\n")

def streaming_example():
    """
    Streaming example
    Allows the response to come in chunks (real-time view)
    """
    print("=== STREAMING EXAMPLE ===")
    
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        streaming=True
    )
    
    print("Question: Tell me briefly about the history of computer programming")
    print("Answer (streaming): ", end="")
    
    # Get response in chunks with stream
    for chunk in llm.stream("Tell me briefly about the history of computer programming"):
        print(chunk.content, end="", flush=True)
    
    print("\n")

def main():
    """
    Main function - run all examples
    """
    print("LangChain Basic LLM Examples Starting...\n")
    
    # Check API key existence
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not found!")
        print("Please create .env file and add your API key.")
        return
    
    try:
        # Run basic examples
        basic_llm_example()
        chat_with_messages()
        different_models_comparison()
        streaming_example()
        
        print("✅ All basic LLM examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Check your API key and internet connection.")

if __name__ == "__main__":
    main()