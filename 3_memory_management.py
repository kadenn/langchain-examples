"""
LangChain Memory Management
This file demonstrates LangChain's memory management features:
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
    Conversation Buffer Memory example
    Stores entire conversation history as is
    """
    print("=== CONVERSATION BUFFER MEMORY ===")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    # Create buffer memory
    memory = ConversationBufferMemory()
    
    # Create conversation chain
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True  # Show what's in memory
    )
    
    # First question
    response1 = conversation.predict(input="Hello, my name is John. What's your name?")
    print(f"AI: {response1}\n")
    
    # Second question (should remember the first question)
    response2 = conversation.predict(input="Do you remember my name?")
    print(f"AI: {response2}\n")
    
    # Third question
    response3 = conversation.predict(input="What do you know about Python programming?")
    print(f"AI: {response3}\n")
    
    # Show entire conversation in memory
    print("Conversation in Memory:")
    print(memory.buffer)
    print("-" * 50)

def conversation_summary_memory_example():
    """
    Conversation Summary Memory example
    Stores conversation history as summaries (ideal for long conversations)
    """
    print("=== CONVERSATION SUMMARY MEMORY ===")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    
    # Create summary memory
    memory = ConversationSummaryMemory(llm=llm)
    
    # Create conversation chain
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    # Long conversation simulation
    questions = [
        "Hello, I'm a software developer and I'm learning Python.",
        "What are the most commonly used data structures in Python?",
        "What's the difference between lists and dictionaries?",
        "Can you suggest a simple Python project for me?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n--- Question {i} ---")
        response = conversation.predict(input=question)
        print(f"User: {question}")
        print(f"AI: {response}")
    
    # Show summarized memory
    print(f"\nSummarized Memory:\n{memory.buffer}")
    print("-" * 50)

def conversation_token_buffer_memory_example():
    """
    Conversation Token Buffer Memory example
    Manages memory according to specific token limits
    """
    print("=== CONVERSATION TOKEN BUFFER MEMORY ===")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
    
    # Create token buffer memory (maximum 100 tokens)
    memory = ConversationTokenBufferMemory(
        llm=llm,
        max_token_limit=100  # Very low limit for testing
    )
    
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    # Ask questions with lots of text
    long_questions = [
        "Hello, I've been interested in programming for a long time and I'm currently in the process of learning Python.",
        "Can you provide detailed information about Django and Flask frameworks in web development?",
        "What is Python's role in machine learning and artificial intelligence, and which libraries are used?",
        "In my first question I talked about myself, do you remember me?"  # This question will test memory limits
    ]
    
    for i, question in enumerate(long_questions, 1):
        print(f"\n--- Question {i} ---")
        response = conversation.predict(input=question)
        print(f"AI: {response}")
        
        # Show token count in memory
        print(f"Token count in memory: {memory.llm.get_num_tokens(memory.buffer)}")
    
    print("-" * 50)

def conversation_summary_buffer_memory_example():
    """
    Conversation Summary Buffer Memory example
    Hybrid approach using both buffer and summary
    """
    print("=== CONVERSATION SUMMARY BUFFER MEMORY ===")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4)
    
    # Create summary buffer memory
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=150  # When this limit is reached, it creates summaries
    )
    
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    # Progressively longer conversation
    conversation_flow = [
        "Hello, I'm Sarah.",
        "I live in New York and I'm a computer engineer.",
        "I'm currently developing an e-commerce website.",
        "I'm using Python and Django for this project.",
        "I also chose PostgreSQL as the database.",
        "Do you remember my name and profession?"
    ]
    
    for i, message in enumerate(conversation_flow, 1):
        print(f"\n--- Message {i} ---")
        response = conversation.predict(input=message)
        print(f"User: {message}")
        print(f"AI: {response}")
        
        # Show memory status
        print(f"Buffer: {memory.chat_memory.messages}")
        if hasattr(memory, 'moving_summary_buffer') and memory.moving_summary_buffer:
            print(f"Summary: {memory.moving_summary_buffer}")
    
    print("-" * 50)

def conversation_window_memory_example():
    """
    Conversation Window Memory example
    Only remembers the last K messages (sliding window)
    """
    print("=== CONVERSATION WINDOW MEMORY ===")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6)
    
    # Create window memory (remember only last 2 interactions)
    memory = ConversationBufferWindowMemory(k=2)
    
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    # Ask many questions
    questions = [
        "My name is Michael.",
        "I'm 30 years old.",
        "I live in Chicago.", 
        "I work as a doctor.",
        "I love reading books.",
        "Do you remember my name?",      # First message should be forgotten
        "Do you remember my age?",       # This should also be forgotten
        "Do you remember my profession?" # This should be remembered
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n--- Question {i} ---")
        response = conversation.predict(input=question)
        print(f"User: {question}")
        print(f"AI: {response}")
        
        # Show message count in window
        print(f"Message count in memory: {len(memory.chat_memory.messages)}")
    
    print("-" * 50)

def custom_memory_with_specific_info():
    """
    Custom memory management example
    Using custom prompt template to store specific information
    """
    print("=== CUSTOM MEMORY MANAGEMENT ===")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
    
    # Custom prompt template with memory
    template = """You are a helpful assistant. You remember important information about the user.

User Information:
{history}

User: {input}
Assistant:"""
    
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
    
    # Personal information gathering conversation
    personal_questions = [
        "Hello, I'm Emma. I'm 25 years old and I'm a graphic designer.",
        "I live in Boston and I have a cat.",
        "In my free time, I like to paint and listen to music.",
        "My favorite music genre is jazz.",
        "Can you tell me what you remember about me?"
    ]
    
    for question in personal_questions:
        response = conversation.predict(input=question)
        print(f"User: {question}")
        print(f"Assistant: {response}\n")
    
    print("-" * 50)

def main():
    """
    Main function - run all memory management examples
    """
    print("LangChain Memory Management Examples Starting...\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not found!")
        return
    
    try:
        # Test different memory types
        conversation_buffer_memory_example()
        conversation_summary_memory_example()
        conversation_token_buffer_memory_example()
        conversation_summary_buffer_memory_example()
        conversation_window_memory_example()
        custom_memory_with_specific_info()
        
        print("✅ All memory management examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")

if __name__ == "__main__":
    main()