"""
LangChain Prompt Templates and Chains
This file demonstrates LangChain's prompt management and chain features:
- How to create prompt templates
- How to make dynamic prompts
- How to use chains
- How to create sequential chains
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
    Custom output parser example
    Parses the response from LLM as a comma-separated list
    """
    
    def parse(self, text: str):
        """Convert text to comma-separated list"""
        return text.strip().split(", ")

def basic_prompt_template():
    """
    Basic prompt template usage
    Creating prompts with variables
    """
    print("=== BASIC PROMPT TEMPLATE ===")
    
    # Create simple prompt template
    template = """
    You are a {role} expert.
    Provide a {style} explanation about {topic}.
    The explanation should be maximum {max_words} words.
    """
    
    prompt = PromptTemplate(
        input_variables=["role", "topic", "style", "max_words"],
        template=template
    )
    
    # Use template with different values
    formatted_prompt = prompt.format(
        role="software developer",
        topic="Python programming",
        style="simple and understandable",
        max_words="100"
    )
    
    print("Generated Prompt:")
    print(formatted_prompt)
    print("-" * 50)

def chat_prompt_template():
    """
    Special prompt template for chat
    Using system and user messages as templates
    """
    print("=== CHAT PROMPT TEMPLATE ===")
    
    # Create chat prompt template
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "You are a {expertise} expert. Always respond in a {tone} manner."),
        ("human", "{user_question}")
    ])
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    # Format template and send to LLM
    messages = chat_template.format_messages(
        expertise="nutrition",
        tone="friendly and helpful",
        user_question="What do you recommend for a healthy breakfast?"
    )
    
    response = llm.invoke(messages)
    print("Question: What do you recommend for a healthy breakfast?")
    print(f"Expert Answer: {response.content}\n")

def few_shot_prompt_example():
    """
    Few-shot prompting example
    Teaching LLM with example input-output pairs
    """
    print("=== FEW-SHOT PROMPT EXAMPLE ===")
    
    # Define examples
    examples = [
        {
            "input": "Create a list in Python",
            "output": "my_list = [1, 2, 3, 4, 5]"
        },
        {
            "input": "Create a dictionary in Python", 
            "output": "my_dict = {'key': 'value', 'name': 'John'}"
        },
        {
            "input": "Write a loop in Python",
            "output": "for i in range(5):\n    print(i)"
        }
    ]
    
    # Template for each example
    example_template = """
    Task: {input}
    Code: {output}
    """
    
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template=example_template
    )
    
    # Create few-shot prompt template
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="You are a Python code assistant. Write short code examples for given tasks:\n",
        suffix="\nTask: {input}\nCode:",
        input_variables=["input"]
    )
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    
    # Give a new task
    formatted_prompt = few_shot_prompt.format(input="Read a file in Python")
    response = llm.invoke(formatted_prompt)
    
    print("LLM trained with few-shot examples for new task:")
    print("Task: Read a file in Python")
    print(f"Code: {response.content}\n")

def basic_chain_example():
    """
    Basic LLM Chain usage
    Prompt template + LLM + Output parsing
    """
    print("=== BASIC CHAIN EXAMPLE ===")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    # Create prompt template
    prompt = PromptTemplate(
        input_variables=["product"],
        template="Suggest 5 creative slogans for this product: {product}. Separate each slogan with a comma."
    )
    
    # Create output parser
    parser = CommaSeparatedListOutputParser()
    
    # Create chain (prompt + llm + parser)
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        output_parser=parser
    )
    
    # Run the chain
    result = chain.run(product="smartphone")
    
    print("Product: smartphone")
    print("Generated slogans:")
    for i, slogan in enumerate(result, 1):
        print(f"{i}. {slogan}")
    print()

def sequential_chain_example():
    """
    Sequential Chain example
    Using output of one chain as input to another
    """
    print("=== SEQUENTIAL CHAIN EXAMPLE ===")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    # First chain: Create topic summary
    first_prompt = PromptTemplate(
        input_variables=["topic"],
        template="Write a 100-word summary about '{topic}'."
    )
    first_chain = LLMChain(llm=llm, prompt=first_prompt, output_key="summary")
    
    # Second chain: Convert summary to tweet
    second_prompt = PromptTemplate(
        input_variables=["summary"],
        template="Convert this summary into a 280-character tweet:\n{summary}"
    )
    second_chain = LLMChain(llm=llm, prompt=second_prompt, output_key="tweet")
    
    # Create sequential chain
    overall_chain = SequentialChain(
        chains=[first_chain, second_chain],
        input_variables=["topic"],
        output_variables=["summary", "tweet"],
        verbose=True  # Show steps
    )
    
    # Run the chain
    result = overall_chain({"topic": "Artificial Intelligence and the Future"})
    
    print(f"Topic: Artificial Intelligence and the Future")
    print(f"\nSummary:\n{result['summary']}")
    print(f"\nTweet:\n{result['tweet']}\n")

def simple_sequential_chain_example():
    """
    Simple Sequential Chain example
    Simpler sequential processing chain
    """
    print("=== SIMPLE SEQUENTIAL CHAIN EXAMPLE ===")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8)
    
    # First chain: Story beginning
    story_prompt = PromptTemplate(
        input_variables=["character"],
        template="Write a 3-sentence story beginning featuring {character}."
    )
    story_chain = LLMChain(llm=llm, prompt=story_prompt)
    
    # Second chain: Continue the story
    continue_prompt = PromptTemplate(
        input_variables=["story_beginning"],
        template="Continue this story beginning with 2 more sentences:\n{story_beginning}"
    )
    continue_chain = LLMChain(llm=llm, prompt=continue_prompt)
    
    # Simple sequential chain (only one output variable)
    overall_chain = SimpleSequentialChain(
        chains=[story_chain, continue_chain],
        verbose=True
    )
    
    # Run the chain
    story = overall_chain.run("a young engineer")
    
    print(f"Character: a young engineer")
    print(f"\nCompleted Story:\n{story}\n")

def main():
    """
    Main function - run all prompt and chain examples
    """
    print("LangChain Prompt and Chain Examples Starting...\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not found!")
        return
    
    try:
        # Prompt examples
        basic_prompt_template()
        chat_prompt_template()
        few_shot_prompt_example()
        
        # Chain examples
        basic_chain_example()
        sequential_chain_example()
        simple_sequential_chain_example()
        
        print("✅ All prompt and chain examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")

if __name__ == "__main__":
    main()