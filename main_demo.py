"""
LangChain Comprehensive Demo Script
This main script runs all LangChain features sequentially and provides a menu system.
Users can select and run the desired module.
"""

import os
import sys
from dotenv import load_dotenv

# Import all modules
try:
    import importlib.util
    
    # Module files check
    modules = {
        "1": ("Basic LLM Usage", "1_basic_llm.py"),
        "2": ("Prompt Templates and Chains", "2_prompts_and_chains.py"),
        "3": ("Memory Management", "3_memory_management.py"),
        "4": ("Document Loading", "4_document_loading.py"),
        "5": ("Vector Stores and Embeddings", "5_vector_stores_embeddings.py"),
        "6": ("Agents and Tools", "6_agents_and_tools.py"),
        "7": ("RAG System", "7_rag_system.py")
    }
    
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def check_environment():
    """
    Check environment variables and requirements
    """
    print("=== ENVIRONMENT CHECKS ===")
    
    # Load .env file
    load_dotenv()
    
    # API key check
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY environment variable not found!")
        print("\nSolution steps:")
        print("1. Copy .env.example file as .env")
        print("2. Add your OpenAI API key to .env file")
        print("3. Example: OPENAI_API_KEY=your_api_key_here")
        return False
    
    # Check basic libraries
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
        print(f"❌ Missing libraries: {', '.join(missing_modules)}")
        print("\nTo install run:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ All requirements met")
    return True

def display_menu():
    """
    Display main menu
    """
    print("\n" + "="*60)
    print("           LANGCHAIN COMPREHENSIVE DEMO PROGRAM")
    print("="*60)
    print("\nAvailable Modules:")
    print("-" * 40)
    
    for key, (name, filename) in modules.items():
        # Check file existence
        if os.path.exists(filename):
            status = "✅"
        else:
            status = "❌"
        print(f"{key}. {name} {status}")
    
    print("\nOther Options:")
    print("-" * 40)
    print("a. Run all modules sequentially")
    print("h. Help and usage information")
    print("q. Exit")
    print("\n" + "="*60)

def show_help():
    """
    Show help information
    """
    help_text = """
    LANGCHAIN DEMO PROGRAM HELP
    ============================
    
    This program demonstrates all basic features of the LangChain library:
    
    📚 Module Descriptions:
    
    1. Basic LLM Usage:
       - Using LLM models
       - Different model types
       - Streaming and basic interaction
    
    2. Prompt Templates and Chains:
       - Creating dynamic prompts
       - Using templates
       - Connecting chains together
    
    3. Memory Management:
       - Storing conversation history
       - Different memory types
       - Context management
    
    4. Document Loading:
       - Loading and processing documents
       - Text splitting
       - Different format support
    
    5. Vector Stores and Embeddings:
       - Semantic search
       - Vector databases
       - Creating embeddings
    
    6. Agents and Tools:
       - Intelligent agents
       - Creating custom tools
       - Multi-step problem solving
    
    7. RAG System:
       - Retrieval Augmented Generation
       - Document-supported QA
       - Advanced search techniques
    
    🔧 Installation Requirements:
    - Python 3.8+
    - pip install -r requirements.txt
    - OpenAI API key (in .env file)
    
    💡 Usage Tips:
    - Each module works independently
    - Detailed explanations in code
    - Check API key if errors occur
    
    🐛 Troubleshooting:
    - API error: Check OPENAI_API_KEY
    - Import error: Install requirements.txt
    - Network error: Check internet connection
    """
    print(help_text)

def run_module(module_key):
    """
    Run a specific module
    """
    if module_key not in modules:
        print(f"❌ Invalid module selection: {module_key}")
        return False
    
    name, filename = modules[module_key]
    
    if not os.path.exists(filename):
        print(f"❌ Module file not found: {filename}")
        return False
    
    print(f"\n🚀 Running {name} module...")
    print("-" * 50)
    
    try:
        # Load and run module dynamically
        spec = importlib.util.spec_from_file_location("module", filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Run main function
        if hasattr(module, 'main'):
            module.main()
        else:
            print(f"❌ main() function not found in {filename}")
            return False
            
        print(f"\n✅ {name} module completed successfully!")
        return True
        
    except KeyboardInterrupt:
        print(f"\n⚠️  {name} module stopped by user")
        return False
    except Exception as e:
        print(f"\n❌ Error occurred in {name} module: {str(e)}")
        return False

def run_all_modules():
    """
    Run all modules sequentially
    """
    print("\n🎯 ALL MODULES WILL BE RUN")
    print("="*50)
    
    response = input("Do you want to continue? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("❌ Operation cancelled")
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
            print(f"✅ {name} - SUCCESS")
        else:
            print(f"❌ {name} - FAILED")
        
        # Short wait between modules
        print("\n⏳ Moving to next module...")
        __import__('time').sleep(2)
    
    # Summary report
    end_time = __import__('time').time()
    total_time = end_time - start_time
    
    print("\n" + "="*60)
    print("           COMPLETION REPORT")
    print("="*60)
    
    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)
    
    print(f"📊 Total Modules: {total}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {total - successful}")
    print(f"⏱️  Total Time: {total_time:.1f} seconds")
    
    print(f"\n📋 Detailed Report:")
    for key, result in results.items():
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        print(f"  {key}. {result['name']}: {status}")
    
    if successful == total:
        print(f"\n🎉 ALL MODULES COMPLETED SUCCESSFULLY!")
    else:
        print(f"\n⚠️  {total - successful} modules had issues")

def interactive_mode():
    """
    Interactive mode - continuously get input from user
    """
    print("\n🔄 INTERACTIVE MODE STARTED")
    print("Type 'q' to exit")
    
    while True:
        try:
            display_menu()
            choice = input("\nMake your choice: ").strip().lower()
            
            if choice == 'q':
                print("\n👋 Goodbye!")
                break
            elif choice == 'h':
                show_help()
                input("\nPress Enter to continue...")
            elif choice == 'a':
                run_all_modules()
                input("\nPress Enter to continue...")
            elif choice in modules:
                run_module(choice)
                input("\nPress Enter to continue...")
            else:
                print(f"❌ Invalid choice: {choice}")
                input("Press Enter to continue...")
                
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Program stopped by user")
            break
        except EOFError:
            print(f"\n\n👋 Program terminated")
            break

def main():
    """
    Main function
    """
    print("🚀 LangChain Demo Program Starting...")
    
    # Environment check
    if not check_environment():
        print("\n❌ Environment checks failed. Program terminating.")
        sys.exit(1)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg == 'all':
            run_all_modules()
        elif arg == 'help' or arg == 'h':
            show_help()
        elif arg in modules:
            run_module(arg)
        else:
            print(f"❌ Invalid argument: {arg}")
            print("Usage: python main_demo.py [1-7|all|help]")
            sys.exit(1)
    else:
        # Interactive mode
        interactive_mode()

if __name__ == "__main__":
    main()