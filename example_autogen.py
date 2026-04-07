"""
Example: Using AutoGen for Multi-Agent Research

This script demonstrates how to use the AutoGen-based multi-agent research system.

Usage:
    python example_autogen.py
"""

import os
import yaml
import logging
from dotenv import load_dotenv
from src.autogen_orchestrator import AutoGenOrchestrator


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/example.log")
        ]
    )


def load_config():
    """Load configuration from config.yaml."""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def print_separator(title: str = ""):
    """Print a visual separator."""
    if title:
        print(f"\n{'=' * 70}")
        print(f"{title:^70}")
        print(f"{'=' * 70}\n")
    else:
        print(f"{'=' * 70}\n")


def run_single_query():
    """
    Example 1: Run a single research query.
    
    This is the simplest way to use the system.
    """
    print_separator("Example 1: Single Research Query")
    
    # Load environment and config
    load_dotenv()
    config = load_config()
    
    # Create orchestrator
    orchestrator = AutoGenOrchestrator(config)
    
    # Define your research query
    query = "What are the key principles of accessible user interface design?"
    
    print(f"Query: {query}\n")
    print("Processing... (this may take 1-2 minutes)\n")
    
    # Process the query
    result = orchestrator.process_query(query, max_rounds=20)
    
    # Display results
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    print_separator("Final Response")
    print(result['response'])
    
    print_separator("Metadata")
    print(f"Messages exchanged: {result['metadata']['num_messages']}")
    print(f"Sources gathered: {result['metadata']['num_sources']}")
    print(f"Agents involved: {', '.join(result['metadata']['agents_involved'])}")


def run_multiple_queries():
    """
    Example 2: Process multiple queries in sequence.
    
    Shows how to reuse the orchestrator for multiple queries.
    """
    print_separator("Example 2: Multiple Research Queries")
    
    load_dotenv()
    config = load_config()
    
    # Create orchestrator once
    orchestrator = AutoGenOrchestrator(config)
    
    # List of queries to process
    queries = [
        "What is cognitive load theory in HCI?",
        "How does eye tracking improve user experience?",
        "What are best practices for mobile UI design?",
    ]
    
    results = []
    
    for i, query in enumerate(queries, 1):
        print(f"\n[Query {i}/{len(queries)}] {query}")
        print("-" * 70)
        
        result = orchestrator.process_query(query, max_rounds=15)
        results.append(result)
        
        # Print brief summary
        if "error" not in result:
            response_preview = result['response'][:200] + "..."
            print(f"Response preview: {response_preview}\n")
    
    print_separator("Summary")
    print(f"Processed {len(queries)} queries successfully")
    
    return results


def inspect_conversation():
    """
    Example 3: Inspect the conversation history.
    
    Shows how to access and examine the agent-to-agent conversation.
    """
    print_separator("Example 3: Inspecting Conversation History")
    
    load_dotenv()
    config = load_config()
    
    orchestrator = AutoGenOrchestrator(config)
    
    query = "What is the difference between usability and user experience?"
    
    print(f"Query: {query}\n")
    result = orchestrator.process_query(query, max_rounds=20)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    print_separator("Conversation Flow")
    
    # Display each message in the conversation
    for i, msg in enumerate(result['conversation_history'], 1):
        agent = msg.get('source', msg.get('name', 'Unknown'))
        content = msg.get('content', '')
        
        # Truncate long messages for readability
        if len(content) > 300:
            content = content[:300] + "...[truncated]"
        
        print(f"[{i}] {agent}:")
        print(f"    {content}\n")


def view_workflow():
    """
    Example 4: Visualize the workflow.
    
    Shows the structure of the multi-agent system.
    """
    print_separator("Example 4: Workflow Visualization")
    
    load_dotenv()
    config = load_config()
    
    orchestrator = AutoGenOrchestrator(config)
    
    # Print workflow diagram
    print(orchestrator.visualize_workflow())
    
    # Print agent descriptions
    print_separator("Agent Descriptions")
    for agent_name, description in orchestrator.get_agent_descriptions().items():
        print(f"• {agent_name}: {description}")


def check_setup():
    """
    Check if the system is properly configured.
    
    Verifies API keys and dependencies.
    """
    print_separator("Setup Check")
    
    load_dotenv()
    
    checks = {
        "Environment file (.env)": os.path.exists(".env"),
        "Config file (config.yaml)": os.path.exists("config.yaml"),
        "Logs directory": os.path.exists("logs"),
        "GROQ_API_KEY": bool(os.getenv("GROQ_API_KEY")),
        "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        "TAVILY_API_KEY": bool(os.getenv("TAVILY_API_KEY")),
    }
    
    print("Configuration Status:\n")
    
    all_good = True
    for check, status in checks.items():
        status_str = "✓ OK" if status else "✗ MISSING"
        print(f"  {check:.<40} {status_str}")
        
        if not status and "API_KEY" in check:
            all_good = False
    
    print("\nRequired API Keys:")
    print("  - At least one LLM key (GROQ_API_KEY or OPENAI_API_KEY)")
    print("  - At least one search key (TAVILY_API_KEY recommended)")
    
    if not checks["GROQ_API_KEY"] and not checks["OPENAI_API_KEY"]:
        print("\n⚠ Warning: No LLM API key found. Please add one to .env")
    
    if not checks["TAVILY_API_KEY"]:
        print("\n⚠ Warning: No search API key found. Research capabilities will be limited.")
    
    print()


def main():
    """
    Main function - run all examples or choose one.
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    setup_logging()
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║           AutoGen Multi-Agent Research System Examples               ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # First, check setup
    check_setup()
    
    # Menu
    print("\nChoose an example to run:\n")
    print("  1. Single research query (simple)")
    print("  2. Multiple queries in sequence")
    print("  3. Inspect conversation history")
    print("  4. View workflow diagram")
    print("  5. Check setup")
    print("  0. Exit\n")
    
    try:
        choice = input("Enter your choice (0-5): ").strip()
        
        if choice == "1":
            run_single_query()
        elif choice == "2":
            run_multiple_queries()
        elif choice == "3":
            inspect_conversation()
        elif choice == "4":
            view_workflow()
        elif choice == "5":
            check_setup()
        elif choice == "0":
            print("Goodbye!")
        else:
            print("Invalid choice. Please run again and select 0-5.")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nError: {e}")
        logging.exception("Error in main")


if __name__ == "__main__":
    main()

