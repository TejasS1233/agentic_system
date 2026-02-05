import sys
import os

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from architecture.llm_manager import get_groq_manager


def test_initialization():
    print("Testing initialization...")
    try:
        mgr = get_groq_manager()
        print(f"Manager initialized: {mgr.provider} / {mgr.model}")
        return mgr
    except Exception as e:
        print(f"Initialization failed: {e}")
        return None


def test_completion(mgr):
    print("\nTesting sample completion...")
    result = mgr.generate_text("Say hello in one word.")
    if result.get("content"):
        print(f"Response: {result['content']}")
        return True
    else:
        print(f"Error: {result.get('error')}")
        return False


if __name__ == "__main__":
    mgr = test_initialization()
    if mgr:
        print("Initialization Test Passed")
        if test_completion(mgr):
            print("Completion Test Passed")
        else:
            print("Completion Test Failed")
    else:
        print("Initialization Test Failed")
