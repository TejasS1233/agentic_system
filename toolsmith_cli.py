import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getcwd())

from architecture.toolsmith import Toolsmith  # noqa: E402


def interactive_mode():
    print("\n=== Interactive Toolsmith CLI ===")
    print("Type your tool request below.")
    print("Examples:")
    print("  - 'A tool to calculate fibonacci numbers'")
    print("  - 'A tool to delete the system32 folder' (Should be blocked)")
    print("Type 'exit' to code.\n")

    ts = Toolsmith()

    while True:
        try:
            req = input("\n> ")
            if req.lower() in ["exit", "quit"]:
                break

            if not req.strip():
                continue

            print("\n[Thinking] Sending request to Gemini 2.5 Flash...")
            result = ts.create_tool(req)

            print("\n----- RESULT -----")
            print(result)
            print("------------------")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    interactive_mode()
