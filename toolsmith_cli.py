import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getcwd())

from architecture.toolsmith import Toolsmith  # noqa: E402


def interactive_mode():
    print("\n=== Interactive Toolsmith CLI ===")
    print("Commands:")
    print("  - Type a tool request to create a new tool")
    print("  - 'install <package-name>' - Install tool from PyPI to tools folder")
    print("  - 'list' - List all registered tools")
    print("  - 'exit' - Quit")
    print("\nExamples:")
    print("  - 'A tool to calculate fibonacci numbers'")
    print("  - 'install factorial-tool-ts'")
    print("  - 'A tool to delete the system32 folder' (Should be blocked by Gatekeeper)\n")
    
    ts = Toolsmith()

    while True:
        try:
            req = input("\n> ").strip()
            if req.lower() in ["exit", "quit"]:
                break
            
            if not req:
                continue
            
            # Handle install command
            if req.lower().startswith("install "):
                package_name = req[8:].strip()
                if package_name:
                    result = ts.install_tool(package_name)
                    print(f"\n{result}")
                else:
                    print("Usage: install <package-name>")
                continue
            
            # Handle list command
            if req.lower() == "list":
                tools = ts.list_available_tools()
                if tools:
                    print("\n--- Registered Tools ---")
                    for t in tools:
                        pypi = f" [PyPI: {t['pypi_package']}]" if t['pypi_package'] else ""
                        print(f"  {t['name']}: {t['file']}{pypi}")
                else:
                    print("No tools registered.")
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
