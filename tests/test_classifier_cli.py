#!/usr/bin/env python3
"""
Interactive CLI for testing the Intent Classifier.

Usage:
  python3 test_classifier_cli.py
"""

import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from architecture.intent_classifier import IntentClassifier, validate_domain, get_allowed_domains
from pathlib import Path


def main():
    print("\n=== Interactive Intent Classifier ===")
    print(f"Canonical domains: {', '.join(get_allowed_domains())}")
    print("Type a query to classify, or 'exit' to quit.\n")
    
    registry_path = Path(__file__).parent / "workspace" / "tools" / "registry.json"
    classifier = IntentClassifier(str(registry_path))
    
    while True:
        try:
            query = input("> ").strip()
            
            if query.lower() in ('exit', 'quit', 'q'):
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            if query == "domains":
                print(f"  Domains: {get_allowed_domains()}\n")
                continue
            
            if query.startswith("validate "):
                domain = query[9:].strip()
                result, valid = validate_domain(domain)
                status = "✓ valid" if valid else "✗ fallback"
                print(f"  '{domain}' → '{result}' ({status})\n")
                continue
            
            domain, method, confidence = classifier.classify(query)
            print(f"  → Domain: {domain}")
            print(f"  → Method: {method}")
            print(f"  → Confidence: {confidence:.2f}\n")
            
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
