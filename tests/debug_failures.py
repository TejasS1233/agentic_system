
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from architecture.intent_classifier import IntentClassifier, get_nlp

def debug_failures():
    registry_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'workspace/tools/registry.json'
    )
    classifier = IntentClassifier(registry_path)
    nlp = get_nlp()
    
    failures = [
        "check if number is prime",
        "calculate the length of the url",
        "compress the json data",
        "show me a picture of the data",
        "generate a random password",
        "ping the server",
        "read the email header",
        "write a script to calculate sum",
        "check if the server is up",
        "normalize the vectors"
    ]
    
    print("Debug analysis of failed cases:")
    print("-" * 60)
    
    for query in failures:
        print(f"\nQuery: '{query}'")
        
        # 1. SpaCy Analysis
        doc = nlp(query.lower())
        print("  Tokens:", [(t.text, t.pos_, t.dep_, t.head.text) for t in doc])
        
        verb, obj = classifier._extract_verb_object(query)
        print(f"  Extracted: Verb='{verb}', Obj='{obj}'")
        
        # 2. Current Classification
        domain, method, conf = classifier.classify(query)
        print(f"  Result: {domain} ({method}, {conf})")

if __name__ == "__main__":
    debug_failures()
