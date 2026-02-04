
import random
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from architecture.intent_classifier import IntentClassifier

# =============================================================================
# VOCABULARY LISTS
# =============================================================================

DOMAINS = ["math", "text", "file", "web", "data", "system", "visualization", "conversion", "search", "validation"]

VOCAB = {
    "math": {
        "verbs": ["calculate", "compute", "find", "determine", "evaluate", "solve", "add", "sum", "multiply", "divide", "subtract"],
        "nouns": ["factorial", "sum", "average", "mean", "median", "mode", "standard deviation", "square root", "logarithm", "integral", "derivative", "prime factors", "gcd", "lcm", "fibonacci sequence"],
        "contexts": ["of the numbers", "for these values", "using the input", "of the series", "math problem"],
        "adjectives": ["mathematical", "arithmetic", "geometric", "numeric", "complex"]
    },
    "text": {
        "verbs": ["capitalize", "lowercase", "uppercase", "trim", "strip", "split", "join", "replace", "reverse", "count", "check"],
        "nouns": ["words", "characters", "sentences", "strings", "lines", "paragraphs", "palindrome", "anagram", "tokens", "spelling", "grammar"],
        "contexts": ["in the text", "from the string", "ignoring case", "alphabetically", "using regex"],
        "adjectives": ["textual", "lexical", "linguistic", "string"]
    },
    "file": {
        "verbs": ["read", "write", "delete", "create", "copy", "move", "rename", "list", "compress", "archive", "zip", "unzip", "find"],
        "nouns": ["files", "directories", "folders", "paths", "content", "backup", "logs", "archives", "permissions", "metadata"],
        "contexts": ["recursively", "in the current directory", "to the destination", "on the disk", "forcefully"],
        "adjectives": ["hidden", "read-only", "executable", "temporary", "nested"]
    },
    "web": {
        "verbs": ["download", "fetch", "scrape", "crawl", "get", "post", "request", "ping", "query"],
        "nouns": ["url", "website", "webpage", "html", "api", "endpoint", "json response", "headers", "cookies", "server"],
        "contexts": ["from the internet", "via http", "asynchronously", "using authenticated session", "from the remote server"],
        "adjectives": ["remote", "online", "external", "secure"]
    },
    "data": {
        "verbs": ["parse", "analyze", "process", "load", "save", "export", "import", "serialize", "deserialize", "query"],
        "nouns": ["json", "csv", "xml", "yaml", "database", "schema", "records", "rows", "columns", "dataframe", "dataset"],
        "contexts": ["into a dictionary", "from the file", "to sql", "using pandas", "without errors"],
        "adjectives": ["structured", "tabular", "raw", "nested", "relational"]
    },
    "system": {
        "verbs": ["run", "execute", "launch", "kill", "terminate", "monitor", "check", "set", "get"],
        "nouns": ["process", "command", "script", "shell", "terminal", "environment variables", "cpu usage", "memory", "disk space", "system time", "permissions"],
        "contexts": ["in the background", "as root", "silently", "with arguments", "on startup"],
        "adjectives": ["system", "background", "blocking", "concurrent", "OS-level"]
    },
    "visualization": {
        "verbs": ["plot", "graph", "chart", "visualize", "draw", "render", "show", "display", "generate"],
        "nouns": ["histogram", "barchart", "scatterplot", "line chart", "pie chart", "heatmap", "diagram", "figure", "image", "picture"],
        "contexts": ["of the data", "using matplotlib", "in 3d", "with colors", "for the report"],
        "adjectives": ["interactive", "static", "visual", "graphical"]
    },
    "conversion": {
        "verbs": ["convert", "transform", "transcode", "encode", "decode", "format", "render"],
        "nouns": ["pdf", "image", "audio", "video", "markdown", "html", "date format", "currency", "units", "celsius to fahrenheit", "json to yaml"],
        "contexts": ["to new format", "losslessly", "for web", "using codec", "to standard output"],
        "adjectives": ["converted", "formatted", "encoded", "binary"]
    },
    "search": {
        "verbs": ["search", "find", "locate", "lookup", "query", "scan"],
        "nouns": ["results", "matches", "keywords", "patterns", "index", "database", "documentation"],
        "contexts": ["in the database", "using fuzzy match", "quickly", "by relevance", "across all files"],
        "adjectives": ["relevant", "matching", "indexed", "searchable"]
    },
    "validation": {
        "verbs": ["validate", "verify", "check", "audit", "ensure", "confirm", "test"],
        "nouns": ["email", "input", "schema", "format", "integrity", "checksum", "credentials", "constraints", "types"],
        "contexts": ["against the rules", "strictly", "before processing", "recursively", "securely"],
        "adjectives": ["valid", "invalid", "compliant", "secure", "trusted"]
    }
}

# Ambiguous Nouns designed to trip up the classifier if it ignores verbs
AMBIGUOUS_NOUNS = {
    "file": ["data", "text", "script", "image", "logs"],
    "math": ["value", "numbers", "result", "count"],
    "text": ["input", "content", "message"],
    "web": ["file", "data", "content", "resource"],
}

TEMPLATES = [
    "{verb} the {noun}",
    "{verb} {noun}",
    "I want to {verb} {noun}",
    "can you {verb} the {adj} {noun}",
    "{verb} {noun} {context}",
    "tool to {verb} {noun} {context}",
    "please {verb} my {noun}",
    "{verb} {adj} {noun} right now",
]

def generate_test_case(domain):
    """Generate a single random test case for a domain."""
    data = VOCAB[domain]
    
    # Randomly pick components
    verb = random.choice(data["verbs"])
    noun = random.choice(data["nouns"])
    adj = random.choice(data["adjectives"])
    context = random.choice(data["contexts"])
    template = random.choice(TEMPLATES)
    
    # 20% chance to swap noun with an ambiguous one to test verb strength
    if random.random() < 0.2 and domain in AMBIGUOUS_NOUNS:
         noun = random.choice(AMBIGUOUS_NOUNS[domain])
    
    query = template.format(verb=verb, noun=noun, adj=adj, context=context)
    return query, domain

def run_stress_test(num_cases=500):
    classifier = IntentClassifier()
    
    classification_counts = {d: 0 for d in DOMAINS}
    correct_counts = {d: 0 for d in DOMAINS}
    failures = []
    
    print(f"Generating and running {num_cases} synthetic test cases...")
    print("-" * 60)
    
    start_time = time.time()
    
    for i in range(num_cases):
        target_domain = random.choice(DOMAINS)
        query, expected = generate_test_case(target_domain)
        
        # Run classifier
        predicted, method, conf = classifier.classify(query)
        
        classification_counts[target_domain] += 1
        
        # Soft matching for certain overlaps (e.g., search vs web, or file vs system)
        is_correct = (predicted == expected)
        
        if is_correct:
            correct_counts[target_domain] += 1
        else:
            failures.append({
                "query": query,
                "expected": expected,
                "predicted": predicted,
                "method": method,
                "conf": conf
            })
            
    duration = time.time() - start_time
    total_acc = sum(correct_counts.values()) / num_cases * 100
    
    print(f"\nResults (taking {duration:.2f}s):")
    print(f"Overall Accuracy: {total_acc:.1f}%")
    print("-" * 40)
    
    print(f"{'Domain':<15} | {'Acc':<6} | {'Count'}")
    print("-" * 35)
    for d in DOMAINS:
        acc = 0 if classification_counts[d] == 0 else (correct_counts[d] / classification_counts[d] * 100)
        print(f"{d:<15} | {acc:5.1f}% | {classification_counts[d]}")
        
    print("-" * 60)
    print("Top 20 Failures:")
    for f in failures[:20]:
        print(f"Exp: {f['expected']:<12} | Pred: {f['predicted']:<12} | '{f['query']}' ({f['method']})")
        
    return total_acc

if __name__ == "__main__":
    acc = run_stress_test(500)
    sys.exit(0 if acc > 95.0 else 1)
