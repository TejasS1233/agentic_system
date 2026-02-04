"""
Comprehensive test suite for the IntentClassifier and domain validation.

Run with: python -m pytest tests/test_intent_classifier.py -v
Or:       python tests/test_intent_classifier.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from architecture.intent_classifier import (
    IntentClassifier, 
    validate_domain, 
    get_allowed_domains,
    get_domain_prompt_string,
    ALLOWED_DOMAINS
)


class TestDomainValidation:
    """Tests for the validate_domain function."""
    
    def test_direct_domain_match(self):
        """All canonical domains should match directly."""
        for domain in get_allowed_domains():
            result, valid = validate_domain(domain)
            assert result == domain, f"Direct match failed for {domain}"
            assert valid == True
    
    def test_case_insensitive(self):
        """Domain matching should be case insensitive."""
        test_cases = [("MATH", "math"), ("Math", "math"), ("WEB", "web")]
        for input_domain, expected in test_cases:
            result, valid = validate_domain(input_domain)
            assert result == expected, f"Case insensitive match failed for {input_domain}"
            assert valid == True
    
    def test_alias_mapping(self):
        """Aliases should map to canonical domains."""
        alias_tests = [
            ("mathematics", "math"),
            ("string", "text"),
            ("filesystem", "file"),
            ("http", "web"),
            ("api", "web"),
            ("json", "data"),
            ("chart", "visualization"),
            ("encode", "conversion"),
            ("verify", "validation"),
        ]
        for alias, expected in alias_tests:
            result, valid = validate_domain(alias)
            assert result == expected, f"Alias {alias} should map to {expected}, got {result}"
            assert valid == True
    
    def test_fuzzy_matching(self):
        """Fuzzy matching should work for typos/similar words."""
        fuzzy_tests = [
            ("maths", "math"),
            ("texts", "text"),
        ]
        for input_domain, expected in fuzzy_tests:
            result, valid = validate_domain(input_domain)
            assert result == expected, f"Fuzzy match failed: {input_domain} should be {expected}"
    
    def test_invalid_domain_fallback(self):
        """Invalid domains should fall back to 'system'."""
        invalid_tests = ["audio", "neural", "unknown_domain", ""]
        for invalid in invalid_tests:
            result, valid = validate_domain(invalid)
            assert result == "system", f"Invalid domain {invalid} should fallback to 'system'"
            assert valid == False


class TestIntentClassifier:
    """Tests for the IntentClassifier."""
    
    @classmethod
    def setup_class(cls):
        """Set up classifier for all tests."""
        registry_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'workspace/tools/registry.json'
        )
        cls.classifier = IntentClassifier(registry_path)
    
    def test_ambiguous_queries(self):
        """Test queries where the same word has different meanings based on context."""
        test_cases = [
            ("sum of two numbers", "math"),
            ("sum of files in a directory", "file"),
            ("sum total of API responses", "web"),
            ("count words in text", "text"),
            ("count files in folder", "file"),
            ("count API calls", "web"),
        ]
        self._run_tests(test_cases, "Ambiguous queries")
    
    def test_single_word_queries(self):
        """Test minimal, single-word queries."""
        test_cases = [
            ("factorial", "math"),
            ("palindrome", "text"),
            ("download", "web"),
            ("plot", "visualization"),
            ("validate", "validation"),
        ]
        self._run_tests(test_cases, "Single-word queries")
    
    def test_unusual_phrasing(self):
        """Test natural language requests."""
        test_cases = [
            ("i want to add two numbers together", "math"),
            ("give me the square root of something", "math"),
            ("help me find files", "file"),
            ("make a chart", "visualization"),
            ("is this a valid email", "validation"),
            ("fetch data from URL", "web"),
        ]
        self._run_tests(test_cases, "Unusual phrasing")
    
    def test_multi_domain_queries(self):
        """Test queries that span multiple domains - should pick primary."""
        test_cases = [
            ("read JSON file from disk", "file"),      # file operation, data type
            ("download and parse CSV", "web"),         # web + data
            ("plot numbers from file", "visualization"), # viz + file + math
        ]
        self._run_tests(test_cases, "Multi-domain queries")
    
    def test_validation_edge_cases(self):
        """Test validation-related edge cases."""
        test_cases = [
            ("check if NOT a number", "validation"),
            ("verify string is not empty", "validation"),
            ("validate an email address format", "validation"),
        ]
        self._run_tests(test_cases, "Validation edge cases")
    
    def test_real_world_requests(self):
        """Test realistic tool creation requests."""
        test_cases = [
            ("tool to check if my input is even or odd", "math"),
            ("a tool to search the web and return results", "web"),
            ("calculate the factorial of a given number", "math"),
            ("check if a string is a palindrome", "text"),
            ("list all files in current directory", "file"),
            ("convert temperature from celsius to fahrenheit", "conversion"),
            ("generate a bar chart from data", "visualization"),
            ("compress a file using gzip", "file"),
            ("send an HTTP POST request", "web"),
            ("execute shell command", "system"),
        ]
        self._run_tests(test_cases, "Real-world requests")
    
    def test_extended_edge_cases(self):
        """Additional edge cases for comprehensive coverage."""
        test_cases = [
            ("multiply two integers", "math"),
            ("concatenate strings", "text"),
            ("merge PDF files", "file"),
            ("call REST API", "web"),
            ("serialize object to JSON", "data"),
            ("create histogram", "visualization"),
            ("kilometers to miles", "conversion"),
            ("websocket connection", "web"),
            ("parse XML document", "data"),
            ("draw scatter plot", "visualization"),
            ("base64 encode", "conversion"),
            ("check null values", "validation"),
            ("copy directory", "file"),
            ("HTTP GET request", "web"),
            ("YAML to JSON", "data"),
            ("render pie chart", "visualization"),
            ("hex to decimal", "conversion"),
            ("run subprocess", "system"),
        ]
        self._run_tests(test_cases, "Extended edge cases")
    
    def _run_tests(self, test_cases, category_name):
        """Helper to run a batch of tests."""
        passed = 0
        failed = []
        for query, expected in test_cases:
            domain, method, conf = self.classifier.classify(query)
            if domain == expected:
                passed += 1
            else:
                failed.append((query, expected, domain, method))
        
        if failed:
            fail_msg = "\n".join([f"  '{q}': expected {e}, got {g} (via {m})" 
                                  for q, e, g, m in failed])
            assert False, f"{category_name}: {passed}/{len(test_cases)} passed\nFailed:\n{fail_msg}"


class TestToolsmithIntegration:
    """Tests for Toolsmith integration with domain validation."""
    
    def test_canonical_domains_count(self):
        """Should have exactly 10 canonical domains."""
        domains = get_allowed_domains()
        assert len(domains) == 10, f"Expected 10 domains, got {len(domains)}"
    
    def test_prompt_string_format(self):
        """Prompt string should be pipe-separated."""
        prompt = get_domain_prompt_string()
        assert "|" in prompt
        parts = prompt.split("|")
        assert len(parts) == 10
    
    def test_all_domains_in_prompt(self):
        """All canonical domains should appear in prompt string."""
        prompt = get_domain_prompt_string()
        for domain in get_allowed_domains():
            assert domain in prompt, f"Domain {domain} missing from prompt string"


def run_all_tests():
    """Run all tests and print summary."""
    print("=" * 70)
    print("INTENT CLASSIFIER TEST SUITE")
    print("=" * 70)
    
    # Count results
    total_passed = 0
    total_failed = 0
    
    # Domain validation tests
    print("\n1. DOMAIN VALIDATION TESTS")
    print("-" * 40)
    
    test_val = TestDomainValidation()
    for method_name in dir(test_val):
        if method_name.startswith('test_'):
            try:
                getattr(test_val, method_name)()
                print(f"   ✓ {method_name}")
                total_passed += 1
            except AssertionError as e:
                print(f"   ✗ {method_name}: {e}")
                total_failed += 1
    
    # Intent classifier tests
    print("\n2. INTENT CLASSIFIER TESTS")
    print("-" * 40)
    
    test_cls = TestIntentClassifier()
    test_cls.setup_class()
    for method_name in dir(test_cls):
        if method_name.startswith('test_'):
            try:
                getattr(test_cls, method_name)()
                print(f"   ✓ {method_name}")
                total_passed += 1
            except AssertionError as e:
                print(f"   ✗ {method_name}")
                print(f"      {str(e)[:100]}")
                total_failed += 1
    
    # Integration tests
    print("\n3. TOOLSMITH INTEGRATION TESTS")
    print("-" * 40)
    
    test_int = TestToolsmithIntegration()
    for method_name in dir(test_int):
        if method_name.startswith('test_'):
            try:
                getattr(test_int, method_name)()
                print(f"   ✓ {method_name}")
                total_passed += 1
            except AssertionError as e:
                print(f"   ✗ {method_name}: {e}")
                total_failed += 1
    
    # Summary
    print("\n" + "=" * 70)
    print(f"TOTAL: {total_passed} passed, {total_failed} failed")
    if total_failed == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ Some tests failed")
    print("=" * 70)
    
    return total_failed == 0


if __name__ == "__main__":
    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore")
    
    success = run_all_tests()
    sys.exit(0 if success else 1)
