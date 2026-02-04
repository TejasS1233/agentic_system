#!/usr/bin/env python3
"""
Generate and test 1000 hard classification cases for the Intent Classifier.

Hard cases include:
- Ambiguous phrases (e.g., "sum of files" vs "sum of numbers")
- Domain boundary cases
- Typos and variations
- Conflicting signals
- Edge cases and unusual phrasing

Usage:
  uv run test_hard_cases.py
"""

import sys
import os
import random
from collections import defaultdict

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from architecture.intent_classifier import IntentClassifier, get_allowed_domains
from pathlib import Path


# =============================================================================
# Hard Case Templates by Domain
# Each tuple is (template, expected_domain)
# =============================================================================

HARD_CASES = []

# -----------------------------------------------------------------------------
# MATH - Hard cases
# -----------------------------------------------------------------------------
MATH_CASES = [
    # Core math operations
    ("calculate the factorial of {n}", "math"),
    ("what is the sum of {n} and {m}", "math"),
    ("find the average of these numbers", "math"),
    ("compute the square root of {n}", "math"),
    ("check if {n} is prime", "math"),
    ("is {n} an even number", "math"),
    ("determine if {n} is odd", "math"),
    ("multiply {n} by {m}", "math"),
    ("divide {n} by {m}", "math"),
    ("add {n} to {m}", "math"),
    ("subtract {m} from {n}", "math"),
    ("what's the remainder of {n} divided by {m}", "math"),
    ("find the modulo of {n} and {m}", "math"),
    ("compute the GCD of {n} and {m}", "math"),
    ("find LCM of {n} and {m}", "math"),
    ("evaluate this arithmetic expression", "math"),
    ("solve the equation x + {n} = {m}", "math"),
    ("calculate the mean of the dataset", "math"),
    ("find the median value", "math"),
    ("compute standard deviation", "math"),
    ("calculate the variance", "math"),
    ("find the nth fibonacci number", "math"),
    ("generate fibonacci sequence up to {n}", "math"),
    ("calculate the integral", "math"),
    ("find the derivative", "math"),
    ("normalize this vector", "math"),
    ("calculate the dot product", "math"),
    ("find the matrix determinant", "math"),
    ("solve this system of equations", "math"),
    ("what is the logarithm of {n}", "math"),
    ("calculate log base 2 of {n}", "math"),
    ("find the exponential of {n}", "math"),
    ("compute e to the power of {n}", "math"),
    ("calculate sine of {n} degrees", "math"),
    ("find cosine of angle theta", "math"),
    ("what is tangent of {n}", "math"),
    ("calculate percentage of {n} out of {m}", "math"),
    ("find the ratio between {n} and {m}", "math"),
    ("compute the absolute value", "math"),
    ("round {n} to two decimal places", "math"),
    ("floor of {n}", "math"),
    ("ceiling of {n}", "math"),
    ("is this number negative", "math"),
    ("check if the value is positive", "math"),
    ("calculate compound interest", "math"),
    ("find simple interest", "math"),
    ("compute the hypotenuse", "math"),
    ("pythagorean theorem calculation", "math"),
    ("evaluate the polynomial", "math"),
    ("find the roots of the quadratic equation", "math"),
]

# -----------------------------------------------------------------------------
# TEXT - Hard cases
# -----------------------------------------------------------------------------
TEXT_CASES = [
    ("check if the string is a palindrome", "text"),
    ("reverse the text", "text"),
    ("convert text to uppercase", "text"),
    ("make this lowercase", "text"),
    ("capitalize the first letter", "text"),
    ("count the words in this sentence", "text"),
    ("split the string by delimiter", "text"),
    ("join these words with spaces", "text"),
    ("trim whitespace from the string", "text"),
    ("strip leading and trailing spaces", "text"),
    ("replace all occurrences of 'foo' with 'bar'", "text"),
    ("find substring in text", "text"),
    ("check if string contains word", "text"),
    ("extract numbers from the string", "text"),
    ("remove all punctuation", "text"),
    ("count character frequency", "text"),
    ("find the longest word", "text"),
    ("check if text is empty", "text"),
    ("generate random password", "text"),
    ("create a UUID string", "text"),
    ("slugify the title", "text"),
    ("camelCase to snake_case", "text"),
    ("convert snake_case to camelCase", "text"),
    ("check for anagram", "text"),
    ("are these two strings anagrams", "text"),
    ("sort the characters alphabetically", "text"),
    ("find duplicate characters", "text"),
    ("remove duplicate words", "text"),
    ("count sentences in paragraph", "text"),
    ("split text into sentences", "text"),
    ("extract email addresses from text", "text"),
    ("find all URLs in string", "text"),
    ("mask sensitive data in text", "text"),
    ("truncate string to {n} characters", "text"),
    ("pad the string with zeros", "text"),
    ("left pad to length {n}", "text"),
    ("right align the text", "text"),
    ("center the text", "text"),
    ("wrap text at {n} columns", "text"),
    ("check spelling of the text", "text"),
    ("check grammar in the sentence", "text"),
    ("tokenize this paragraph", "text"),
    ("lemmatize the words", "text"),
    ("stem the words in text", "text"),
    ("find part of speech tags", "text"),
    ("extract named entities", "text"),
    ("sentiment analysis of text", "text"),
    ("is this string alphanumeric", "text"),
    ("check if string starts with prefix", "text"),
    ("does the string end with suffix", "text"),
]

# -----------------------------------------------------------------------------
# FILE - Hard cases
# -----------------------------------------------------------------------------
FILE_CASES = [
    ("list all files in the directory", "file"),
    ("read contents of the file", "file"),
    ("write data to file.txt", "file"),
    ("delete the temporary file", "file"),
    ("rename file.old to file.new", "file"),
    ("move file to another folder", "file"),
    ("copy the file to backup", "file"),
    ("create a new directory", "file"),
    ("check if file exists", "file"),
    ("get file size in bytes", "file"),
    ("find file modification time", "file"),
    ("list subdirectories recursively", "file"),
    ("count files in folder", "file"),
    ("sum the file sizes in directory", "file"),
    ("find all .py files", "file"),
    ("search for files matching pattern", "file"),
    ("compress files into archive", "file"),
    ("zip the directory", "file"),
    ("unzip the archive", "file"),
    ("extract tar.gz file", "file"),
    ("create a gzip archive", "file"),
    ("read the log file", "file"),
    ("append to the log", "file"),
    ("find empty files", "file"),
    ("delete empty directories", "file"),
    ("get file permissions", "file"),
    ("change file permissions", "file"),
    ("set file as read-only", "file"),
    ("find duplicate files", "file"),
    ("compare two files", "file"),
    ("diff between files", "file"),
    ("merge multiple files", "file"),
    ("split file into chunks", "file"),
    ("read binary file", "file"),
    ("write bytes to file", "file"),
    ("seek to position in file", "file"),
    ("truncate file at offset", "file"),
    ("find the largest file", "file"),
    ("sort files by size", "file"),
    ("organize files by extension", "file"),
    ("move images to photos folder", "file"),
    ("backup the config files", "file"),
    ("restore from backup", "file"),
    ("sync directories", "file"),
    ("watch directory for changes", "file"),
    ("monitor file modifications", "file"),
    ("read the first {n} lines", "file"),
    ("tail the log file", "file"),
    ("head of the file", "file"),
    ("count lines in file", "file"),
]

# -----------------------------------------------------------------------------
# WEB - Hard cases
# -----------------------------------------------------------------------------
WEB_CASES = [
    ("fetch URL content", "web"),
    ("download file from URL", "web"),
    ("make HTTP GET request", "web"),
    ("send POST request to API", "web"),
    ("call the REST endpoint", "web"),
    ("scrape the webpage", "web"),
    ("crawl the website", "web"),
    ("ping the server", "web"),
    ("check if URL is reachable", "web"),
    ("get the response headers", "web"),
    ("send JSON payload", "web"),
    ("upload file to server", "web"),
    ("download the CSV from URL", "web"),
    ("fetch API data", "web"),
    ("make authenticated request", "web"),
    ("send request with headers", "web"),
    ("follow redirects", "web"),
    ("handle cookies", "web"),
    ("manage sessions", "web"),
    ("websocket connection", "web"),
    ("establish socket connection", "web"),
    ("stream response data", "web"),
    ("download in chunks", "web"),
    ("check website status", "web"),
    ("is the server online", "web"),
    ("get webpage title", "web"),
    ("extract links from page", "web"),
    ("find images on webpage", "web"),
    ("submit form data", "web"),
    ("grab the webpage content", "web"),
    ("fetch and save image", "web"),
    ("download PDF from link", "web"),
    ("search the web for query", "web"),
    ("google search results", "web"),
    ("query external API", "web"),
    ("rate limit the requests", "web"),
    ("retry failed request", "web"),
    ("handle 404 error", "web"),
    ("cache the response", "web"),
    ("proxy the request", "web"),
    ("check SSL certificate", "web"),
    ("verify HTTPS connection", "web"),
    ("get DNS records", "web"),
    ("resolve hostname", "web"),
    ("trace route to server", "web"),
    ("measure response time", "web"),
    ("benchmark the API", "web"),
    ("test API endpoint", "web"),
    ("health check the service", "web"),
    ("monitor uptime", "web"),
]

# -----------------------------------------------------------------------------
# DATA - Hard cases
# -----------------------------------------------------------------------------
DATA_CASES = [
    ("parse the JSON object", "data"),
    ("convert JSON to dict", "data"),
    ("serialize to JSON", "data"),
    ("read CSV data", "data"),
    ("parse CSV with headers", "data"),
    ("write data to CSV", "data"),
    ("parse XML document", "data"),
    ("extract XML elements", "data"),
    ("convert XML to JSON", "data"),
    ("load YAML config", "data"),
    ("parse YAML file", "data"),
    ("dump to YAML format", "data"),
    ("query the database", "data"),
    ("execute SQL statement", "data"),
    ("insert records into table", "data"),
    ("update database rows", "data"),
    ("delete from database", "data"),
    ("join tables", "data"),
    ("aggregate the data", "data"),
    ("group by column", "data"),
    ("filter records", "data"),
    ("sort the dataset", "data"),
    ("load the dataframe", "data"),
    ("read CSV into pandas", "data"),
    ("process columns in dataframe", "data"),
    ("normalize the data", "data"),
    ("clean the dataset", "data"),
    ("handle missing values", "data"),
    ("fill NaN values", "data"),
    ("drop null rows", "data"),
    ("pivot the table", "data"),
    ("melt the dataframe", "data"),
    ("reshape the data", "data"),
    ("import data from file", "data"),
    ("export to spreadsheet", "data"),
    ("save as parquet", "data"),
    ("read HDF5 file", "data"),
    ("process structured data", "data"),
    ("extract field from record", "data"),
    ("validate JSON schema", "data"),
    ("transform data format", "data"),
    ("map values to new keys", "data"),
    ("flatten nested JSON", "data"),
    ("merge datasets", "data"),
    ("concatenate dataframes", "data"),
    ("append rows to table", "data"),
    ("process ETL pipeline", "data"),
    ("load data warehouse", "data"),
    ("analyze column statistics", "data"),
    ("compute data metrics", "data"),
]

# -----------------------------------------------------------------------------
# VISUALIZATION - Hard cases
# -----------------------------------------------------------------------------
VISUALIZATION_CASES = [
    ("plot the data", "visualization"),
    ("create a bar chart", "visualization"),
    ("generate line graph", "visualization"),
    ("make a pie chart", "visualization"),
    ("draw histogram", "visualization"),
    ("create scatter plot", "visualization"),
    ("visualize the distribution", "visualization"),
    ("render the chart", "visualization"),
    ("display the graph", "visualization"),
    ("show the plot", "visualization"),
    ("generate heatmap", "visualization"),
    ("create box plot", "visualization"),
    ("draw violin plot", "visualization"),
    ("make area chart", "visualization"),
    ("create stacked bar", "visualization"),
    ("plot time series", "visualization"),
    ("generate candlestick chart", "visualization"),
    ("create dashboard", "visualization"),
    ("build interactive plot", "visualization"),
    ("render 3D graph", "visualization"),
    ("create surface plot", "visualization"),
    ("draw contour map", "visualization"),
    ("generate treemap", "visualization"),
    ("create sunburst diagram", "visualization"),
    ("make radar chart", "visualization"),
    ("plot correlation matrix", "visualization"),
    ("create wordcloud", "visualization"),
    ("generate network graph", "visualization"),
    ("draw flowchart", "visualization"),
    ("create Sankey diagram", "visualization"),
    ("plot geographic map", "visualization"),
    ("create choropleth", "visualization"),
    ("render bubble chart", "visualization"),
    ("make donut chart", "visualization"),
    ("generate sparkline", "visualization"),
    ("create gauge chart", "visualization"),
    ("plot funnel chart", "visualization"),
    ("draw Gantt chart", "visualization"),
    ("create waterfall chart", "visualization"),
    ("generate polar plot", "visualization"),
    ("plot on log scale", "visualization"),
    ("create dual axis chart", "visualization"),
    ("add chart title", "visualization"),
    ("label the axes", "visualization"),
    ("add legend to plot", "visualization"),
    ("customize chart colors", "visualization"),
    ("save plot as PNG", "visualization"),
    ("export chart to PDF", "visualization"),
    ("resize the image", "visualization"),
    ("crop the figure", "visualization"),
]

# -----------------------------------------------------------------------------
# CONVERSION - Hard cases
# -----------------------------------------------------------------------------
CONVERSION_CASES = [
    ("convert celsius to fahrenheit", "conversion"),
    ("fahrenheit to celsius", "conversion"),
    ("convert kilometers to miles", "conversion"),
    ("miles to kilometers", "conversion"),
    ("convert meters to feet", "conversion"),
    ("feet to meters", "conversion"),
    ("convert pounds to kilograms", "conversion"),
    ("kg to lbs", "conversion"),
    ("convert inches to centimeters", "conversion"),
    ("cm to inches", "conversion"),
    ("convert liters to gallons", "conversion"),
    ("gallons to liters", "conversion"),
    ("convert bytes to megabytes", "conversion"),
    ("MB to GB", "conversion"),
    ("convert seconds to hours", "conversion"),
    ("hours to minutes", "conversion"),
    ("convert decimal to binary", "conversion"),
    ("binary to decimal", "conversion"),
    ("convert hex to decimal", "conversion"),
    ("decimal to hexadecimal", "conversion"),
    ("convert to base64", "conversion"),
    ("decode base64 string", "conversion"),
    ("encode URL", "conversion"),
    ("decode URL encoding", "conversion"),
    ("convert RGB to hex", "conversion"),
    ("hex color to RGB", "conversion"),
    ("convert PDF to Word", "conversion"),
    ("Word to PDF conversion", "conversion"),
    ("convert image to PNG", "conversion"),
    ("JPEG to PNG", "conversion"),
    ("convert video format", "conversion"),
    ("MP4 to AVI", "conversion"),
    ("transcode audio file", "conversion"),
    ("convert MP3 to WAV", "conversion"),
    ("render markdown to HTML", "conversion"),
    ("HTML to markdown", "conversion"),
    ("convert JSON to XML", "conversion"),
    ("XML to JSON conversion", "conversion"),
    ("translate text to French", "conversion"),
    ("convert currency USD to EUR", "conversion"),
    ("exchange rate conversion", "conversion"),
    ("timestamp to datetime", "conversion"),
    ("datetime to timestamp", "conversion"),
    ("format date to ISO", "conversion"),
    ("convert timezone", "conversion"),
    ("ASCII to Unicode", "conversion"),
    ("encode UTF-8", "conversion"),
    ("decode character encoding", "conversion"),
    ("convert case to title", "conversion"),
    ("transform data format", "conversion"),
]

# -----------------------------------------------------------------------------
# VALIDATION - Hard cases
# -----------------------------------------------------------------------------
VALIDATION_CASES = [
    ("validate email address", "validation"),
    ("check if email is valid", "validation"),
    ("verify the input format", "validation"),
    ("validate phone number", "validation"),
    ("check credit card number", "validation"),
    ("validate URL format", "validation"),
    ("is this a valid IP address", "validation"),
    ("check IPv4 format", "validation"),
    ("validate IPv6 address", "validation"),
    ("check if date is valid", "validation"),
    ("validate timestamp format", "validation"),
    ("verify JSON schema", "validation"),
    ("check XML syntax", "validation"),
    ("validate YAML format", "validation"),
    ("is the input empty", "validation"),
    ("check for null values", "validation"),
    ("validate required fields", "validation"),
    ("verify password strength", "validation"),
    ("check password requirements", "validation"),
    ("validate username format", "validation"),
    ("is this alphanumeric", "validation"),
    ("check if string is numeric", "validation"),
    ("validate integer input", "validation"),
    ("check if value is boolean", "validation"),
    ("verify file type", "validation"),
    ("validate file extension", "validation"),
    ("check mime type", "validation"),
    ("validate checksum", "validation"),
    ("verify MD5 hash", "validation"),
    ("check SHA256 hash", "validation"),
    ("validate digital signature", "validation"),
    ("verify certificate", "validation"),
    ("check SSL validity", "validation"),
    ("validate token format", "validation"),
    ("verify JWT token", "validation"),
    ("check API key format", "validation"),
    ("validate UUID format", "validation"),
    ("is this a valid GUID", "validation"),
    ("check regex pattern", "validation"),
    ("validate against schema", "validation"),
    ("ensure field constraints", "validation"),
    ("check data integrity", "validation"),
    ("validate form input", "validation"),
    ("verify user input", "validation"),
    ("audit the data quality", "validation"),
    ("test input validation", "validation"),
    ("check compliance rules", "validation"),
    ("validate business rules", "validation"),
    ("verify access permissions", "validation"),
    ("check authorization", "validation"),
]

# -----------------------------------------------------------------------------
# SEARCH - Hard cases
# -----------------------------------------------------------------------------
SEARCH_CASES = [
    ("search for keyword", "search"),
    ("find matching results", "search"),
    ("lookup the value", "search"),
    ("query the index", "search"),
    ("retrieve matching records", "search"),
    ("filter by criteria", "search"),
    ("find all occurrences", "search"),
    ("locate the pattern", "search"),
    ("scan for matches", "search"),
    ("search in database", "search"),
    ("find in documentation", "search"),
    ("lookup definition", "search"),
    ("query for results", "search"),
    ("retrieve from index", "search"),
    ("find relevant documents", "search"),
    ("search the text corpus", "search"),
    ("locate matching files", "search"),
    ("find by name", "search"),
    ("search by tag", "search"),
    ("filter and search", "search"),
    ("discover matching entries", "search"),
    ("explore the results", "search"),
    ("hunt for pattern", "search"),
    ("seek matching records", "search"),
    ("browse the catalog", "search"),
    ("locate in archive", "search"),
    ("find similar items", "search"),
    ("search for synonyms", "search"),
    ("fuzzy search query", "search"),
    ("full-text search", "search"),
    ("keyword lookup in docs", "search"),
    ("query the knowledge base", "search"),
    ("search FAQ", "search"),
    ("find in help articles", "search"),
    ("locate settings option", "search"),
    ("search configuration", "search"),
    ("find API reference", "search"),
    ("lookup function signature", "search"),
    ("search code examples", "search"),
    ("find usage patterns", "search"),
    ("query logs for error", "search"),
    ("search audit trail", "search"),
    ("find recent activity", "search"),
    ("locate user records", "search"),
    ("search customer data", "search"),
    ("find order details", "search"),
    ("query inventory", "search"),
    ("search product catalog", "search"),
    ("find matching SKU", "search"),
    ("locate warehouse item", "search"),
]

# -----------------------------------------------------------------------------
# SYSTEM - Hard cases
# -----------------------------------------------------------------------------
SYSTEM_CASES = [
    ("run the shell command", "system"),
    ("execute terminal command", "system"),
    ("spawn subprocess", "system"),
    ("start a background process", "system"),
    ("kill the process", "system"),
    ("terminate the job", "system"),
    ("get process status", "system"),
    ("list running processes", "system"),
    ("check CPU usage", "system"),
    ("monitor memory usage", "system"),
    ("get disk space", "system"),
    ("check available memory", "system"),
    ("set environment variable", "system"),
    ("get environment variables", "system"),
    ("read system config", "system"),
    ("check system time", "system"),
    ("set the timezone", "system"),
    ("get hostname", "system"),
    ("check OS version", "system"),
    ("get system info", "system"),
    ("execute bash script", "system"),
    ("run Python script", "system"),
    ("launch application", "system"),
    ("open the program", "system"),
    ("schedule the task", "system"),
    ("create cron job", "system"),
    ("monitor system resources", "system"),
    ("get network interfaces", "system"),
    ("check permissions", "system"),
    ("change user permissions", "system"),
    ("install package", "system"),
    ("update system packages", "system"),
    ("manage services", "system"),
    ("restart the service", "system"),
    ("stop the daemon", "system"),
    ("check service status", "system"),
    ("parse command arguments", "system"),
    ("handle signals", "system"),
    ("trap interrupts", "system"),
    ("manage threads", "system"),
    ("create worker pool", "system"),
    ("parallel execution", "system"),
    ("async task management", "system"),
    ("queue background job", "system"),
    ("check exit code", "system"),
    ("get return status", "system"),
    ("pipe command output", "system"),
    ("redirect stdout", "system"),
    ("capture stderr", "system"),
    ("format disk partition", "system"),
]

# -----------------------------------------------------------------------------
# AMBIGUOUS / EDGE CASES - These are the truly hard ones
# -----------------------------------------------------------------------------
AMBIGUOUS_CASES = [
    # Ambiguous between math and file
    ("sum of files", "file"),  # Files context
    ("count the files", "file"),
    ("sum of numbers", "math"),  # Math context
    ("count the numbers", "math"),
    
    # Ambiguous between web and file
    ("download the file", "web"),  # Action matters
    ("read the file", "file"),
    ("fetch the data", "web"),
    ("load the data", "data"),
    
    # Ambiguous between text and validation
    ("check if string is valid", "validation"),
    ("check the string content", "text"),
    ("validate the text format", "validation"),
    ("format the text", "text"),
    
    # Ambiguous between data and file
    ("parse the file", "data"),
    ("read CSV file", "data"),
    ("write to CSV", "data"),
    ("save the file", "file"),
    
    # Ambiguous between conversion and data
    ("convert JSON", "data"),
    ("transform JSON to XML", "conversion"),
    ("encode the data", "conversion"),
    ("serialize the data", "data"),
    
    # Ambiguous check operations
    ("check if prime", "math"),
    ("check email", "validation"),
    ("check file exists", "file"),
    ("check connection", "web"),
    ("check system status", "system"),
    
    # Ambiguous between math and data
    ("calculate statistics", "math"),
    ("analyze the data", "data"),
    ("compute metrics", "math"),
    ("aggregate values", "data"),
    
    # Ambiguous between visualization and conversion
    ("render the chart", "visualization"),
    ("render markdown", "conversion"),
    ("export as image", "visualization"),
    ("convert to PNG", "conversion"),
    
    # Ambiguous search operations
    ("find the file", "file"),
    ("find in database", "search"),
    ("search for text", "search"),
    ("find pattern in string", "text"),
    
    # Queries that could go multiple ways
    ("process the data", "data"),
    ("process the request", "web"),
    ("process the command", "system"),
    
    # Format ambiguity
    ("format the date", "conversion"),
    ("format the disk", "system"),
    ("format the string", "text"),
    ("format the code", "text"),
    
    # Various tricky cases
    ("encrypt the message", "conversion"),
    ("compress the archive", "file"),
    ("decode the token", "conversion"),
    ("parse command line", "system"),
    ("validate schema", "validation"),
    ("test the API", "web"),
    ("benchmark performance", "system"),
    ("profile the code", "system"),
    ("generate report", "visualization"),
    ("build the chart", "visualization"),
    
    # Multi-domain hints
    ("download and parse JSON", "web"),
    ("read file and compute sum", "file"),
    ("fetch URL and validate", "web"),
    ("convert and save file", "conversion"),
    
    # Unusual phrasing
    ("gimme the file contents", "file"),
    ("whatcha got for primes", "math"),
    ("yo download that stuff", "web"),
    ("make it uppercase plz", "text"),
    ("check dat email fam", "validation"),
    
    # Technical jargon
    ("CRUD operations on database", "data"),
    ("RESTful API call", "web"),
    ("regex pattern matching", "text"),
    ("bitwise operations", "math"),
    ("filesystem traversal", "file"),
    
    # Context-dependent
    ("count items", "math"),
    ("count files", "file"),
    ("count words", "text"),
    ("count records", "data"),
    
    # Verb-first priorities
    ("download the matplotlib chart", "web"),
    ("validate the downloaded file", "validation"),
    ("fetch and store CSV", "web"),
    
    # Complex queries
    ("find largest prime under 1000", "math"),
    ("search for pattern in directory", "search"),
    ("convert all images in folder", "conversion"),
    ("validate all JSON files", "validation"),
]

# Combine all cases
ALL_DOMAIN_CASES = (
    MATH_CASES + TEXT_CASES + FILE_CASES + WEB_CASES + 
    DATA_CASES + VISUALIZATION_CASES + CONVERSION_CASES +
    VALIDATION_CASES + SEARCH_CASES + SYSTEM_CASES + AMBIGUOUS_CASES
)


def generate_variants(template: str) -> list[str]:
    """Generate variants of a template with random numbers/values."""
    variants = []
    for _ in range(2):
        n = random.randint(1, 1000)
        m = random.randint(1, 1000)
        variant = template.format(n=n, m=m)
        variants.append(variant)
    return variants


def run_tests(num_cases: int = 1000):
    """Run classification tests on hard cases."""
    print(f"\n{'='*70}")
    print(f"  INTENT CLASSIFIER STRESS TEST - {num_cases} HARD CASES")
    print(f"{'='*70}\n")
    
    # Initialize classifier
    registry_path = Path(__file__).parent / "workspace" / "tools" / "registry.json"
    classifier = IntentClassifier(str(registry_path))
    
    # Generate test cases
    test_cases = []
    
    # Add base cases with variants
    for template, expected in ALL_DOMAIN_CASES:
        if "{n}" in template or "{m}" in template:
            for variant in generate_variants(template):
                test_cases.append((variant, expected))
        else:
            test_cases.append((template, expected))
    
    # If we need more cases, duplicate with slight variations
    while len(test_cases) < num_cases:
        template, expected = random.choice(ALL_DOMAIN_CASES)
        if "{n}" in template or "{m}" in template:
            for variant in generate_variants(template):
                test_cases.append((variant, expected))
        else:
            # Add variations like changing case or adding please
            variations = [
                template,
                template.upper(),
                template.lower(),
                f"please {template}",
                f"{template} please",
                f"can you {template}",
                f"I need to {template}",
            ]
            test_cases.append((random.choice(variations), expected))
        
        if len(test_cases) >= num_cases:
            break
    
    # Truncate to exact count
    test_cases = test_cases[:num_cases]
    random.shuffle(test_cases)
    
    # Run tests
    results = {
        "correct": 0,
        "incorrect": 0,
        "by_domain": defaultdict(lambda: {"correct": 0, "incorrect": 0}),
        "failed_cases": [],
    }
    
    print(f"Testing {len(test_cases)} cases...\n")
    
    for i, (query, expected) in enumerate(test_cases):
        domain, method, confidence = classifier.classify(query)
        
        if domain == expected:
            results["correct"] += 1
            results["by_domain"][expected]["correct"] += 1
        else:
            results["incorrect"] += 1
            results["by_domain"][expected]["incorrect"] += 1
            results["failed_cases"].append({
                "query": query,
                "expected": expected,
                "got": domain,
                "method": method,
                "confidence": confidence,
            })
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            acc = results["correct"] / (i + 1) * 100
            print(f"  [{i+1:4d}/{num_cases}] Accuracy so far: {acc:.1f}%")
    
    # Print results
    total = results["correct"] + results["incorrect"]
    accuracy = results["correct"] / total * 100 if total > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}\n")
    
    print(f"  Total:    {total}")
    print(f"  Correct:  {results['correct']}")
    print(f"  Incorrect: {results['incorrect']}")
    print(f"  Accuracy: {accuracy:.2f}%")
    
    print(f"\n  Per-Domain Breakdown:")
    print(f"  {'-'*50}")
    
    for domain in sorted(results["by_domain"].keys()):
        stats = results["by_domain"][domain]
        total_d = stats["correct"] + stats["incorrect"]
        acc_d = stats["correct"] / total_d * 100 if total_d > 0 else 0
        print(f"    {domain:15s}: {acc_d:5.1f}% ({stats['correct']}/{total_d})")
    
    if results["failed_cases"]:
        print(f"\n  Failed Cases (first 20):")
        print(f"  {'-'*50}")
        
        for case in results["failed_cases"][:20]:
            print(f"    Query:    {case['query'][:50]}")
            print(f"    Expected: {case['expected']}, Got: {case['got']} ({case['method']}, {case['confidence']:.2f})")
            print()
    
    print(f"\n{'='*70}")
    
    return accuracy >= 95  # Return True if 95%+ accuracy


if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    success = run_tests(1000)
    sys.exit(0 if success else 1)
