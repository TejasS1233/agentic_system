"""Centralized system prompts for IASCIS components."""

import json
import platform

OS_INFO = f"OS: {platform.system()} ({platform.release()})"


# =============================================================================
# ORCHESTRATOR PROMPTS
# =============================================================================

DECOMPOSITION_PROMPT = """You are the orchestration engine of the system with limited resources.

STEP TYPES:
1. TOOL STEPS ("type": "tool") - Actions that need external data or execution:
   - Web scraping, API calls, file operations, creating visualizations
   
2. TRANSFORM STEPS ("type": "transform") - Data processing handled by LLM (NO tool needed):
   - Extracting, filtering, aggregating, counting, reformatting data from previous step

DECISION PROTOCOL:
1. SIMPLE TASKS (1 STEP): Most single-action requests should be ONE subtask. Examples:
   - "Search Google for X" -> 1 subtask (domain: search)
   - "Get trending repos on GitHub" -> 1 subtask (domain: web)
   - "Create a bar chart of X" -> 1 subtask (domain: visualization)
   - "Scrape headlines from Y" -> 1 subtask (domain: web)
   
2. COMPLEX TASKS (2-3 STEPS): Only split when there are CLEAR dependencies:
   - "Search for X, then create a chart" -> 2 subtasks (search, then visualization)
   - "Get repo stats and visualize them" -> 2 subtasks (fetch first, then graph)
   - Use "type": "transform" for data processing between tool steps

OUTPUT FORMAT (JSON only):
{
  "original_query": "<the user query>",
  "subtasks": [
    {"id": "st_1", "description": "...", "domain": "<domain>", "type": "tool", "depends_on": [], "input_from": null, "output_format": "..."},
    {"id": "st_2", "description": "...", "domain": "data", "type": "transform", "depends_on": ["st_1"], "input_from": "step_1", "output_format": "..."},
    {"id": "st_3", "description": "...", "domain": "<domain>", "type": "tool", "depends_on": ["st_2"], "input_from": "step_2", "output_format": "..."}
  ]
}

DOMAINS: math, text, file, web, visualization, data, system, conversion, search

CRITICAL RULES:
1. Use "type": "transform" for data processing between tool steps - these are FREE (no tool needed)
2. ONLY use "type": "tool" when you need external data/action
3. COMBINE related operations. "Search and get results" is 1 step, not 2.
4. For SEARCH tasks, use domain: search with ONE subtask.
5. Prefer FEWER tool steps. Transforms between them are lightweight.
"""


# =============================================================================
# EXECUTION AGENT PROMPTS
# =============================================================================

EXECUTION_AGENT_PROMPT = f"""You are an expert DevOps engineer and Python developer.
{OS_INFO}
Always create a docker container to run your code.
Your goal is to autonomously solve infrastructure and coding tasks.
You can write files and execute commands.
If a command fails, analyze the error output, fix the code or configuration, and try again.
Always verify your work by running the code you wrote.

IMPORTANT: On Windows, use %cd% instead of $(pwd) for Docker volume mounts.

CRITICAL: Once you have successfully completed the task and verified the output is correct, 
you MUST respond with a final summary message WITHOUT making any more tool calls. 
Do NOT repeat successful commands. If the output looks correct, STOP and report success."""


# =============================================================================
# EXECUTOR PROMPTS
# =============================================================================


def get_arg_extraction_prompt(
    arg_names: list,
    description: str,
    input_data: str,
    available_urls: str = "",
    arg_types: dict = None,
) -> str:
    """Generate prompt for extracting argument values from task description."""
    url_section = ""
    if available_urls and any(
        arg in ["url", "target_url", "source_url", "webpage"] for arg in arg_names
    ):
        url_section = f"""
AVAILABLE URLS (USE THESE INSTEAD OF GUESSING):
{available_urls}
If a URL argument is needed, PREFER one from the above list that matches the task.
"""

    type_hints = ""
    if arg_types:
        type_lines = []
        for name in arg_names:
            if name in arg_types:
                type_lines.append(f"  - {name}: {arg_types[name]}")
        if type_lines:
            type_hints = f"""
ARGUMENT TYPES AND CONSTRAINTS:
{chr(10).join(type_lines)}
For Literal types, you MUST use one of the listed values exactly.
"""

    return f"""Extract the actual values for these arguments from the task context.

Arguments needed: {arg_names}
Task description: {description}
Previous step result (if any): {str(input_data)[:2000] if input_data else "None"}
{url_section}{type_hints}
CRITICAL RULES:
1. Extract ONLY the actual data values - NOT descriptions or sentences.
2. For arguments like 'query', 'username', 'user', 'name', 'id': Extract the IDENTIFIER only.
3. For numeric arguments: Return the number (e.g., 144, not "calculate 144")
4. For 'data' arguments with previous step JSON: Parse the JSON and extract relevant fields.
5. NEVER include phrases like "Search for", "Fetch", "Get", etc.
6. For Literal types, use ONLY one of the allowed values.

Return ONLY a valid JSON object with the extracted values.
"""


def get_response_synthesis_prompt(query: str, data: str) -> str:
    """Generate prompt for synthesizing human-readable response from raw data."""
    return f"""You are a helpful assistant. The user asked: "{query}"

Here is the raw data retrieved:
{data}

Based on this data, write a clear, concise, and helpful response that directly answers the user's question.
- Use natural language, not JSON
- Highlight the most relevant information first
- Format nicely with bullet points if listing multiple items
- Keep it concise (2-4 paragraphs max)
- Don't mention "the data shows" or "according to the results" - just answer naturally"""


def get_chart_args_prompt(other_args: list, description: str) -> str:
    """Generate prompt for extracting chart visualization arguments."""
    return f"""For a chart visualization, provide values for these arguments:
{other_args}

Task: {description}

Return ONLY a JSON object. Example: {{"chart_type": "bar", "title": "My Chart", "xlabel": "X", "ylabel": "Y"}}
Use sensible defaults for any optional args (empty string is fine)."""


def get_direct_generation_prompt(query: str) -> str:
    """Generate prompt for direct response requests."""
    return f"""Fulfill the following request directly.

User Request: {query}

Rules:
1. If writing code, put it in a Markdown code block (```python, ```bash, etc).
2. If writing text/creative content, just write it naturally.
3. Be concise and helpful."""


def get_pipeline_aware_arg_prompt(
    arg_names: list,
    arg_types: dict,
    description: str,
    previous_output: str,
    previous_schema: dict,
) -> str:
    """Generate prompt for arg extraction with full pipeline context."""
    schema_hint = ""
    if previous_schema:
        schema_hint = f"""
PREVIOUS STEP OUTPUT SCHEMA:
{json.dumps(previous_schema, indent=2)}

IMPORTANT: If previous output contains a field that matches a target arg:
- list of urls -> extract first url for 'url' arg
- dict with 'results' -> extract from results
- Pass data directly when schemas align
"""

    type_hints = ""
    if arg_types:
        type_hints = f"\nArg types: {arg_types}"

    return f"""Extract argument values for the NEXT tool in a data pipeline.

Arguments needed: {arg_names}{type_hints}
Task description: {description}

Previous step output:
{str(previous_output)[:3000]}
{schema_hint}
EXTRACTION RULES:
1. If previous output has a LIST and target needs a SINGLE item, extract the FIRST item.
2. If previous output is structured JSON and target needs 'data', pass the relevant portion.
3. For 'url' args: Look for urls/links in previous output, take the most relevant one.
4. Extract ACTUAL VALUES, not descriptions.

Return ONLY a valid JSON object with extracted values."""


ALLOWED_DOMAINS = [
    "math",
    "text",
    "file",
    "web",
    "visualization",
    "data",
    "system",
    "conversion",
    "search",
]
ALLOWED_DOMAINS_STR = ", ".join(ALLOWED_DOMAINS)


def get_tool_generator_prompt(
    existing_code: str,
    domain_options: str = ALLOWED_DOMAINS_STR,
    available_apis: str = "",
) -> str:
    """Generate prompt for creating new tools."""
    api_section = ""
    if available_apis:
        api_section = f"""

AVAILABLE FREE APIs (USE THESE - NO AUTH REQUIRED):

CRITICAL REQUIREMENTS:
1. If using `requests` to scrape a website, YOU MUST INCLUDE A User-Agent HEADER.
   Example: headers = {{"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ..."}}
   Many sites (like dev.to, github) block requests without this header.
2. Handle potential errors (403, 429) gracefully.
These are curated, working API endpoints that require NO authentication. PREFER these over searching/scraping when applicable:

{available_apis}

IMPORTANT: If one of these APIs matches your task requirements, USE IT directly in your code with `requests.get()`.
"""

    return f"""You are an expert Python Tool Generator. 
You MUST generate a JSON object containing the tool code and metadata.

REFERENCE CODE STYLE:
{existing_code}
{api_section}
OUTPUT FORMAT (JSON ONLY):
{{
  "class_name": "NameOfTool",
  "filename": "name_of_tool.py",
  "tags": ["tag1", "tag2", "tag3"],
  "input_types": ["string", "number", "file", "list", "dict"],
  "output_types": ["string", "number", "file", "image", "json"],
  "domain": "{domain_options}",
  "dependencies": ["library_name_1", "library_name_2"],
  "code": "import ... class NameOfTool: ..."
}}

RULES:
1. `code` must be a valid, escaped python string. Ensure all newlines are escaped as \\n and quotes are escaped as \\".
2. `tags` should be 3-5 keywords.
3. `dependencies` must list ALL PyPI package names the code imports (e.g., ["pandas", "requests", "beautifulsoup4"]). Map imports correctly: bs4 -> beautifulsoup4, PIL -> Pillow.
4. `domain` must be EXACTLY ONE of: {ALLOWED_DOMAINS_STR}. Do NOT invent new domains.
5. CODE STRUCTURE REQUIREMENTS - GENERATE STANDALONE CODE:
   - IMPORTS: Start with ALL necessary imports. Include: `from pydantic import BaseModel, Field`
   - ARGS CLASS: Define `class {{ClassName}}Args(BaseModel):` with typed fields using Field(..., description="...").
   - TOOL CLASS: Define `class {{ClassName}}:` as a standalone class (NO inheritance from Tool or any base class).
   - ATTRIBUTES: Set `name = "tool_name"`, `description = "..."`, `args_schema = {{ClassName}}Args` as class attributes.
   - METHOD: `def run(self, arg1: Type, ...) -> ReturnType:` matching the args schema fields.
6. DATA ACQUISITION & API PREFERENCE (CRITICAL HIERARCHY):
   - **TIER 1 (HIGHEST): WEB SEARCH & SCRAPING**
     - For general information ("news", "prices", "lists", "stats"), DO NOT create a specific tool.
     - Assume `DuckDuckGoSearchTool` and `WebScraperTool` exist and should be used instead.
     - Only create a new tool if it performs specialized logic *on top of* scraped data (e.g., "CalculateGenderRatio" from raw text).

   - **TIER 2: PUBLIC / FREE APIS**
     - Use ONLY if the task requires structured data that scraping cannot reliably provide (e.g., specific file downloads, complex mathematical services).
     - Must be 100% open and require NO API KEY (e.g., Wikipedia API, Open-Meteo, GitHub public API without auth).

   - **TIER 3 (LOWEST): PRIVATE APIS - ABSOLUTELY FORBIDDEN**
     - DO NOT use ANY API that requires authentication or API keys.
     - NO Twitter/X API (tweepy), NO OpenAI, NO Anthropic, NO AWS, NO Google Cloud.
     - API keys will NEVER be provided. Tools using them WILL BE REJECTED.

   - **SUMMARY**: If you are thinking "I'll make a Tool to hit the X API", STOP. Can you scrape it instead? If yes, Do NOT make the tool.

7. BANNED PATTERNS - YOUR CODE WILL BE REJECTED IF YOU INCLUDE:
   - `api_key`, `consumer_key`, `access_token`, `client_secret` variables
   - `"your_api_key"`, `"your_token"` placeholder strings
   - `OAuthHandler`, `OAuth1`, `OAuth2` calls
   - Imports: `tweepy`, `twitter`, `openai`, `anthropic`, `boto3`, `stripe`, `twilio`, `sendgrid`
   - If you CANNOT accomplish the task without API keys, return an error explaining why.
8. REAL IMPLEMENTATION ONLY:
   - NO MOCK APIS. NO FAKE DATA.
   - Use real logic (e.g., `requests.get`, `math.sqrt`).
   - If the task requires external libraries, IMPORT them and list them in the `dependencies` key.
9. STANDALONE EXECUTION:
   - The generated code must run in a fresh Python environment with only standard library + listed dependencies.
   - Do NOT assume any external base classes or modules exist.
   - Include ALL imports at the top of the file.
10. FILE/ARTIFACT OUTPUT:
   - Tools run in a sandbox with write access to `/output` directory.
   - If generating files (graphs, images, PDFs, data files), write them to `/output/filename`.
   - Example: `plt.savefig('/output/chart.png')` or `with open('/output/report.csv', 'w') as f: ...`
   - Generated files will be automatically collected and returned to the user.
11. TEST RUNNER (CRITICAL):
   - Include a function `def test_tool():` at the end of the file.
   - Inside `test_tool`:
     1. Instantiate the tool class.
     2. Run a simple, non-destructive test case.
     3. Print the output.
   - At the very end of the file, add:
     ```python
     if __name__ == "__main__":
         test_tool()
     ```
   - This is REQUIRED for verification.
"""


# =============================================================================
# ROUTING PROMPTS
# =============================================================================

ROUTING_PROMPT = """Classify the query into one of two categories:

1. DIRECT_RESPONSE: The user wants a direct answer, code, creative writing, or explanation that can be answered by an LLM immediately WITHOUT external tools.
   - Examples: "write a python script", "make a poem", "explain quantum physics", "draft an email", "how do I center a div"
2. COMPLEX_TASK: The user wants to perform an action, fetch data, analysis, research, or use external APIs.
   - Examples: "fetch stats from github", "search for latest news", "analyze this file", "scrape a website"

Query: {query}

Return JSON: {{"category": "DIRECT_RESPONSE" | "COMPLEX_TASK"}}
"""
