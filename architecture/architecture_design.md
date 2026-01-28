# Architecture Design: Independent Autonomous Self-Correcting Intelligent System (IASCIS) - Student/Free Tier Edition

## 1. High-Level Architecture
The system follows a **Hierarchical Multi-Agent Event-Driven Architecture**, optimized for **Zero-Cost Operation** using Free Tier APIs and Local Compute.

### Core Components
1.  **Orchestrator** (Gemini 1.5 Flash - Free Tier): High-speed planning and decision making. Free tier allows high RPM suitable for orchestration.
2.  **Dispatcher** (Local/Simple): Routes requests.
    -   *Logic*: Simple keyword matching or a tiny local model (Qwen 2.5 0.5B) to save API calls.
3.  **Zones (The Workspaces)**:
    -   **Public Zone**: **Gemini 1.5 Pro (Free)** for complex logic, **Flash** for speed.
    -   **Private Zone**: **Ollama (Qwen 2.5 Coder)** running locally for sensitive code/data.
4.  **Toolsmith** (Gemini 1.5 Pro): Generates tools.
5.  **Reflector** (Gemini 1.5 Pro/Ollama): Analyzes logs.
6.  **Knowledge Graph**: **ChromaDB** (Local/Open Source) for vector storage.

---

## 2. Differentiation: IASCIS vs. Agentic IDEs

You asked: *"How do I differentiate myself from Agentic IDEs (Cursor, Windsurf, Copilot Workspace)?"*

| Feature                 | Agentic IDE (Cursor, Windsurf)                                       | IASCIS (Your System)                                                                                                     |
| :---------------------- | :------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------- |
| **Primary Interaction** | **Human-in-the-loop** (Copilot). User drives the logic, AI suggests. | **Human-on-the-loop** (Colleague). User sets a goal, AI drives the execution.                                            |
| **Correction**          | User manually reviews and accepts/rejects diffs.                     | **Autonomous Self-Correction**. The system catches its own errors (via Reflector) and fixes them *before* notifying you. |
| **Tooling**             | Fixed set of tools (Edit, Run Terminal).                             | **Dynamic Toolsmithing**. Can *create its own tools* (e.g., "I need a PDF parser, let me write one").                    |
| **Memory**              | Context window based (Project context).                              | **Long-term Evolution** (TTL + Usage Weights). It "learns" which tools are efficient over weeks of usage.                |
| **Privacy**             | Cloud-first (usually).                                               | **Strict Privacy Zones** (Dispatcher). Guarantees sensitive data *never* leaves localhost via architectural routing.     |

**The "Pitch"**:
> "Agentic IDEs make you a faster *typist*. IASCIS makes you a *manager*."

---

## 3. Honest Architecture Critique & Risks

### The "Student" Reality Check
While the architecture is robust, here are the real-world bottlenecks you will face with this "Free" stack:

> [!WARNING]
> **1. The Context Context Switch Overhead**
> Switching between Gemini (Cloud) and Ollama (Local) creates network latency. Your "Dispatcher" needs to be extremely fast (ms level) to not become a bottleneck itself. Hardcoded rules > AI Dispatcher for cost/speed here.

> [!WARNING]
> **2. The RAM Trap**
> Running **Qwen 2.5 Coder (7B)** + **Docker Containers** + **VectorDB** + **Python Logic** simultaneously will easily eat 16GB+ RAM.
> *Mitigation*: Ensure your "Private Zone" shuts down the LLM model from VRAM when not in active use (Ollama does this automatically after 5m).

> [!WARNING]
> **3. Gemini Free Tier Limits**
> Gemini Free has Rate Limits (RPM/TPM). An autonomous loop can spiral and hit these in seconds.
> *Mitigation*: You **MUST** implement a "Backoff/Sleep" utility in your `Orchestrator`.

---

## 4. Why LiteLLM? (The Glue)

You asked: *"Is it actually useful?"*
**Verdict: YES, it is critical for your use case.**

Without LiteLLM, your code looks like this:
```python
if model == "gemini":
    response = google.genai.generate(...)
    text = response.text
elif model == "ollama":
    response = openai_client.chat.completions.create(...)
    text = response.choices[0].message.content
```
This is unmaintainable spaghetti.

**With LiteLLM**:
```python
from litellm import completion

# It normalizes EVERYTHING (Google, Ollama, Anthropic) into standard OpenAI format
response = completion(
    model="gemini/gemini-1.5-flash", # or "ollama/qwen2.5-coder"
    messages=[...]
)
```
*   **Cost Effective**: It allows you to swap "Gemini Pro" for "Ollama" in *one line of config* if you hit rate limits.
*   **Unified interface**: You write the agent logic ONCE. The backend is just a string variable.

---

## 5. Component Detail

### 5.1 The Dispatcher (Revised for efficiency)
-   **Old Idea**: LLM decides routing.
-   **New Idea (Cost Effective)**: Regex + file path analysis.
    -   If file path contains `.env`, `keys`, `private`: **FORCE PRIVATE ZONE**.
    -   If user prompt contains "upload", "internet", "search": **FORCE PUBLIC ZONE**.
    -   Only use LLM (Gemini Flash) if ambiguous.

### 5.2 Security Zones (The Sandbox)
-   **Containerization**: Docker.
    -   **Optimization**: Use `alpine` based images for speed/size. Keep a "warm" container running to avoid the 2-3s startup penalty.

### 5.3 The Toolsmith
-   **Mode**: Repo-to-Tool.
    -   **Optimization**: Don't index *everything*. Use `ctags` or `tree-sitter` (fast, static) to map the repo structure first, then only feed *relevant* file contents to the LLM to generate the tool wrapper. Saves massive amounts of tokens.

---

## 6. Novel Algorithms (Revised)

### 6.1 Adaptive Tool Lifecycle (TTL)
-   **Implementation**: A JSON file `tool_metadata.json`.
-   **Logic**: Every time a tool is used, update `last_used_timestamp`.
-   **Cleanup**: On Agent startup, valid `if (now - last_used) > 7_days: delete_tool()`. Simple, effective, zero cost.

---

## 7. Implementation Stack (Student Edition)

| Component        | Technology                                               | Cost                |
| :--------------- | :------------------------------------------------------- | :------------------ |
| **Language**     | Python 3.12+                                             | Free                |
| **Orchestrator** | **LiteLLM** + Native Python                              | Free                |
| **LLM Backend**  | **Gemini 1.5 Flash** (Main) + **Qwen 2.5 Coder** (Local) | Free / HW Dependent |
| **Vector Store** | **ChromaDB** (Local persistence)                         | Free                |
| **Sandbox**      | Docker Desktop                                           | Free                |
| **Profiling**    | standard `cProfile`                                      | Free                |
