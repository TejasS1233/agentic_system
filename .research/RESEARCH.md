# IASCIS Research Notes

## Net Goal
A system that is independent, autonomous, self-correcting, and intelligent. This is necessary as the number of available tools is increasing faster than standard Large Language Models (LLMs) can adapt to.

## Basic Architecture (Abstracted)

### 1. Orchestrator
Responsible for overall handling and high-level decision making.

### 2. The Dispatcher
Acts as a traffic controller and routing module. It routes tasks to the correct zone (public vs. private) based on data sensitivity and task requirements.

### 3. Reflector
Analyzes execution logs to understand why code failed. It also receives feedback on how to prevent similar failures in the future.

### 4. Toolsmith
Handles code generation for tools.
- **Standard**: Generates Python functions.
- **Repo-to-Tool**: Converts external GitHub repositories or libraries into usable agent tools by wrapping their APIs.
- **Skill Library**: Indexes these tools into the Vector Store.

### 5. Static Gatekeeper
Provides guardrails and ensures code safety to prevent system crashes or corruption.
- **AST Analysis**: For structural code validation.
- **Vulnerability Scanning**: Primarily utilizes CodeQL.
- **State Gatekeeper**: Categorizes operations into risky vs. safe.

### 6. Sandbox
The execution environment, likely implemented via containers.
- **Tech Stack**: A hybrid environment using Virtual Machines (VM), Docker, or Kubernetes.
- **Isolation**: Network access is restricted by default using an allowlist.

### 7. Profiler Agent
Dynamically adapts its analysis based on the tool's computation type:
- **CPU Profiling**: Uses `cProfile`/`perf` to measure latency and recursion depth, pinpointing bottlenecks (targeting >50% compute).
- **GPU Profiling**: Leverages `torch.cuda.Event` and `nvidia-smi` to track VRAM usage and kernel timing.
- **Memory Profiling**: Monitors peak RAM to detect leaks and inefficient data structures.

### 8. Vector Store (Knowledge Graph)
Stores synthesized tools for caching.
- **Weighted Retrieval**: Retrieval is based on more than just similarity.
- **Enrichment**: Metadata includes usage frequency, success rate, and average latency (added by the LLM during a "Metadata Enrichment" phase).
- **Structure**: Uses Knowledge Graph RAG with lightweight embeddings.

## Implementation Notes
- All agents require an LLM layer except for the Gatekeeper (mostly).
- The Orchestrator and Reflector require reasoning models.
- The Toolsmith requires a code generation model.
- Both code and metadata (e.g., TTL, frequency) must be stored in the Vector Store.
- Constraint: Internet access should be restricted to prevent unauthorized API calls.
- **Important**: The ethical existence of this framework must be justified.
- Requires weighted embeddings in the Vector Store.

## Documentation and Research References

### 1. Autonomous Tool Making
- [https://arxiv.org/pdf/2302.04761](https://arxiv.org/pdf/2302.04761) - **Toolformer**: Demonstrates how LMs can teach themselves to use external tools through simple API calls.
- [https://proceedings.iclr.cc/paper_files/paper/2024/file/ed91353f700d113e5d848c7e04a858b0-Paper-Conference.pdf](https://proceedings.iclr.cc/paper_files/paper/2024/file/ed91353f700d113e5d848c7e04a858b0-Paper-Conference.pdf) - **LATM**: Large Language Models as Tool Makers, introduced the concept of agents generating reusable tools for other models.
- [https://arxiv.org/pdf/2305.16291](https://arxiv.org/pdf/2305.16291) - **Voyager**: An open-ended embodied agent in Minecraft that autonomously learns and masters skills using LLMs.

### 2. Self-Correction Mechanisms
- [CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing](https://www.researchgate.net/publication/370938047_CRITIC_Large_Language_Models_Can_Self-Correct_with_Tool-Interactive_Critiquing) - Introduces a framework for models to verify and correct their own outputs via external tool interactions.
- [https://arxiv.org/pdf/2506.06303](https://arxiv.org/pdf/2506.06303) - **Reward is Enough**: Explores how LLMs can act as In-Context Reinforcement Learners to improve performance during inference.

### 3. System Design
- [https://arxiv.org/pdf/2309.02427](https://arxiv.org/pdf/2309.02427) - **CoALA**: Cognitive Architectures for Language Agents, a systematic framework for modular memory and decision-making.

### 4. LLM Agents Making Tools
- [https://aclanthology.org/2025.acl-long.1266/](https://aclanthology.org/2025.acl-long.1266/) - Investigates standardizing how LLM agents generate and utilize new tools in complex environments.

## Novel Ideas to Implement

1. **Time-to-Live (TTL) for Tools**: Implementation of a lifespan for generated tools.
   - **Algorithm**: Modified ARC (Adaptive Replacement Cache) combining LFU (Least Frequently Used) and LRU (Least Recently Used).
   - **Decay Formula**: A semantic caching score with dynamic TTL.
2. **Decoupling Diagnosis and Correction**: Separate the logic for identifying a problem from the logic that fixes it.
3. **Tool Verification**: Verify the code that the tool itself uses.
4. **Latency-Aware Program Synthesis**: Synthesis should focus on both functional correctness and efficiency.
5. **Weighted Retrieval**: Tool retrieval should be weighted by usage frequency rather than just problem similarity.
6. **Unified Containerization**: Containerize all agents using a combination of VMs and containers to balance isolation and lightweight performance.
7. **Privacy Zones**:
   - **Public Zone**: Uses general LLMs (e.g., GPT-4, Claude) for logic and non-sensitive planning.
   - **Private Zone**: Uses locally hosted models (e.g., Llama, Mistral) for sensitive data processing.

## Open Source Codebases for Reference
- [Voyager](https://github.com/MineDojo/Voyager) - Codebase for an embodied agent that learns skills and tool-use in a complex environment.
- [LLM-ToolMaker](https://github.com/ctlllll/LLM-ToolMaker) - An implementation of the LATM framework for agents creating and storing tools.
- [E2B Code Interpreter](https://github.com/e2b-dev/code-interpreter) - A leading cloud-based sandbox SDK for running AI-generated code securely.
- [Open Interpreter](https://github.com/openinterpreter/open-interpreter) - A popular open-source tool for running LLM-generated code locally on a user's machine.
- [RestrictedPython](https://github.com/zopefoundation/RestrictedPython) - A security-focused subset of Python that restricts dangerous operations during execution.

## Proposed Research Paper Structure
1. **Introduction**
2. **Related Work**
   - 2.1 Cognitive Architectures & Memory Systems
   - 2.2 Autonomous Tool Construction & Discovery
   - 2.3 Inference-Time Improvement & Self-Correction
3. **Methodology**
   - 3.1 System Architecture Overview
   - 3.2 The Safe Foraging Loop (Gatekeeper & Diagnosis)
   - 3.3 Latency-Aware Program Synthesis
   - 3.4 Lifecycle Management (Weighted Retrieval & TTL Decay)
4. **Experimental Setup**
   - 4.1 Datasets & Baselines
   - 4.2 Evaluation Metrics
5. **Results & Analysis**
   - 5.1 Performance vs. Efficiency Trade-offs
   - 5.2 Impact of Lifecycle Management on Retrieval
   - 5.3 Safety & Latency Analysis
6. **Ethical Considerations & Broader Impact**
7. **Conclusion**
