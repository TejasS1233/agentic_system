# IASCIS - Independent Autonomous Self-Correcting Intelligent System

An autonomous, self-correcting multi-agent system that dynamically creates, profiles, and manages its own tools while enforcing strict privacy isolation between local and cloud execution zones.

## Setup

```bash
uv sync
```

## Commands

```bash
# Run the agent
uv run python run_agent.py

# Run benchmarks
uv run python -m benchmark.benchmark                    # Full IASCIS mode
uv run python -m benchmark.benchmark --mode cloud       # Cloud-only (Gemini)
uv run python -m benchmark.benchmark --mode local       # Local-only (Ollama)
uv run python -m benchmark.benchmark --mode single      # Single-shot baseline
uv run python -m benchmark.benchmark --mode no_planning # Skip planning phase
uv run python -m benchmark.benchmark --quick            # Quick test (2 tasks)

# Visualize results
uv run python -m benchmark.visualize
uv run python -m benchmark.visualize --compare          # Compare all modes
```

## Requirements

- **Gemini API**: Set `GEMINI_API_KEY` in `.env`
- **Ollama**: Run `ollama serve` with `qwen2.5-coder:7b` model
