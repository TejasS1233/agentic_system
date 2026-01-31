"""
IASCIS Benchmark Runner
Complete benchmarking script with instrumented agent execution.
Supports multiple modes for baseline comparison.
"""

import os
import sys
import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from benchmark.metrics_collector import MetricsCollector


class BenchmarkMode(Enum):
    """
    Benchmark execution modes for baseline comparison.
    
    - IASCIS: Full system with dispatcher routing (public→cloud, private→local)
    - CLOUD_ONLY: All tasks go to cloud LLM (Gemini) - no privacy
    - LOCAL_ONLY: All tasks go to local LLM (Ollama) - max privacy
    - SINGLE_SHOT: One LLM call per task, no agent loop - baseline simplicity
    - NO_PLANNING: Skip orchestrator planning phase - measures planning overhead
    """
    IASCIS = "iascis"           # Full system with routing
    CLOUD_ONLY = "cloud"        # All tasks → Gemini (no privacy)
    LOCAL_ONLY = "local"        # All tasks → Ollama (max privacy)
    SINGLE_SHOT = "single"      # One LLM call, no agent loop
    NO_PLANNING = "no_planning" # Skip orchestrator planning


@dataclass
class BenchmarkTask:
    """Definition of a benchmark task"""
    id: str
    description: str
    category: str  # "public", "private", "mixed"
    complexity: str  # "simple", "medium", "complex"
    expected_zone: str  # Expected routing decision
    file_context: List[str] = None
    validation_fn: Optional[callable] = None  # Optional validation function
    
    def __post_init__(self):
        if self.file_context is None:
            self.file_context = []


# === BENCHMARK TASK SUITE ===
BENCHMARK_TASKS = [
    # Public Zone Tasks (Cloud-based, non-sensitive)
    BenchmarkTask(
        id="pub_001",
        description="Write a Python function that calculates the fibonacci sequence up to n terms",
        category="public",
        complexity="simple",
        expected_zone="public"
    ),
    BenchmarkTask(
        id="pub_002",
        description="Create a simple REST API endpoint using Flask that returns a JSON greeting",
        category="public",
        complexity="medium",
        expected_zone="public"
    ),
    BenchmarkTask(
        id="pub_003",
        description="Write a Python script to scrape headlines from a news website",
        category="public",
        complexity="complex",
        expected_zone="public"
    ),
    BenchmarkTask(
        id="pub_004",
        description="Create a Docker container that runs a simple Python hello world script",
        category="public",
        complexity="medium",
        expected_zone="public"
    ),
    BenchmarkTask(
        id="pub_005",
        description="Write unit tests for a calculator class with add, subtract, multiply, divide methods",
        category="public",
        complexity="medium",
        expected_zone="public"
    ),
    
    # Private Zone Tasks (Local-only, sensitive data)
    BenchmarkTask(
        id="priv_001",
        description="Analyze the sensitive_payroll.csv and calculate the sum of all bonuses",
        category="private",
        complexity="simple",
        expected_zone="private",
        file_context=["sensitive_payroll.csv"]
    ),
    BenchmarkTask(
        id="priv_002",
        description="Read the .env file and list all API keys without exposing their values",
        category="private",
        complexity="simple",
        expected_zone="private",
        file_context=[".env"]
    ),
    BenchmarkTask(
        id="priv_003",
        description="Process the customer_pii.json file and anonymize all email addresses",
        category="private",
        complexity="medium",
        expected_zone="private",
        file_context=["customer_pii.json"]
    ),
    BenchmarkTask(
        id="priv_004",
        description="Analyze the private_financial_data.csv and generate a summary report",
        category="private",
        complexity="complex",
        expected_zone="private",
        file_context=["private_financial_data.csv"]
    ),
    BenchmarkTask(
        id="priv_005",
        description="Decrypt and process the secrets.encrypted file using local key management",
        category="private",
        complexity="complex",
        expected_zone="private",
        file_context=["secrets.encrypted", "keys/"]
    ),
    
    # Mixed/Edge Cases
    BenchmarkTask(
        id="mix_001",
        description="Create a data pipeline that processes user data locally then uploads aggregated stats",
        category="mixed",
        complexity="complex",
        expected_zone="private"  # Default to private when user data involved
    ),
    BenchmarkTask(
        id="mix_002",
        description="Build a monitoring dashboard that shows system metrics and logs",
        category="mixed",
        complexity="medium",
        expected_zone="public"
    ),
]


class InstrumentedLiteLLMClient:
    """
    Wrapper around LiteLLMClient that captures detailed metrics.
    """
    
    def __init__(self, original_client, metrics_collector: MetricsCollector):
        self.client = original_client
        self.metrics = metrics_collector
        self.model_name = original_client.model_name
        self.tools = original_client.tools
        self.history = original_client.history
        
    def start_chat(self, history=None):
        result = self.client.start_chat(history)
        self.history = self.client.history
        return self
    
    def send_message(self, chat, message: str) -> str:
        """Instrumented send_message with full metrics collection"""
        from litellm import completion
        import json
        
        self.client.history.append({"role": "user", "content": message})
        self.history = self.client.history
        
        tools_map = {t.name: t for t in self.tools}
        formatted_tools = self.client._get_tools_schema()
        
        MAX_TURNS = 10
        first_token_recorded = False
        
        for turn in range(MAX_TURNS):
            # Start turn metrics
            turn_start = time.perf_counter()
            self.metrics.start_turn()
            
            print(f"\n[Benchmark] Turn {turn+1}/{MAX_TURNS} - Model: {self.model_name}")
            
            try:
                response = completion(
                    model=self.model_name,
                    messages=self.client.history,
                    tools=formatted_tools,
                    tool_choice="auto" if formatted_tools else None,
                    temperature=0.0
                )
                
                # Record TTFT on first response
                if not first_token_recorded:
                    ttft = (time.perf_counter() - turn_start) * 1000
                    self.metrics.record_time_to_first_token(ttft)
                    first_token_recorded = True
                    
            except Exception as e:
                self.metrics.record_error("LLM_ERROR", str(e))
                # Check for retryable errors
                if "429" in str(e) or "503" in str(e):
                    self.metrics.record_retry(10000)  # 10s backoff
                return f"LiteLLM Error: {e}"
            
            turn_latency = (time.perf_counter() - turn_start) * 1000
            response_message = response.choices[0].message
            
            # Extract token usage
            input_tokens = getattr(response.usage, 'prompt_tokens', 0) if response.usage else 0
            output_tokens = getattr(response.usage, 'completion_tokens', 0) if response.usage else 0
            
            has_tool_calls = bool(response_message.tool_calls)
            self.metrics.end_turn(turn_latency, input_tokens, output_tokens, has_tool_calls)
            
            # Process tool calls
            if response_message.tool_calls:
                self.client.history.append(response_message)
                
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    print(f"  [Tool Call] {function_name} args={function_args}")
                    
                    tool_start = time.perf_counter()
                    tool_success = True
                    tool_error = ""
                    
                    if function_name in tools_map:
                        try:
                            tool_result = tools_map[function_name].run(**function_args)
                            
                            # Check for execution errors in result
                            if "Error" in str(tool_result) or "error" in str(tool_result).lower():
                                # Check if this might trigger code regeneration
                                if "Exit Code:" in str(tool_result) and "0" not in str(tool_result).split("Exit Code:")[-1]:
                                    self.metrics.record_command_reexecution()
                                    
                        except Exception as e:
                            tool_result = f"Error executing {function_name}: {e}"
                            tool_success = False
                            tool_error = str(e)
                    else:
                        tool_result = f"Error: Tool {function_name} not found"
                        tool_success = False
                        tool_error = "Tool not found"
                        self.metrics.record_hallucination()
                    
                    tool_time = (time.perf_counter() - tool_start) * 1000
                    
                    # Record tool call metrics
                    self.metrics.record_tool_call(
                        tool_name=function_name,
                        arguments=function_args,
                        execution_time_ms=tool_time,
                        success=tool_success,
                        error_message=tool_error
                    )
                    
                    print(f"  [Result] {str(tool_result)[:100]}...")
                    
                    self.client.history.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": str(tool_result)
                    })
                
                continue
            
            # No tool calls - final response
            self.client.history.append(response_message)
            self.history = self.client.history
            return response_message.content
        
        return "Error: Max tool turns reached."


class BenchmarkRunner:
    """
    Main benchmark runner for IASCIS system.
    Supports multiple execution modes for baseline comparison.
    """
    
    def __init__(self, output_dir: str = "benchmark_results", 
                 workspace_path: str = None,
                 run_count: int = 1,
                 mode: BenchmarkMode = BenchmarkMode.IASCIS,
                 machine_id: str = None,
                 use_docker: bool = False):
        self.output_dir = output_dir
        self.mode = mode
        self.machine_id = machine_id or self._get_machine_id()
        self.workspace_path = workspace_path or os.path.join(os.getcwd(), "workspace")
        self.run_count = run_count
        self.use_docker = use_docker
        
        # Mode and machine specific output directory
        if machine_id:
            mode_output_dir = os.path.join(output_dir, f"{mode.value}_{self.machine_id}")
        else:
            mode_output_dir = os.path.join(output_dir, mode.value)
        os.makedirs(mode_output_dir, exist_ok=True)
        self.metrics = MetricsCollector(mode_output_dir)
        self.metrics.machine_id = self.machine_id  # Pass machine ID to metrics
        
        os.makedirs(self.workspace_path, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
    
    def _get_machine_id(self) -> str:
        """Auto-detect machine identifier."""
        import platform
        import socket
        return f"{socket.gethostname()}_{platform.system()}"
        
    def setup_test_files(self):
        """Create test data files for benchmark tasks"""
        test_files = {
            "sensitive_payroll.csv": """employee_id,name,salary,bonus
1,John Doe,75000,5000
2,Jane Smith,82000,7500
3,Bob Johnson,68000,3000
4,Alice Brown,95000,10000
5,Charlie Wilson,71000,4500
""",
            "customer_pii.json": json.dumps({
                "customers": [
                    {"id": 1, "name": "John Doe", "email": "john@example.com", "ssn": "123-45-6789"},
                    {"id": 2, "name": "Jane Smith", "email": "jane@example.com", "ssn": "987-65-4321"}
                ]
            }, indent=2),
            "private_financial_data.csv": """date,transaction_id,amount,type,account
2024-01-01,TXN001,1500.00,credit,ACC001
2024-01-02,TXN002,750.50,debit,ACC002
2024-01-03,TXN003,2200.00,credit,ACC001
""",
        }
        
        for filename, content in test_files.items():
            filepath = os.path.join(self.workspace_path, filename)
            with open(filepath, "w") as f:
                f.write(content)
        
        print(f"[Benchmark] Created {len(test_files)} test files in {self.workspace_path}")
    
    def run_single_task(self, task: BenchmarkTask, run_number: int = 1, use_docker: bool = False) -> Dict[str, Any]:
        """Run a single benchmark task with full instrumentation based on mode"""
        from architecture.orchestrator import Orchestrator
        from architecture.dispatcher import Dispatcher
        from execution.tools import WriteFileTool, RunCommandTool, ReadFileTool, DockerRunCommandTool
        from execution.llm import LiteLLMClient
        from execution.core import Agent
        from litellm import completion
        
        task_id = f"{task.id}_run{run_number}"
        print(f"\n{'='*60}")
        print(f"[Benchmark] Running Task: {task_id}")
        print(f"[Benchmark] Mode: {self.mode.value.upper()}")
        print(f"[Benchmark] Description: {task.description[:50]}...")
        print(f"{'='*60}")
        
        # Start task metrics
        self.metrics.start_task(
            task_id=task_id,
            description=task.description,
            category=task.category,
            complexity=task.complexity
        )
        
        try:
            # === ROUTING PHASE ===
            # Different modes handle routing differently
            if self.mode == BenchmarkMode.IASCIS:
                # Full system: Use dispatcher to route
                dispatcher = Dispatcher()
                routing_start = time.perf_counter()
                zone = dispatcher.route(task_description=task.description, file_context=task.file_context)
                routing_time = (time.perf_counter() - routing_start) * 1000
            elif self.mode == BenchmarkMode.CLOUD_ONLY:
                # Cloud baseline: Always route to cloud (no privacy)
                zone = "public"
                routing_time = 0
                print(f"[Baseline] CLOUD_ONLY mode - forcing public zone")
            elif self.mode == BenchmarkMode.LOCAL_ONLY:
                # Local baseline: Always route locally (max privacy)
                zone = "private"
                routing_time = 0
                print(f"[Baseline] LOCAL_ONLY mode - forcing private zone")
            else:
                # Other modes: Use dispatcher but record for comparison
                dispatcher = Dispatcher()
                routing_start = time.perf_counter()
                zone = dispatcher.route(task_description=task.description, file_context=task.file_context)
                routing_time = (time.perf_counter() - routing_start) * 1000
            
            self.metrics.record_dispatcher_routing(
                zone=zone,
                routing_time_ms=routing_time,
                expected_zone=task.expected_zone
            )
            print(f"[Benchmark] Routed to: {zone} (expected: {task.expected_zone})")
            
            # === PLANNING PHASE ===
            if self.mode == BenchmarkMode.NO_PLANNING or self.mode == BenchmarkMode.SINGLE_SHOT:
                # Skip planning for these baselines
                plan = task.description  # Use raw task as "plan"
                planning_time = 0
                print(f"[Baseline] Skipping planning phase")
            else:
                # Normal planning with orchestrator
                # Use Ollama for planning in LOCAL_ONLY mode
                if self.mode == BenchmarkMode.LOCAL_ONLY:
                    planner_model = "ollama/qwen2.5-coder:7b"
                else:
                    planner_model = "gemini/gemini-3-flash-preview"
                    
                orchestrator = Orchestrator(model_name=planner_model)
                planning_start = time.perf_counter()
                plan = orchestrator.run(task.description)
                planning_time = (time.perf_counter() - planning_start) * 1000
            
            self.metrics.record_planning_time(planning_time)
            print(f"[Benchmark] Planning completed in {planning_time:.0f}ms")
            
            # === MODEL SELECTION ===
            if self.mode == BenchmarkMode.CLOUD_ONLY:
                model_name = "gemini/gemini-3-flash-preview"
            elif self.mode == BenchmarkMode.LOCAL_ONLY:
                model_name = "ollama/qwen2.5-coder:7b"
            else:
                # IASCIS, SINGLE_SHOT, NO_PLANNING use zone-based selection
                model_name = "ollama/qwen2.5-coder:7b" if zone == "private" else "gemini/gemini-3-flash-preview"
            
            self.metrics.record_model_used(model_name)
            print(f"[Benchmark] Using model: {model_name}")
            
            # === EXECUTION PHASE ===
            # Choose between Docker or local execution
            if use_docker:
                print("[Benchmark] Using Docker isolation for command execution")
                run_tool = DockerRunCommandTool(self.workspace_path)
            else:
                run_tool = RunCommandTool(self.workspace_path)
            
            tools = [
                WriteFileTool(self.workspace_path),
                run_tool,
                ReadFileTool(self.workspace_path)
            ]
            
            if self.mode == BenchmarkMode.SINGLE_SHOT:
                # Single-shot baseline: One LLM call, no agent loop
                print(f"[Baseline] SINGLE_SHOT mode - one LLM call only")
                
                execution_start = time.perf_counter()
                self.metrics.start_turn()
                
                # Build tool schema
                formatted_tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters
                        }
                    }
                    for t in tools
                ]
                
                prompt = f"""Complete this task in a SINGLE response. You have one chance to provide the solution.

Task: {task.description}

Provide complete code and commands. You cannot iterate or fix errors."""
                
                try:
                    response = completion(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        tools=formatted_tools,
                        tool_choice="auto",
                        temperature=0.0
                    )
                    
                    ttft = (time.perf_counter() - execution_start) * 1000
                    self.metrics.record_time_to_first_token(ttft)
                    
                    input_tokens = getattr(response.usage, 'prompt_tokens', 0) if response.usage else 0
                    output_tokens = getattr(response.usage, 'completion_tokens', 0) if response.usage else 0
                    
                    turn_latency = (time.perf_counter() - execution_start) * 1000
                    has_tool_calls = bool(response.choices[0].message.tool_calls)
                    self.metrics.end_turn(turn_latency, input_tokens, output_tokens, has_tool_calls)
                    
                    result = response.choices[0].message.content or "Single-shot completed"
                    
                    # Execute any tool calls (but only once, no loop)
                    if response.choices[0].message.tool_calls:
                        for tool_call in response.choices[0].message.tool_calls[:1]:  # Only first tool
                            function_name = tool_call.function.name
                            function_args = json.loads(tool_call.function.arguments)
                            tools_map = {t.name: t for t in tools}
                            
                            if function_name in tools_map:
                                tool_start = time.perf_counter()
                                try:
                                    tool_result = tools_map[function_name].run(**function_args)
                                    self.metrics.record_tool_call(
                                        tool_name=function_name,
                                        arguments=function_args,
                                        execution_time_ms=(time.perf_counter() - tool_start) * 1000,
                                        success=True
                                    )
                                except Exception as e:
                                    self.metrics.record_tool_call(
                                        tool_name=function_name,
                                        arguments=function_args,
                                        execution_time_ms=(time.perf_counter() - tool_start) * 1000,
                                        success=False,
                                        error_message=str(e)
                                    )
                    
                except Exception as e:
                    self.metrics.record_error("SINGLE_SHOT_ERROR", str(e))
                    result = f"Error: {e}"
                
                execution_time = (time.perf_counter() - execution_start) * 1000
                
            else:
                # Full agent execution (IASCIS, CLOUD_ONLY, LOCAL_ONLY, NO_PLANNING)
                original_client = LiteLLMClient(model_name=model_name, tools=tools)
                instrumented_client = InstrumentedLiteLLMClient(original_client, self.metrics)
                
                agent = Agent(
                    workspace_path=self.workspace_path,
                    tools=tools,
                    llm_client=instrumented_client
                )
                
                execution_start = time.perf_counter()
                goal = f"Execute this plan:\n{plan}\n\nAnalyze the plan, write the necessary code, and run it."
                result = agent.run(goal)
                execution_time = (time.perf_counter() - execution_start) * 1000
            
            self.metrics.record_execution_time(execution_time)
            
            # Determine success
            success = "Error" not in str(result) and result and "Failed" not in str(result)
            self.metrics.record_execution_success(success)
            
            # End task
            task_metrics = self.metrics.end_task(completed=True, output_correct=None)
            
            print(f"[Benchmark] Task completed in {task_metrics.total_duration_ms:.0f}ms")
            print(f"[Benchmark] Turns: {task_metrics.total_turns}, Tool Calls: {task_metrics.total_tool_calls}")
            
            return task_metrics.to_dict()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.metrics.record_error("TASK_ERROR", str(e))
            task_metrics = self.metrics.end_task(completed=False)
            return task_metrics.to_dict()
    
    def run_benchmark(self, tasks: List[BenchmarkTask] = None, 
                      categories: List[str] = None) -> Dict[str, Any]:
        """Run the complete benchmark suite"""
        if tasks is None:
            tasks = BENCHMARK_TASKS
        
        if categories:
            tasks = [t for t in tasks if t.category in categories]
        
        print("\n" + "=" * 80)
        print(" IASCIS BENCHMARK SUITE")
        print("=" * 80)
        print(f" Tasks: {len(tasks)}")
        print(f" Runs per task: {self.run_count}")
        print(f" Total executions: {len(tasks) * self.run_count}")
        print("=" * 80 + "\n")
        
        # Setup test files
        self.setup_test_files()
        
        # Run all tasks
        all_results = []
        for run in range(1, self.run_count + 1):
            print(f"\n{'#'*40}")
            print(f" RUN {run}/{self.run_count}")
            print(f"{'#'*40}")
            
            for task in tasks:
                try:
                    result = self.run_single_task(task, run, use_docker=self.use_docker)
                    all_results.append(result)
                except Exception as e:
                    print(f"[ERROR] Task {task.id} failed: {e}")
        
        # Generate and save results
        tasks_file, summary_file = self.metrics.save_results()
        self.metrics.print_summary()
        
        print(f"\n[Benchmark] Results saved to:")
        print(f"  - Tasks: {tasks_file}")
        print(f"  - Summary: {summary_file}")
        
        return {
            "tasks_file": tasks_file,
            "summary_file": summary_file,
            "summary": self.metrics.generate_summary()
        }
    
    def run_quick_test(self) -> Dict[str, Any]:
        """Run a quick test with just 2 simple tasks"""
        quick_tasks = [
            BenchmarkTask(
                id="quick_001",
                description="Write a Python function that adds two numbers",
                category="public",
                complexity="simple",
                expected_zone="public"
            ),
            BenchmarkTask(
                id="quick_002",
                description="Read the sensitive_payroll.csv file and count the rows",
                category="private",
                complexity="simple",
                expected_zone="private",
                file_context=["sensitive_payroll.csv"]
            ),
        ]
        return self.run_benchmark(tasks=quick_tasks)

def run_all_baselines(output_dir: str = "benchmark_results", quick: bool = True):
    """
    Run benchmarks across all modes for comparison.
    Returns paths to all result files.
    """
    results = {}
    
    for mode in BenchmarkMode:
        print(f"\n{'#'*80}")
        print(f"# RUNNING BASELINE: {mode.value.upper()}")
        print(f"{'#'*80}\n")
        
        runner = BenchmarkRunner(
            output_dir=output_dir,
            run_count=1,
            mode=mode
        )
        
        if quick:
            result = runner.run_quick_test()
        else:
            result = runner.run_benchmark()
        
        results[mode.value] = result
    
    print("\n" + "="*80)
    print(" ALL BASELINES COMPLETE")
    print("="*80)
    print(f"\nResults saved in mode-specific directories under: {output_dir}/")
    for mode_name in results:
        print(f"  - {mode_name}/")
    
    return results


def main():
    """Main entry point for benchmark"""
    import argparse
    
    parser = argparse.ArgumentParser(description="IASCIS Benchmark Suite with Baseline Comparisons")
    parser.add_argument("--output", "-o", default="benchmark_results", 
                        help="Output directory for results")
    parser.add_argument("--runs", "-r", type=int, default=1,
                        help="Number of runs per task")
    parser.add_argument("--category", "-c", choices=["public", "private", "mixed", "all"],
                        default="all", help="Task category to run")
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Run quick test (2 tasks only)")
    parser.add_argument("--mode", "-m", 
                        choices=["iascis", "cloud", "local", "single", "no_planning", "all"],
                        default="iascis",
                        help="Execution mode: iascis (full system), cloud (cloud-only baseline), "
                             "local (local-only baseline), single (single-shot baseline), "
                             "no_planning (no planning baseline), all (run all modes for comparison)")
    parser.add_argument("--machine", 
                        help="Machine identifier for multi-machine benchmarking (e.g., 'laptop_tejas', 'desktop_friend'). "
                             "Results from different machines are stored separately for comparison.")
    parser.add_argument("--docker", "-d", action="store_true",
                        help="Run commands inside a Docker container (python:3.11-slim). "
                             "Provides pip and full Python environment isolation.")
    
    args = parser.parse_args()
    
    # Handle "all" mode - run all baselines
    if args.mode == "all":
        results = run_all_baselines(args.output, args.quick)
        return results
    
    # Map mode string to enum
    mode_map = {
        "iascis": BenchmarkMode.IASCIS,
        "cloud": BenchmarkMode.CLOUD_ONLY,
        "local": BenchmarkMode.LOCAL_ONLY,
        "single": BenchmarkMode.SINGLE_SHOT,
        "no_planning": BenchmarkMode.NO_PLANNING
    }
    
    runner = BenchmarkRunner(
        output_dir=args.output,
        run_count=args.runs,
        mode=mode_map[args.mode],
        machine_id=args.machine,
        use_docker=args.docker
    )
    
    if args.quick:
        results = runner.run_quick_test()
    else:
        categories = None if args.category == "all" else [args.category]
        results = runner.run_benchmark(categories=categories)
    
    return results


if __name__ == "__main__":
    main()
