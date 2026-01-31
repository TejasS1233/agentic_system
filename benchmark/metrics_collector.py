"""
IASCIS Metrics Collector Module
Comprehensive metrics collection for research paper benchmarking.
"""

import time
import json
import os
import psutil
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
import statistics


@dataclass
class TokenMetrics:
    """LLM Token usage metrics"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class ToolCallMetrics:
    """Individual tool call metrics"""
    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    arguments_size_bytes: int = 0
    execution_time_ms: float = 0.0
    success: bool = True
    error_message: str = ""
    timestamp: str = ""


@dataclass
class TurnMetrics:
    """Metrics for a single reasoning turn"""
    turn_number: int = 0
    latency_ms: float = 0.0
    tokens: TokenMetrics = field(default_factory=TokenMetrics)
    tool_calls: List[ToolCallMetrics] = field(default_factory=list)
    has_tool_calls: bool = False
    timestamp: str = ""


@dataclass
class ResourceSnapshot:
    """System resource snapshot"""
    timestamp: str = ""
    ram_usage_mb: float = 0.0
    ram_percent: float = 0.0
    cpu_percent: float = 0.0
    gpu_memory_mb: float = 0.0  # If available
    disk_read_mb: float = 0.0
    disk_write_mb: float = 0.0


@dataclass
class TaskMetrics:
    """Complete metrics for a single task execution"""
    # Task identification
    task_id: str = ""
    task_description: str = ""
    task_category: str = ""  # public, private, mixed
    complexity_level: str = ""  # simple, medium, complex
    
    # Timing metrics
    start_time: str = ""
    end_time: str = ""
    total_duration_ms: float = 0.0
    dispatcher_routing_time_ms: float = 0.0
    planning_time_ms: float = 0.0
    execution_time_ms: float = 0.0
    time_to_first_token_ms: float = 0.0
    
    # Routing metrics
    routed_zone: str = ""  # public or private
    routing_correct: Optional[bool] = None  # For validation
    
    # LLM metrics
    model_used: str = ""
    total_turns: int = 0
    max_turns_reached: bool = False
    turns: List[TurnMetrics] = field(default_factory=list)
    total_tokens: TokenMetrics = field(default_factory=TokenMetrics)
    
    # Tool metrics
    total_tool_calls: int = 0
    tool_call_distribution: Dict[str, int] = field(default_factory=dict)
    tool_success_rate: float = 0.0
    tool_errors: List[Dict[str, str]] = field(default_factory=list)
    average_tool_execution_time_ms: float = 0.0
    
    # Self-correction metrics
    retry_count: int = 0
    backoff_time_total_ms: float = 0.0
    errors_recovered: int = 0
    code_regenerations: int = 0
    commands_reexecuted: int = 0
    
    # Quality metrics
    task_completed: bool = False
    output_correct: Optional[bool] = None  # For manual validation
    execution_success: bool = False
    hallucinations_detected: int = 0
    
    # Resource metrics
    peak_ram_mb: float = 0.0
    average_ram_mb: float = 0.0
    peak_cpu_percent: float = 0.0
    average_cpu_percent: float = 0.0
    resource_snapshots: List[ResourceSnapshot] = field(default_factory=list)
    
    # Error tracking
    errors: List[Dict[str, str]] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)


class MetricsCollector:
    """
    Comprehensive metrics collector for IASCIS benchmarking.
    Collects all metrics across performance, LLM, tools, privacy, resources, and quality.
    """
    
    # Pricing per 1M tokens (update as needed)
    PRICING = {
        "gemini/gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        "gemini/gemini-1.5-pro": {"input": 1.25, "output": 5.00},
        "gemini/gemini-3-flash-preview": {"input": 0.10, "output": 0.40},
        "ollama/qwen2.5-coder:7b": {"input": 0.0, "output": 0.0},  # Local Ollama, free
    }
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.current_task: Optional[TaskMetrics] = None
        self.all_tasks: List[TaskMetrics] = []
        
        # Resource monitoring
        self._resource_monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._resource_snapshots: List[ResourceSnapshot] = []
        
        # Initial disk counters
        self._initial_disk_io = psutil.disk_io_counters()
        
    def start_task(self, task_id: str, description: str, category: str = "unknown", 
                   complexity: str = "unknown") -> TaskMetrics:
        """Start collecting metrics for a new task"""
        self.current_task = TaskMetrics(
            task_id=task_id,
            task_description=description,
            task_category=category,
            complexity_level=complexity,
            start_time=datetime.now().isoformat()
        )
        self._resource_snapshots = []
        self._start_resource_monitoring()
        return self.current_task
    
    def end_task(self, completed: bool = True, output_correct: Optional[bool] = None) -> TaskMetrics:
        """Finalize metrics collection for current task"""
        if not self.current_task:
            raise ValueError("No active task to end")
        
        self._stop_resource_monitoring()
        
        task = self.current_task
        task.end_time = datetime.now().isoformat()
        task.task_completed = completed
        task.output_correct = output_correct
        
        # Calculate total duration
        start = datetime.fromisoformat(task.start_time)
        end = datetime.fromisoformat(task.end_time)
        task.total_duration_ms = (end - start).total_seconds() * 1000
        
        # Aggregate token metrics
        for turn in task.turns:
            task.total_tokens.input_tokens += turn.tokens.input_tokens
            task.total_tokens.output_tokens += turn.tokens.output_tokens
            task.total_tokens.total_tokens += turn.tokens.total_tokens
            task.total_tokens.cost_usd += turn.tokens.cost_usd
        
        # Calculate tool metrics
        all_tool_calls = []
        for turn in task.turns:
            all_tool_calls.extend(turn.tool_calls)
        
        task.total_tool_calls = len(all_tool_calls)
        
        if all_tool_calls:
            # Tool call distribution
            for tc in all_tool_calls:
                task.tool_call_distribution[tc.tool_name] = \
                    task.tool_call_distribution.get(tc.tool_name, 0) + 1
            
            # Tool success rate
            successful = sum(1 for tc in all_tool_calls if tc.success)
            task.tool_success_rate = (successful / len(all_tool_calls)) * 100
            
            # Average tool execution time
            exec_times = [tc.execution_time_ms for tc in all_tool_calls]
            task.average_tool_execution_time_ms = statistics.mean(exec_times)
            
            # Collect tool errors
            for tc in all_tool_calls:
                if not tc.success:
                    task.tool_errors.append({
                        "tool": tc.tool_name,
                        "error": tc.error_message,
                        "timestamp": tc.timestamp
                    })
        
        # Resource metrics aggregation
        task.resource_snapshots = self._resource_snapshots
        if self._resource_snapshots:
            ram_values = [s.ram_usage_mb for s in self._resource_snapshots]
            cpu_values = [s.cpu_percent for s in self._resource_snapshots]
            
            task.peak_ram_mb = max(ram_values)
            task.average_ram_mb = statistics.mean(ram_values)
            task.peak_cpu_percent = max(cpu_values)
            task.average_cpu_percent = statistics.mean(cpu_values)
        
        # Check max turns
        task.max_turns_reached = task.total_turns >= 10  # MAX_TURNS from llm.py
        
        self.all_tasks.append(task)
        self.current_task = None
        
        return task
    
    def record_dispatcher_routing(self, zone: str, routing_time_ms: float, 
                                   expected_zone: Optional[str] = None):
        """Record dispatcher routing decision"""
        if self.current_task:
            self.current_task.routed_zone = zone
            self.current_task.dispatcher_routing_time_ms = routing_time_ms
            if expected_zone:
                self.current_task.routing_correct = (zone == expected_zone)
    
    def record_planning_time(self, planning_time_ms: float):
        """Record orchestrator planning time"""
        if self.current_task:
            self.current_task.planning_time_ms = planning_time_ms
    
    def record_execution_time(self, execution_time_ms: float):
        """Record agent execution time"""
        if self.current_task:
            self.current_task.execution_time_ms = execution_time_ms
    
    def record_time_to_first_token(self, ttft_ms: float):
        """Record time to first token"""
        if self.current_task:
            self.current_task.time_to_first_token_ms = ttft_ms
    
    def record_model_used(self, model_name: str):
        """Record which model was used"""
        if self.current_task:
            self.current_task.model_used = model_name
    
    def start_turn(self) -> int:
        """Start a new reasoning turn, returns turn number"""
        if self.current_task:
            turn_num = len(self.current_task.turns) + 1
            turn = TurnMetrics(
                turn_number=turn_num,
                timestamp=datetime.now().isoformat()
            )
            self.current_task.turns.append(turn)
            self.current_task.total_turns = turn_num
            return turn_num
        return 0
    
    def end_turn(self, latency_ms: float, input_tokens: int = 0, output_tokens: int = 0,
                 has_tool_calls: bool = False):
        """End current turn with metrics"""
        if self.current_task and self.current_task.turns:
            turn = self.current_task.turns[-1]
            turn.latency_ms = latency_ms
            turn.has_tool_calls = has_tool_calls
            
            # Calculate tokens and cost
            turn.tokens.input_tokens = input_tokens
            turn.tokens.output_tokens = output_tokens
            turn.tokens.total_tokens = input_tokens + output_tokens
            
            # Calculate cost
            model = self.current_task.model_used
            if model in self.PRICING:
                pricing = self.PRICING[model]
                turn.tokens.cost_usd = (
                    (input_tokens / 1_000_000) * pricing["input"] +
                    (output_tokens / 1_000_000) * pricing["output"]
                )
    
    def record_tool_call(self, tool_name: str, arguments: Dict[str, Any],
                         execution_time_ms: float, success: bool = True,
                         error_message: str = ""):
        """Record a tool call within the current turn"""
        if self.current_task and self.current_task.turns:
            tool_call = ToolCallMetrics(
                tool_name=tool_name,
                arguments=arguments,
                arguments_size_bytes=len(json.dumps(arguments).encode()),
                execution_time_ms=execution_time_ms,
                success=success,
                error_message=error_message,
                timestamp=datetime.now().isoformat()
            )
            self.current_task.turns[-1].tool_calls.append(tool_call)
    
    def record_retry(self, backoff_time_ms: float = 0):
        """Record a retry attempt"""
        if self.current_task:
            self.current_task.retry_count += 1
            self.current_task.backoff_time_total_ms += backoff_time_ms
    
    def record_error_recovered(self):
        """Record an error that was automatically recovered"""
        if self.current_task:
            self.current_task.errors_recovered += 1
    
    def record_code_regeneration(self):
        """Record when code was regenerated after an error"""
        if self.current_task:
            self.current_task.code_regenerations += 1
    
    def record_command_reexecution(self):
        """Record when a command was re-executed"""
        if self.current_task:
            self.current_task.commands_reexecuted += 1
    
    def record_hallucination(self):
        """Record a detected hallucination"""
        if self.current_task:
            self.current_task.hallucinations_detected += 1
    
    def record_execution_success(self, success: bool):
        """Record whether task execution was successful"""
        if self.current_task:
            self.current_task.execution_success = success
    
    def record_error(self, error_type: str, message: str):
        """Record an error that occurred"""
        if self.current_task:
            self.current_task.errors.append({
                "type": error_type,
                "message": message,
                "timestamp": datetime.now().isoformat()
            })
    
    def _start_resource_monitoring(self, interval_seconds: float = 0.5):
        """Start background thread for resource monitoring"""
        self._stop_monitoring.clear()
        self._initial_disk_io = psutil.disk_io_counters()
        
        def monitor():
            process = psutil.Process()
            while not self._stop_monitoring.is_set():
                try:
                    mem_info = process.memory_info()
                    cpu_percent = process.cpu_percent(interval=0.1)
                    disk_io = psutil.disk_io_counters()
                    
                    # Calculate disk delta
                    disk_read_mb = (disk_io.read_bytes - self._initial_disk_io.read_bytes) / (1024 * 1024)
                    disk_write_mb = (disk_io.write_bytes - self._initial_disk_io.write_bytes) / (1024 * 1024)
                    
                    # Try to get GPU memory if available
                    gpu_memory_mb = 0.0
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        gpu_memory_mb = info.used / (1024 * 1024)
                    except:
                        pass
                    
                    snapshot = ResourceSnapshot(
                        timestamp=datetime.now().isoformat(),
                        ram_usage_mb=mem_info.rss / (1024 * 1024),
                        ram_percent=process.memory_percent(),
                        cpu_percent=cpu_percent,
                        gpu_memory_mb=gpu_memory_mb,
                        disk_read_mb=disk_read_mb,
                        disk_write_mb=disk_write_mb
                    )
                    self._resource_snapshots.append(snapshot)
                except Exception:
                    pass
                
                time.sleep(interval_seconds)
        
        self._resource_monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._resource_monitor_thread.start()
    
    def _stop_resource_monitoring(self):
        """Stop resource monitoring thread"""
        self._stop_monitoring.set()
        if self._resource_monitor_thread:
            self._resource_monitor_thread.join(timeout=2)
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate aggregate summary across all tasks"""
        if not self.all_tasks:
            return {"error": "No tasks recorded"}
        
        summary = {
            "benchmark_info": {
                "total_tasks": len(self.all_tasks),
                "timestamp": datetime.now().isoformat()
            },
            "performance_metrics": {},
            "llm_metrics": {},
            "tool_metrics": {},
            "routing_metrics": {},
            "resource_metrics": {},
            "self_correction_metrics": {},
            "quality_metrics": {},
            "per_model_breakdown": {},
            "per_zone_breakdown": {}
        }
        
        # Performance metrics
        durations = [t.total_duration_ms for t in self.all_tasks]
        planning_times = [t.planning_time_ms for t in self.all_tasks if t.planning_time_ms > 0]
        execution_times = [t.execution_time_ms for t in self.all_tasks if t.execution_time_ms > 0]
        routing_times = [t.dispatcher_routing_time_ms for t in self.all_tasks if t.dispatcher_routing_time_ms > 0]
        ttfts = [t.time_to_first_token_ms for t in self.all_tasks if t.time_to_first_token_ms > 0]
        
        summary["performance_metrics"] = {
            "total_duration": {
                "mean_ms": statistics.mean(durations),
                "median_ms": statistics.median(durations),
                "min_ms": min(durations),
                "max_ms": max(durations),
                "std_dev_ms": statistics.stdev(durations) if len(durations) > 1 else 0
            },
            "planning_time": {
                "mean_ms": statistics.mean(planning_times) if planning_times else 0,
                "median_ms": statistics.median(planning_times) if planning_times else 0
            },
            "execution_time": {
                "mean_ms": statistics.mean(execution_times) if execution_times else 0,
                "median_ms": statistics.median(execution_times) if execution_times else 0
            },
            "dispatcher_routing_time": {
                "mean_ms": statistics.mean(routing_times) if routing_times else 0,
                "median_ms": statistics.median(routing_times) if routing_times else 0
            },
            "time_to_first_token": {
                "mean_ms": statistics.mean(ttfts) if ttfts else 0,
                "median_ms": statistics.median(ttfts) if ttfts else 0
            },
            "tasks_per_minute": len(self.all_tasks) / (sum(durations) / 60000) if sum(durations) > 0 else 0
        }
        
        # LLM metrics
        total_input = sum(t.total_tokens.input_tokens for t in self.all_tasks)
        total_output = sum(t.total_tokens.output_tokens for t in self.all_tasks)
        total_cost = sum(t.total_tokens.cost_usd for t in self.all_tasks)
        turns = [t.total_turns for t in self.all_tasks]
        
        summary["llm_metrics"] = {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_cost_usd": round(total_cost, 6),
            "average_tokens_per_task": (total_input + total_output) / len(self.all_tasks),
            "average_cost_per_task_usd": round(total_cost / len(self.all_tasks), 6),
            "reasoning_turns": {
                "mean": statistics.mean(turns),
                "median": statistics.median(turns),
                "max": max(turns),
                "min": min(turns)
            },
            "max_turns_reached_count": sum(1 for t in self.all_tasks if t.max_turns_reached),
            "max_turns_reached_percent": (sum(1 for t in self.all_tasks if t.max_turns_reached) / len(self.all_tasks)) * 100
        }
        
        # Tool metrics
        all_tool_calls = []
        for task in self.all_tasks:
            for turn in task.turns:
                all_tool_calls.extend(turn.tool_calls)
        
        tool_distribution = defaultdict(int)
        tool_exec_times = defaultdict(list)
        tool_success = defaultdict(lambda: {"success": 0, "fail": 0})
        tool_errors_by_type = defaultdict(list)
        
        for tc in all_tool_calls:
            tool_distribution[tc.tool_name] += 1
            tool_exec_times[tc.tool_name].append(tc.execution_time_ms)
            if tc.success:
                tool_success[tc.tool_name]["success"] += 1
            else:
                tool_success[tc.tool_name]["fail"] += 1
                tool_errors_by_type[tc.tool_name].append(tc.error_message)
        
        summary["tool_metrics"] = {
            "total_tool_calls": len(all_tool_calls),
            "average_tool_calls_per_task": len(all_tool_calls) / len(self.all_tasks) if self.all_tasks else 0,
            "tool_call_distribution": dict(tool_distribution),
            "tool_call_distribution_percent": {
                k: (v / len(all_tool_calls)) * 100 for k, v in tool_distribution.items()
            } if all_tool_calls else {},
            "tool_execution_times_ms": {
                k: {
                    "mean": statistics.mean(v),
                    "median": statistics.median(v),
                    "min": min(v),
                    "max": max(v)
                } for k, v in tool_exec_times.items()
            },
            "tool_success_rates": {
                k: {
                    "success_rate_percent": (v["success"] / (v["success"] + v["fail"])) * 100,
                    "total_calls": v["success"] + v["fail"]
                } for k, v in tool_success.items()
            },
            "overall_tool_success_rate_percent": (
                sum(1 for tc in all_tool_calls if tc.success) / len(all_tool_calls) * 100
            ) if all_tool_calls else 0,
            "tool_errors": dict(tool_errors_by_type),
            "average_arguments_size_bytes": (
                statistics.mean([tc.arguments_size_bytes for tc in all_tool_calls])
            ) if all_tool_calls else 0
        }
        
        # Routing metrics
        public_tasks = [t for t in self.all_tasks if t.routed_zone == "public"]
        private_tasks = [t for t in self.all_tasks if t.routed_zone == "private"]
        correct_routings = [t for t in self.all_tasks if t.routing_correct is True]
        incorrect_routings = [t for t in self.all_tasks if t.routing_correct is False]
        
        summary["routing_metrics"] = {
            "public_zone_tasks": len(public_tasks),
            "private_zone_tasks": len(private_tasks),
            "public_zone_percent": (len(public_tasks) / len(self.all_tasks)) * 100 if self.all_tasks else 0,
            "private_zone_percent": (len(private_tasks) / len(self.all_tasks)) * 100 if self.all_tasks else 0,
            "routing_accuracy_percent": (
                len(correct_routings) / (len(correct_routings) + len(incorrect_routings)) * 100
            ) if (correct_routings or incorrect_routings) else None,
            "correct_routings": len(correct_routings),
            "incorrect_routings": len(incorrect_routings)
        }
        
        # Resource metrics
        peak_rams = [t.peak_ram_mb for t in self.all_tasks if t.peak_ram_mb > 0]
        avg_rams = [t.average_ram_mb for t in self.all_tasks if t.average_ram_mb > 0]
        peak_cpus = [t.peak_cpu_percent for t in self.all_tasks if t.peak_cpu_percent > 0]
        avg_cpus = [t.average_cpu_percent for t in self.all_tasks if t.average_cpu_percent > 0]
        
        summary["resource_metrics"] = {
            "ram_usage_mb": {
                "peak_max": max(peak_rams) if peak_rams else 0,
                "peak_mean": statistics.mean(peak_rams) if peak_rams else 0,
                "average_mean": statistics.mean(avg_rams) if avg_rams else 0
            },
            "cpu_usage_percent": {
                "peak_max": max(peak_cpus) if peak_cpus else 0,
                "peak_mean": statistics.mean(peak_cpus) if peak_cpus else 0,
                "average_mean": statistics.mean(avg_cpus) if avg_cpus else 0
            }
        }
        
        # Self-correction metrics
        retries = [t.retry_count for t in self.all_tasks]
        errors_recovered = [t.errors_recovered for t in self.all_tasks]
        code_regens = [t.code_regenerations for t in self.all_tasks]
        cmd_reexecs = [t.commands_reexecuted for t in self.all_tasks]
        backoff_times = [t.backoff_time_total_ms for t in self.all_tasks]
        
        summary["self_correction_metrics"] = {
            "total_retries": sum(retries),
            "average_retries_per_task": statistics.mean(retries) if retries else 0,
            "total_errors_recovered": sum(errors_recovered),
            "error_recovery_rate_percent": (
                sum(errors_recovered) / (sum(errors_recovered) + sum(len(t.errors) for t in self.all_tasks)) * 100
            ) if sum(errors_recovered) + sum(len(t.errors) for t in self.all_tasks) > 0 else 0,
            "total_code_regenerations": sum(code_regens),
            "total_commands_reexecuted": sum(cmd_reexecs),
            "total_backoff_time_ms": sum(backoff_times),
            "average_backoff_time_per_retry_ms": (
                sum(backoff_times) / sum(retries)
            ) if sum(retries) > 0 else 0
        }
        
        # Quality metrics
        completed = sum(1 for t in self.all_tasks if t.task_completed)
        correct = sum(1 for t in self.all_tasks if t.output_correct is True)
        incorrect = sum(1 for t in self.all_tasks if t.output_correct is False)
        exec_success = sum(1 for t in self.all_tasks if t.execution_success)
        hallucinations = sum(t.hallucinations_detected for t in self.all_tasks)
        
        summary["quality_metrics"] = {
            "task_completion_rate_percent": (completed / len(self.all_tasks)) * 100 if self.all_tasks else 0,
            "tasks_completed": completed,
            "tasks_incomplete": len(self.all_tasks) - completed,
            "output_correctness_rate_percent": (
                correct / (correct + incorrect) * 100
            ) if (correct + incorrect) > 0 else None,
            "correct_outputs": correct,
            "incorrect_outputs": incorrect,
            "unvalidated_outputs": len(self.all_tasks) - correct - incorrect,
            "execution_success_rate_percent": (exec_success / len(self.all_tasks)) * 100 if self.all_tasks else 0,
            "total_hallucinations_detected": hallucinations,
            "average_hallucinations_per_task": hallucinations / len(self.all_tasks) if self.all_tasks else 0
        }
        
        # Per-model breakdown
        models = set(t.model_used for t in self.all_tasks if t.model_used)
        for model in models:
            model_tasks = [t for t in self.all_tasks if t.model_used == model]
            summary["per_model_breakdown"][model] = {
                "task_count": len(model_tasks),
                "average_duration_ms": statistics.mean([t.total_duration_ms for t in model_tasks]),
                "total_tokens": sum(t.total_tokens.total_tokens for t in model_tasks),
                "total_cost_usd": round(sum(t.total_tokens.cost_usd for t in model_tasks), 6),
                "completion_rate_percent": (
                    sum(1 for t in model_tasks if t.task_completed) / len(model_tasks) * 100
                )
            }
        
        # Per-zone breakdown
        for zone in ["public", "private"]:
            zone_tasks = [t for t in self.all_tasks if t.routed_zone == zone]
            if zone_tasks:
                summary["per_zone_breakdown"][zone] = {
                    "task_count": len(zone_tasks),
                    "average_duration_ms": statistics.mean([t.total_duration_ms for t in zone_tasks]),
                    "total_tokens": sum(t.total_tokens.total_tokens for t in zone_tasks),
                    "completion_rate_percent": (
                        sum(1 for t in zone_tasks if t.task_completed) / len(zone_tasks) * 100
                    )
                }
        
        return summary
    
    def save_results(self, filename_prefix: str = "benchmark"):
        """Save all results to JSON files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual task results
        tasks_file = os.path.join(self.output_dir, f"{filename_prefix}_tasks_{timestamp}.json")
        with open(tasks_file, "w") as f:
            json.dump([t.to_dict() for t in self.all_tasks], f, indent=2)
        
        # Save summary
        summary_file = os.path.join(self.output_dir, f"{filename_prefix}_summary_{timestamp}.json")
        with open(summary_file, "w") as f:
            json.dump(self.generate_summary(), f, indent=2)
        
        return tasks_file, summary_file
    
    def print_summary(self):
        """Print a human-readable summary to console"""
        summary = self.generate_summary()
        
        print("\n" + "=" * 80)
        print(" IASCIS BENCHMARK SUMMARY")
        print("=" * 80)
        
        print(f"\n📊 BENCHMARK INFO")
        print(f"   Total Tasks: {summary['benchmark_info']['total_tasks']}")
        print(f"   Timestamp: {summary['benchmark_info']['timestamp']}")
        
        print(f"\n⏱️  PERFORMANCE METRICS")
        perf = summary["performance_metrics"]
        print(f"   Total Duration (mean): {perf['total_duration']['mean_ms']:.2f} ms")
        print(f"   Total Duration (median): {perf['total_duration']['median_ms']:.2f} ms")
        print(f"   Planning Time (mean): {perf['planning_time']['mean_ms']:.2f} ms")
        print(f"   Execution Time (mean): {perf['execution_time']['mean_ms']:.2f} ms")
        print(f"   Dispatcher Routing (mean): {perf['dispatcher_routing_time']['mean_ms']:.2f} ms")
        print(f"   TTFT (mean): {perf['time_to_first_token']['mean_ms']:.2f} ms")
        print(f"   Tasks/Minute: {perf['tasks_per_minute']:.2f}")
        
        print(f"\n🤖 LLM METRICS")
        llm = summary["llm_metrics"]
        print(f"   Total Tokens: {llm['total_tokens']:,}")
        print(f"   Input Tokens: {llm['total_input_tokens']:,}")
        print(f"   Output Tokens: {llm['total_output_tokens']:,}")
        print(f"   Total Cost: ${llm['total_cost_usd']:.6f}")
        print(f"   Avg Tokens/Task: {llm['average_tokens_per_task']:.0f}")
        print(f"   Reasoning Turns (mean): {llm['reasoning_turns']['mean']:.1f}")
        print(f"   Max Turns Reached: {llm['max_turns_reached_count']} ({llm['max_turns_reached_percent']:.1f}%)")
        
        print(f"\n🔧 TOOL METRICS")
        tools = summary["tool_metrics"]
        print(f"   Total Tool Calls: {tools['total_tool_calls']}")
        print(f"   Avg Calls/Task: {tools['average_tool_calls_per_task']:.1f}")
        print(f"   Overall Success Rate: {tools['overall_tool_success_rate_percent']:.1f}%")
        print(f"   Distribution: {tools['tool_call_distribution']}")
        
        print(f"\n🔀 ROUTING METRICS")
        routing = summary["routing_metrics"]
        print(f"   Public Zone: {routing['public_zone_tasks']} ({routing['public_zone_percent']:.1f}%)")
        print(f"   Private Zone: {routing['private_zone_tasks']} ({routing['private_zone_percent']:.1f}%)")
        if routing['routing_accuracy_percent'] is not None:
            print(f"   Routing Accuracy: {routing['routing_accuracy_percent']:.1f}%")
        
        print(f"\n💾 RESOURCE METRICS")
        res = summary["resource_metrics"]
        print(f"   Peak RAM: {res['ram_usage_mb']['peak_max']:.1f} MB")
        print(f"   Avg RAM: {res['ram_usage_mb']['average_mean']:.1f} MB")
        print(f"   Peak CPU: {res['cpu_usage_percent']['peak_max']:.1f}%")
        print(f"   Avg CPU: {res['cpu_usage_percent']['average_mean']:.1f}%")
        
        print(f"\n🔄 SELF-CORRECTION METRICS")
        sc = summary["self_correction_metrics"]
        print(f"   Total Retries: {sc['total_retries']}")
        print(f"   Errors Recovered: {sc['total_errors_recovered']}")
        print(f"   Code Regenerations: {sc['total_code_regenerations']}")
        print(f"   Commands Re-executed: {sc['total_commands_reexecuted']}")
        print(f"   Total Backoff Time: {sc['total_backoff_time_ms']:.0f} ms")
        
        print(f"\n✅ QUALITY METRICS")
        qual = summary["quality_metrics"]
        print(f"   Task Completion Rate: {qual['task_completion_rate_percent']:.1f}%")
        print(f"   Execution Success Rate: {qual['execution_success_rate_percent']:.1f}%")
        if qual['output_correctness_rate_percent'] is not None:
            print(f"   Output Correctness: {qual['output_correctness_rate_percent']:.1f}%")
        print(f"   Hallucinations Detected: {qual['total_hallucinations_detected']}")
        
        print("\n" + "=" * 80)
