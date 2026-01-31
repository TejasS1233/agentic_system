"""
IASCIS Benchmark Visualization & Export Script
Generates charts and exports to CSV/Excel for research papers.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd

# Set style for publication-quality plots
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


class BenchmarkVisualizer:
    """Generate visualizations from benchmark results."""

    def __init__(self, results_dir: str = "benchmark_results"):
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / "visualizations"
        self.output_dir.mkdir(exist_ok=True)

        # Load latest results
        self.summary = self._load_latest("summary")
        self.tasks = self._load_latest("tasks")

    def _load_latest(self, file_type: str) -> dict:
        """Load the most recent results file of given type."""
        pattern = f"benchmark_{file_type}_*.json"
        files = sorted(self.results_dir.glob(pattern), reverse=True)
        if not files:
            raise FileNotFoundError(f"No {file_type} files found in {self.results_dir}")

        with open(files[0]) as f:
            return json.load(f)

    def generate_all(self):
        """Generate all visualizations and exports."""
        print(f"Generating visualizations in {self.output_dir}...")

        # Generate charts
        self.plot_performance_breakdown()
        self.plot_token_usage()
        self.plot_tool_distribution()
        self.plot_routing_breakdown()
        self.plot_resource_usage()
        self.plot_quality_metrics()
        self.plot_per_task_comparison()
        self.plot_turns_distribution()

        # Export to spreadsheet formats
        self.export_to_csv()
        self.export_to_excel()

        print(f"\n All visualizations saved to: {self.output_dir}")
        return str(self.output_dir)

    def plot_performance_breakdown(self):
        """Bar chart of timing metrics."""
        fig, ax = plt.subplots(figsize=(10, 6))

        perf = self.summary["performance_metrics"]

        metrics = {
            "Planning": perf["planning_time"]["mean_ms"] / 1000,
            "Execution": perf["execution_time"]["mean_ms"] / 1000,
            "Dispatcher": perf["dispatcher_routing_time"]["mean_ms"] / 1000,
            "TTFT": perf["time_to_first_token"]["mean_ms"] / 1000,
        }

        colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]
        bars = ax.bar(
            metrics.keys(),
            metrics.values(),
            color=colors,
            edgecolor="black",
            linewidth=1.2,
        )

        ax.set_ylabel("Time (seconds)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Phase", fontsize=12, fontweight="bold")
        ax.set_title("IASCIS Performance Breakdown", fontsize=14, fontweight="bold")

        # Add value labels on bars
        for bar, val in zip(bars, metrics.values()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}s",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "01_performance_breakdown.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print("  ✓ Performance breakdown chart")

    def plot_token_usage(self):
        """Pie chart of token distribution."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        llm = self.summary["llm_metrics"]

        # Token distribution (input vs output)
        tokens = [llm["total_input_tokens"], llm["total_output_tokens"]]
        labels = [
            f"Input\n({llm['total_input_tokens']:,})",
            f"Output\n({llm['total_output_tokens']:,})",
        ]
        colors = ["#3498db", "#e74c3c"]

        ax1.pie(
            tokens,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            explode=(0.02, 0.02),
        )
        ax1.set_title("Token Distribution", fontsize=12, fontweight="bold")

        # Cost breakdown by model (if available)
        if self.summary.get("per_model_breakdown"):
            models = self.summary["per_model_breakdown"]
            model_names = list(models.keys())
            model_tokens = [m["total_tokens"] for m in models.values()]
            model_costs = [m["total_cost_usd"] for m in models.values()]

            x = range(len(model_names))
            width = 0.35

            ax2.bar(
                [i - width / 2 for i in x],
                [t / 1000 for t in model_tokens],
                width,
                label="Tokens (K)",
                color="#3498db",
            )

            ax2_twin = ax2.twinx()
            ax2_twin.bar(
                [i + width / 2 for i in x],
                [c * 1000 for c in model_costs],
                width,
                label="Cost (m$)",
                color="#2ecc71",
            )

            ax2.set_ylabel("Tokens (thousands)", fontsize=10)
            ax2_twin.set_ylabel("Cost (milli-dollars)", fontsize=10)
            ax2.set_xticks(x)
            ax2.set_xticklabels(
                [m.split("/")[-1] for m in model_names], rotation=15, ha="right"
            )
            ax2.set_title("Usage by Model", fontsize=12, fontweight="bold")

            # Combined legend
            ax2.legend(loc="upper left")
            ax2_twin.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "02_token_usage.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print("  ✓ Token usage chart")

    def plot_tool_distribution(self):
        """Bar chart of tool usage."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        tools = self.summary["tool_metrics"]

        # Tool call frequency
        tool_dist = tools.get("tool_call_distribution", {})
        if tool_dist:
            colors = sns.color_palette("husl", len(tool_dist))
            bars = ax1.bar(
                tool_dist.keys(), tool_dist.values(), color=colors, edgecolor="black"
            )
            ax1.set_ylabel("Number of Calls", fontsize=11, fontweight="bold")
            ax1.set_xlabel("Tool", fontsize=11, fontweight="bold")
            ax1.set_title("Tool Call Frequency", fontsize=12, fontweight="bold")

            for bar, val in zip(bars, tool_dist.values()):
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    str(val),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

        # Tool execution times
        exec_times = tools.get("tool_execution_times_ms", {})
        if exec_times:
            tool_names = list(exec_times.keys())
            means = [exec_times[t]["mean"] for t in tool_names]

            colors = sns.color_palette("husl", len(tool_names))
            bars = ax2.barh(tool_names, means, color=colors, edgecolor="black")
            ax2.set_xlabel("Mean Execution Time (ms)", fontsize=11, fontweight="bold")
            ax2.set_title("Tool Execution Times", fontsize=12, fontweight="bold")

            for bar, val in zip(bars, means):
                ax2.text(
                    bar.get_width() + 10,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}ms",
                    va="center",
                    fontsize=9,
                )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "03_tool_distribution.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print("  ✓ Tool distribution chart")

    def plot_routing_breakdown(self):
        """Pie chart of routing decisions."""
        fig, ax = plt.subplots(figsize=(8, 6))

        routing = self.summary["routing_metrics"]

        zones = ["Public Zone\n(Cloud)", "Private Zone\n(Local)"]
        counts = [routing["public_zone_tasks"], routing["private_zone_tasks"]]
        colors = ["#3498db", "#2ecc71"]
        explode = (0.03, 0.03)

        wedges, texts, autotexts = ax.pie(
            counts,
            labels=zones,
            colors=colors,
            autopct="%1.0f%%",
            startangle=90,
            explode=explode,
            shadow=True,
        )

        for autotext in autotexts:
            autotext.set_fontsize(12)
            autotext.set_fontweight("bold")

        # Add accuracy annotation
        accuracy = routing.get("routing_accuracy_percent")
        if accuracy is not None:
            ax.annotate(
                f"Routing Accuracy: {accuracy:.0f}%",
                xy=(0.5, -0.1),
                xycoords="axes fraction",
                ha="center",
                fontsize=12,
                fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        ax.set_title(
            "Privacy Zone Routing Distribution", fontsize=14, fontweight="bold"
        )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "04_routing_breakdown.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print("  ✓ Routing breakdown chart")

    def plot_resource_usage(self):
        """Line/bar chart of resource utilization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        resources = self.summary["resource_metrics"]

        # RAM usage
        ram = resources["ram_usage_mb"]
        categories = ["Peak Max", "Peak Mean", "Average"]
        values = [ram["peak_max"], ram["peak_mean"], ram["average_mean"]]

        colors = ["#e74c3c", "#f39c12", "#2ecc71"]
        bars = ax1.bar(
            categories, values, color=colors, edgecolor="black", linewidth=1.2
        )
        ax1.set_ylabel("RAM (MB)", fontsize=11, fontweight="bold")
        ax1.set_title("Memory Usage", fontsize=12, fontweight="bold")
        ax1.axhline(
            y=1024, color="red", linestyle="--", alpha=0.5, label="1GB Reference"
        )

        for bar, val in zip(bars, values):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                f"{val:.0f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        # CPU usage
        cpu = resources["cpu_usage_percent"]
        categories = ["Peak Max", "Peak Mean", "Average"]
        values = [cpu["peak_max"], cpu["peak_mean"], cpu["average_mean"]]

        colors = ["#e74c3c", "#f39c12", "#2ecc71"]
        bars = ax2.bar(
            categories, values, color=colors, edgecolor="black", linewidth=1.2
        )
        ax2.set_ylabel("CPU (%)", fontsize=11, fontweight="bold")
        ax2.set_title("CPU Utilization", fontsize=12, fontweight="bold")
        ax2.set_ylim(0, 100)

        for bar, val in zip(bars, values):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "05_resource_usage.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print("  ✓ Resource usage chart")

    def plot_quality_metrics(self):
        """Bar chart of quality metrics."""
        fig, ax = plt.subplots(figsize=(10, 6))

        quality = self.summary["quality_metrics"]

        metrics = {
            "Task\nCompletion": quality["task_completion_rate_percent"],
            "Execution\nSuccess": quality["execution_success_rate_percent"],
            "Tool\nSuccess": self.summary["tool_metrics"][
                "overall_tool_success_rate_percent"
            ],
            "Routing\nAccuracy": self.summary["routing_metrics"].get(
                "routing_accuracy_percent", 0
            )
            or 0,
        }

        colors = [
            "#2ecc71" if v >= 80 else "#f39c12" if v >= 50 else "#e74c3c"
            for v in metrics.values()
        ]
        bars = ax.bar(
            metrics.keys(),
            metrics.values(),
            color=colors,
            edgecolor="black",
            linewidth=1.2,
        )

        ax.set_ylabel("Percentage (%)", fontsize=12, fontweight="bold")
        ax.set_title("IASCIS Quality Metrics", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 110)
        ax.axhline(y=100, color="gray", linestyle="--", alpha=0.3)
        ax.axhline(y=80, color="green", linestyle="--", alpha=0.3, label="Target (80%)")

        for bar, val in zip(bars, metrics.values()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                f"{val:.0f}%",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "06_quality_metrics.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print("  ✓ Quality metrics chart")

    def plot_per_task_comparison(self):
        """Compare metrics across individual tasks."""
        if not self.tasks:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Extract task data
        task_ids = [t["task_id"] for t in self.tasks]
        durations = [t["total_duration_ms"] / 1000 for t in self.tasks]
        turns = [t["total_turns"] for t in self.tasks]
        tool_calls = [t["total_tool_calls"] for t in self.tasks]
        zones = [t["routed_zone"] for t in self.tasks]

        zone_colors = ["#3498db" if z == "public" else "#2ecc71" for z in zones]

        # Duration per task
        ax = axes[0, 0]
        ax.bar(range(len(task_ids)), durations, color=zone_colors, edgecolor="black")
        ax.set_xticks(range(len(task_ids)))
        ax.set_xticklabels(
            [t.split("_")[0] + "_" + t.split("_")[1] for t in task_ids],
            rotation=45,
            ha="right",
        )
        ax.set_ylabel("Duration (seconds)")
        ax.set_title("Duration per Task")

        # Turns per task
        ax = axes[0, 1]
        ax.bar(range(len(task_ids)), turns, color=zone_colors, edgecolor="black")
        ax.set_xticks(range(len(task_ids)))
        ax.set_xticklabels(
            [t.split("_")[0] + "_" + t.split("_")[1] for t in task_ids],
            rotation=45,
            ha="right",
        )
        ax.set_ylabel("Number of Turns")
        ax.set_title("Reasoning Turns per Task")
        ax.axhline(y=10, color="red", linestyle="--", alpha=0.5, label="Max Turns")

        # Tool calls per task
        ax = axes[1, 0]
        ax.bar(range(len(task_ids)), tool_calls, color=zone_colors, edgecolor="black")
        ax.set_xticks(range(len(task_ids)))
        ax.set_xticklabels(
            [t.split("_")[0] + "_" + t.split("_")[1] for t in task_ids],
            rotation=45,
            ha="right",
        )
        ax.set_ylabel("Number of Tool Calls")
        ax.set_title("Tool Calls per Task")

        # Zone distribution
        ax = axes[1, 1]
        zone_counts = {
            "public": zones.count("public"),
            "private": zones.count("private"),
        }
        ax.pie(
            zone_counts.values(),
            labels=zone_counts.keys(),
            colors=["#3498db", "#2ecc71"],
            autopct="%1.0f%%",
            startangle=90,
        )
        ax.set_title("Zone Distribution")

        # Add legend
        public_patch = mpatches.Patch(color="#3498db", label="Public (Cloud)")
        private_patch = mpatches.Patch(color="#2ecc71", label="Private (Local)")
        fig.legend(handles=[public_patch, private_patch], loc="upper right")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "07_per_task_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print("  ✓ Per-task comparison chart")

    def plot_turns_distribution(self):
        """Histogram of turns distribution."""
        if not self.tasks:
            return

        fig, ax = plt.subplots(figsize=(8, 5))

        turns = [t["total_turns"] for t in self.tasks]

        ax.hist(turns, bins=range(1, 12), color="#3498db", edgecolor="black", alpha=0.7)
        ax.axvline(
            x=sum(turns) / len(turns),
            color="red",
            linestyle="--",
            label=f"Mean: {sum(turns) / len(turns):.1f}",
        )
        ax.axvline(x=10, color="orange", linestyle="--", label="Max Turns Limit")

        ax.set_xlabel("Number of Turns", fontsize=11, fontweight="bold")
        ax.set_ylabel("Frequency", fontsize=11, fontweight="bold")
        ax.set_title("Distribution of Reasoning Turns", fontsize=12, fontweight="bold")
        ax.legend()

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "08_turns_distribution.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print("  ✓ Turns distribution chart")

    def export_to_csv(self):
        """Export summary to CSV files."""
        csv_dir = self.output_dir / "csv"
        csv_dir.mkdir(exist_ok=True)

        # Summary metrics
        summary_data = []
        for category, metrics in self.summary.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    if isinstance(value, dict):
                        for sub_metric, sub_value in value.items():
                            summary_data.append(
                                {
                                    "Category": category,
                                    "Metric": f"{metric}.{sub_metric}",
                                    "Value": sub_value,
                                }
                            )
                    else:
                        summary_data.append(
                            {"Category": category, "Metric": metric, "Value": value}
                        )

        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(csv_dir / "summary_metrics.csv", index=False)

        # Task-level data
        if self.tasks:
            task_data = []
            for t in self.tasks:
                task_data.append(
                    {
                        "task_id": t["task_id"],
                        "category": t["task_category"],
                        "complexity": t["complexity_level"],
                        "routed_zone": t["routed_zone"],
                        "model_used": t["model_used"],
                        "duration_ms": t["total_duration_ms"],
                        "planning_time_ms": t["planning_time_ms"],
                        "execution_time_ms": t["execution_time_ms"],
                        "total_turns": t["total_turns"],
                        "total_tool_calls": t["total_tool_calls"],
                        "input_tokens": t["total_tokens"]["input_tokens"],
                        "output_tokens": t["total_tokens"]["output_tokens"],
                        "cost_usd": t["total_tokens"]["cost_usd"],
                        "completed": t["task_completed"],
                        "peak_ram_mb": t["peak_ram_mb"],
                        "peak_cpu_percent": t["peak_cpu_percent"],
                    }
                )

            df_tasks = pd.DataFrame(task_data)
            df_tasks.to_csv(csv_dir / "task_metrics.csv", index=False)

        print(f"  ✓ CSV files exported to {csv_dir}")

    def export_to_excel(self):
        """Export all data to a single Excel file with multiple sheets."""
        excel_path = self.output_dir / "benchmark_results.xlsx"

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            # Summary sheet
            summary_data = []
            for category, metrics in self.summary.items():
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        if isinstance(value, dict):
                            for sub_metric, sub_value in value.items():
                                summary_data.append(
                                    {
                                        "Category": category,
                                        "Metric": metric,
                                        "Sub-Metric": sub_metric,
                                        "Value": sub_value,
                                    }
                                )
                        else:
                            summary_data.append(
                                {
                                    "Category": category,
                                    "Metric": metric,
                                    "Sub-Metric": "-",
                                    "Value": value,
                                }
                            )

            pd.DataFrame(summary_data).to_excel(
                writer, sheet_name="Summary", index=False
            )

            # Tasks sheet
            if self.tasks:
                task_rows = []
                for t in self.tasks:
                    task_rows.append(
                        {
                            "Task ID": t["task_id"],
                            "Description": t["task_description"][:50] + "...",
                            "Category": t["task_category"],
                            "Complexity": t["complexity_level"],
                            "Zone": t["routed_zone"],
                            "Model": t["model_used"].split("/")[-1],
                            "Duration (s)": round(t["total_duration_ms"] / 1000, 2),
                            "Turns": t["total_turns"],
                            "Tool Calls": t["total_tool_calls"],
                            "Tokens": t["total_tokens"]["total_tokens"],
                            "Cost ($)": round(t["total_tokens"]["cost_usd"], 6),
                            "Completed": "Yes" if t["task_completed"] else "No",
                            "RAM (MB)": round(t["peak_ram_mb"], 1),
                            "CPU (%)": round(t["peak_cpu_percent"], 1),
                        }
                    )

                pd.DataFrame(task_rows).to_excel(
                    writer, sheet_name="Tasks", index=False
                )

            # Tool metrics sheet
            tools = self.summary.get("tool_metrics", {})
            if tools.get("tool_call_distribution"):
                tool_data = []
                for tool, count in tools["tool_call_distribution"].items():
                    exec_time = tools.get("tool_execution_times_ms", {}).get(tool, {})
                    success = tools.get("tool_success_rates", {}).get(tool, {})
                    tool_data.append(
                        {
                            "Tool": tool,
                            "Calls": count,
                            "Mean Time (ms)": round(exec_time.get("mean", 0), 2),
                            "Max Time (ms)": round(exec_time.get("max", 0), 2),
                            "Success Rate (%)": round(
                                success.get("success_rate_percent", 0), 1
                            ),
                        }
                    )

                pd.DataFrame(tool_data).to_excel(
                    writer, sheet_name="Tools", index=False
                )

        print(f"  ✓ Excel file exported: {excel_path}")


class BaselineComparison:
    """
    Compare benchmark results across different modes/baselines.
    Generates comparison visualizations for research papers.
    """

    def __init__(self, results_dir: str = "benchmark_results"):
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / "comparison"
        self.output_dir.mkdir(exist_ok=True)

        # Load results from each mode subdirectory
        self.mode_results = {}
        self.modes = ["iascis", "cloud", "local", "single", "no_planning"]

        for mode in self.modes:
            mode_dir = self.results_dir / mode
            if mode_dir.exists():
                try:
                    summary_files = sorted(
                        mode_dir.glob("benchmark_summary_*.json"), reverse=True
                    )
                    if summary_files:
                        with open(summary_files[0]) as f:
                            self.mode_results[mode] = json.load(f)
                except Exception as e:
                    print(f"  Warning: Could not load {mode} results: {e}")

    def generate_comparison(self):
        """Generate all comparison visualizations."""
        if len(self.mode_results) < 2:
            print(
                f"⚠️  Need at least 2 modes to compare. Found: {list(self.mode_results.keys())}"
            )
            return

        print(
            f"Generating comparison charts for modes: {list(self.mode_results.keys())}..."
        )

        self.plot_duration_comparison()
        self.plot_cost_comparison()
        self.plot_turns_comparison()
        self.plot_quality_comparison()
        self.export_comparison_table()

        print(f"\n✅ Comparison charts saved to: {self.output_dir}")

    def plot_duration_comparison(self):
        """Compare total duration across modes."""
        fig, ax = plt.subplots(figsize=(10, 6))

        modes = []
        durations = []

        for mode, data in self.mode_results.items():
            modes.append(mode.upper())
            durations.append(
                data["performance_metrics"]["total_duration"]["mean_ms"] / 1000
            )

        colors = sns.color_palette("husl", len(modes))
        bars = ax.bar(modes, durations, color=colors, edgecolor="black", linewidth=1.5)

        # Highlight IASCIS
        if "IASCIS" in modes:
            idx = modes.index("IASCIS")
            bars[idx].set_edgecolor("gold")
            bars[idx].set_linewidth(3)

        ax.set_ylabel("Mean Duration (seconds)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Execution Mode", fontsize=12, fontweight="bold")
        ax.set_title(
            "Duration Comparison Across Baselines", fontsize=14, fontweight="bold"
        )

        for bar, val in zip(bars, durations):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                f"{val:.1f}s",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "comparison_01_duration.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print("  ✓ Duration comparison chart")

    def plot_cost_comparison(self):
        """Compare cost across modes."""
        fig, ax = plt.subplots(figsize=(10, 6))

        modes = []
        costs = []

        for mode, data in self.mode_results.items():
            modes.append(mode.upper())
            costs.append(
                data["llm_metrics"]["total_cost_usd"] * 1000
            )  # Convert to milli-dollars

        colors = sns.color_palette("husl", len(modes))
        bars = ax.bar(modes, costs, color=colors, edgecolor="black", linewidth=1.5)

        ax.set_ylabel("Total Cost (milli-dollars)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Execution Mode", fontsize=12, fontweight="bold")
        ax.set_title("Cost Comparison Across Baselines", fontsize=14, fontweight="bold")

        for bar, val in zip(bars, costs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"${val:.2f}m",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "comparison_02_cost.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print("  ✓ Cost comparison chart")

    def plot_turns_comparison(self):
        """Compare reasoning turns across modes."""
        fig, ax = plt.subplots(figsize=(10, 6))

        modes = []
        turns = []

        for mode, data in self.mode_results.items():
            modes.append(mode.upper())
            turns.append(data["llm_metrics"]["reasoning_turns"]["mean"])

        colors = sns.color_palette("husl", len(modes))
        bars = ax.bar(modes, turns, color=colors, edgecolor="black", linewidth=1.5)

        ax.set_ylabel("Mean Reasoning Turns", fontsize=12, fontweight="bold")
        ax.set_xlabel("Execution Mode", fontsize=12, fontweight="bold")
        ax.set_title("Reasoning Effort Comparison", fontsize=14, fontweight="bold")
        ax.axhline(
            y=10, color="red", linestyle="--", alpha=0.5, label="Max Turns Limit"
        )

        for bar, val in zip(bars, turns):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.legend()
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "comparison_03_turns.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print("  ✓ Turns comparison chart")

    def plot_quality_comparison(self):
        """Compare quality metrics across modes."""
        fig, ax = plt.subplots(figsize=(12, 6))

        modes = list(self.mode_results.keys())

        x = range(len(modes))
        width = 0.35

        completion_rates = []
        tool_rates = []

        for mode in modes:
            data = self.mode_results[mode]
            completion_rates.append(
                data["quality_metrics"]["task_completion_rate_percent"]
            )
            tool_rates.append(data["tool_metrics"]["overall_tool_success_rate_percent"])

        ax.bar(
            [i - width / 2 for i in x],
            completion_rates,
            width,
            label="Task Completion %",
            color="#2ecc71",
            edgecolor="black",
        )
        ax.bar(
            [i + width / 2 for i in x],
            tool_rates,
            width,
            label="Tool Success %",
            color="#3498db",
            edgecolor="black",
        )

        ax.set_ylabel("Success Rate (%)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Execution Mode", fontsize=12, fontweight="bold")
        ax.set_title("Quality Metrics Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in modes])
        ax.set_ylim(0, 110)
        ax.legend()

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "comparison_04_quality.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print("  ✓ Quality comparison chart")

    def export_comparison_table(self):
        """Export comparison data to CSV and Excel."""
        comparison_data = []

        for mode, data in self.mode_results.items():
            perf = data["performance_metrics"]
            llm = data["llm_metrics"]
            quality = data["quality_metrics"]
            tools = data["tool_metrics"]

            comparison_data.append(
                {
                    "Mode": mode.upper(),
                    "Duration (s)": round(perf["total_duration"]["mean_ms"] / 1000, 2),
                    "Planning (s)": round(perf["planning_time"]["mean_ms"] / 1000, 2),
                    "Execution (s)": round(perf["execution_time"]["mean_ms"] / 1000, 2),
                    "TTFT (s)": round(perf["time_to_first_token"]["mean_ms"] / 1000, 2),
                    "Total Tokens": llm["total_tokens"],
                    "Cost ($)": round(llm["total_cost_usd"], 6),
                    "Avg Turns": round(llm["reasoning_turns"]["mean"], 1),
                    "Task Completion %": round(
                        quality["task_completion_rate_percent"], 1
                    ),
                    "Tool Success %": round(
                        tools["overall_tool_success_rate_percent"], 1
                    ),
                }
            )

        df = pd.DataFrame(comparison_data)

        # Save as CSV
        df.to_csv(self.output_dir / "baseline_comparison.csv", index=False)

        # Save as Excel
        df.to_excel(self.output_dir / "baseline_comparison.xlsx", index=False)

        print("  ✓ Comparison table exported")

        # Print to console
        print("\n" + "=" * 80)
        print(" BASELINE COMPARISON TABLE")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="IASCIS Benchmark Visualizer")
    parser.add_argument(
        "--results-dir",
        "-r",
        default="benchmark_results",
        help="Directory containing benchmark results",
    )
    parser.add_argument(
        "--compare",
        "-c",
        action="store_true",
        help="Generate comparison charts across all baselines",
    )

    args = parser.parse_args()

    if args.compare:
        # Run comparison across modes
        comparator = BaselineComparison(args.results_dir)
        comparator.generate_comparison()
    else:
        # Standard single-mode visualization
        visualizer = BenchmarkVisualizer(args.results_dir)
        output_dir = visualizer.generate_all()

        print("\n📊 Visualizations ready for your research paper!")
        print(f"📁 Location: {output_dir}")


if __name__ == "__main__":
    main()
