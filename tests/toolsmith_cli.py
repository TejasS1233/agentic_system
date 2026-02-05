"""
Toolsmith CLI with Tool Decay Management

Interactive CLI for creating, managing, and monitoring tools with
decay-based lifecycle management.
"""

import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getcwd())

from architecture.toolsmith import Toolsmith  # noqa: E402
from execution.tool_decay import create_decay_manager  # noqa: E402


class ToolsmithCLI:
    """Interactive CLI with decay-aware tool management."""

    def __init__(self):
        self.toolsmith = Toolsmith()

        # Initialize decay manager for tracking tool usage
        self.decay_manager = create_decay_manager(
            decay_minutes=30.0,  # Tools decay after 30 min of inactivity
            protected_tools=[],  # Will be populated from registry
            auto_cleanup=True,
            cleanup_interval=30.0,  # Check every 30 seconds for visible decay updates
            max_capacity=50,
        )

        # Load existing tools into decay manager
        self._load_tools_from_registry()

    def _load_tools_from_registry(self):
        """Load existing tools from registry into the decay manager with historical timestamps."""
        tools = self.toolsmith.list_available_tools()

        # Get historical timestamps from metrics.json
        creation_times = self.toolsmith.get_tool_creation_timestamps()
        last_used_times = self.toolsmith.get_tool_last_used_timestamps()

        loaded_count = 0
        for tool in tools:
            tool_name = tool["name"]

            # Get timestamps if available
            created_at = creation_times.get(tool_name)
            last_used = last_used_times.get(tool_name)

            # Register each tool with semantic grouping based on domain
            self.decay_manager.register_tool(
                name=tool_name,
                tool=tool,  # Store the metadata as the "tool"
                protected=False,
                semantic_group=tool.get("domain", "general"),
                created_at=created_at,
                last_used=last_used,
            )
            loaded_count += 1

            # Log if using historical timestamps
            if created_at:
                import time

                age_days = (time.time() - created_at) / 86400
                print(f"[Decay] {tool_name}: created {age_days:.1f} days ago")

        print(
            f"[Decay] Loaded {loaded_count} tools from registry (with historical timestamps)"
        )

    def _record_tool_creation(self, tool_name: str, duration_ms: float):
        """Record a tool creation event for decay tracking."""
        # Register the new tool if it doesn't exist
        if tool_name not in self.decay_manager:
            self.decay_manager.register_tool(
                name=tool_name,
                tool={"name": tool_name, "created_via": "cli"},
                protected=False,
            )

        # Record the usage with execution time
        self.decay_manager.record_usage(name=tool_name, execution_time_ms=duration_ms)

    def _record_tool_install(self, tool_name: str):
        """Record a tool installation for decay tracking."""
        if tool_name not in self.decay_manager:
            self.decay_manager.register_tool(
                name=tool_name,
                tool={"name": tool_name, "installed_via": "cli"},
                protected=False,
            )
        self.decay_manager.record_usage(tool_name)

    def show_help(self):
        """Display help information."""
        print("\n=== Toolsmith CLI with Decay Management ===")
        print("Commands:")
        print("  <description>              - Create a new tool from description")
        print("  install <package-name>     - Install tool from PyPI")
        print("  list                       - List all registered tools")
        print("  status                     - Show decay status of all tools")
        print("  stats                      - Show cache statistics")
        print("  tier                       - Show tools by tier (Hot/Warm/Cold)")
        print("  protect <tool-name>        - Protect a tool from decay")
        print("  unprotect <tool-name>      - Remove protection from a tool")
        print("  eviction                   - Show eviction candidates")
        print("  cleanup                    - Force cleanup of expired tools")
        print("  exit                       - Quit")
        print("\nExamples:")
        print("  'A tool to calculate fibonacci numbers'")
        print("  'install factorial-tool-ts'")
        print("  'status'")
        print("")

    def cmd_list(self):
        """List all registered tools with decay info."""
        tools = self.toolsmith.list_available_tools()
        if not tools:
            print("No tools registered.")
            return

        print("\n--- Registered Tools ---")
        for t in tools:
            pypi = f" [PyPI: {t['pypi_package']}]" if t.get("pypi_package") else ""

            # Get decay metrics if available
            metrics = self.decay_manager.get_metrics(t["name"])
            if metrics:
                score = metrics.calculate_decay_score()
                tier = metrics.tier.value if hasattr(metrics, "tier") else "unknown"
                decay_info = f" | Score: {score:.3f} | Tier: {tier}"
            else:
                decay_info = " | Not tracked"

            print(f"  {t['name']}: {t['file']}{pypi}{decay_info}")

    def cmd_status(self):
        """Show decay status report."""
        print("\n" + self.decay_manager.get_status_report())

    def cmd_stats(self):
        """Show cache statistics."""
        stats = self.decay_manager.get_statistics()
        print("\n--- Cache Statistics ---")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    def cmd_tier(self):
        """Show tools organized by tier."""
        breakdown = self.decay_manager.get_tier_breakdown()
        print("\n--- Tools by Tier ---")
        for tier, tools in breakdown.items():
            if tools:
                print(f"\n  [{tier.upper()}]")
                for tool in tools:
                    metrics = self.decay_manager.get_metrics(tool)
                    if metrics:
                        print(f"    - {tool} (calls: {metrics.total_calls})")
                    else:
                        print(f"    - {tool}")

    def cmd_protect(self, tool_name: str):
        """Protect a tool from decay."""
        if self.decay_manager.protect_tool(tool_name):
            print(f"✓ Tool '{tool_name}' is now protected from decay")
        else:
            print(f"✗ Tool '{tool_name}' not found")

    def cmd_unprotect(self, tool_name: str):
        """Remove protection from a tool."""
        if self.decay_manager.unprotect_tool(tool_name):
            print(f"✓ Tool '{tool_name}' is no longer protected")
        else:
            print(f"✗ Tool '{tool_name}' not found or not protected")

    def cmd_eviction(self):
        """Show eviction candidates."""
        candidates = self.decay_manager.get_eviction_candidates(10)
        if not candidates:
            print("No eviction candidates.")
            return

        def format_time(seconds):
            """Format seconds into human-readable time."""
            if seconds >= 86400:
                return f"{seconds / 86400:.1f}d"
            elif seconds >= 3600:
                return f"{seconds / 3600:.1f}h"
            elif seconds >= 60:
                return f"{seconds / 60:.1f}m"
            else:
                return f"{seconds:.0f}s"

        print("\n--- Eviction Candidates (lowest scores first) ---")
        for name, score in candidates:
            metrics = self.decay_manager.get_metrics(name)
            if metrics:
                last_used_ago = format_time(metrics.time_since_use)
                age = format_time(metrics.age)
                print(
                    f"  {name}: score={score:.4f}, last_used={last_used_ago} ago, age={age}, calls={metrics.total_calls}"
                )
            else:
                print(f"  {name}: score={score:.4f}")

    def cmd_cleanup(self):
        """Force cleanup of expired and low-performing tools."""
        expired = self.decay_manager.cleanup_expired_tools()
        low_perf = self.decay_manager.cleanup_low_performers(min_score=0.001)

        if expired:
            print(f"Expired tools removed: {', '.join(expired)}")
        if low_perf:
            print(f"Low-performing tools removed: {', '.join(low_perf)}")
        if not expired and not low_perf:
            print("No tools needed cleanup.")

    def cmd_install(self, package_name: str):
        """Install a tool from PyPI."""
        result = self.toolsmith.install_tool(package_name)

        # Extract tool name from package name (remove -ts suffix if present)
        tool_name = package_name.replace("-ts", "").replace("-", "_")
        self._record_tool_install(tool_name)

        print(f"\n{result}")

    def cmd_create(self, requirement: str):
        """Create a new tool from description."""
        print("\n[Thinking] Sending request to Gemini 2.5 Flash...")

        start_time = time.time()
        result = self.toolsmith.create_tool(requirement)
        duration_ms = (time.time() - start_time) * 1000

        # Try to extract tool name from result
        created = "created" in result.lower() or "saved" in result.lower()

        # Extract tool name if possible (look for class name pattern)
        import re

        match = re.search(r"class\s+(\w+)", result)
        tool_name = match.group(1) if match else f"tool_{int(time.time())}"

        self._record_tool_creation(tool_name, duration_ms)

        print("\n----- RESULT -----")
        print(result)
        print("------------------")

        if created:
            print(
                f"\n[Decay] Tool '{tool_name}' registered (creation time: {duration_ms:.0f}ms)"
            )

    def run(self):
        """Run the interactive CLI."""
        self.show_help()

        while True:
            try:
                req = input("\n> ").strip()
                if not req:
                    continue

                # Parse command
                cmd_lower = req.lower()

                if cmd_lower in ["exit", "quit"]:
                    self.shutdown()
                    break
                elif cmd_lower == "help":
                    self.show_help()
                elif cmd_lower == "list":
                    self.cmd_list()
                elif cmd_lower == "status":
                    self.cmd_status()
                elif cmd_lower == "stats":
                    self.cmd_stats()
                elif cmd_lower == "tier":
                    self.cmd_tier()
                elif cmd_lower == "eviction":
                    self.cmd_eviction()
                elif cmd_lower == "cleanup":
                    self.cmd_cleanup()
                elif cmd_lower.startswith("protect "):
                    tool_name = req[8:].strip()
                    self.cmd_protect(tool_name)
                elif cmd_lower.startswith("unprotect "):
                    tool_name = req[10:].strip()
                    self.cmd_unprotect(tool_name)
                elif cmd_lower.startswith("install "):
                    package_name = req[8:].strip()
                    if package_name:
                        self.cmd_install(package_name)
                    else:
                        print("Usage: install <package-name>")
                else:
                    # Treat as tool creation request
                    self.cmd_create(req)

            except KeyboardInterrupt:
                print("\nExiting...")
                self.shutdown()
                break
            except Exception as e:
                print(f"Error: {e}")

    def shutdown(self):
        """Clean shutdown."""
        print("\n[Decay] Stopping background cleanup...")
        self.decay_manager.stop_background_cleanup()

        # Show final stats
        stats = self.decay_manager.get_statistics()
        print(
            f"[Decay] Final stats: {stats.get('cache_size', 0)} tools tracked, "
            f"{stats.get('cache_hits', 0)} hits, {stats.get('cache_misses', 0)} misses"
        )
        print("Goodbye!")


def interactive_mode():
    """Legacy function for backward compatibility."""
    cli = ToolsmithCLI()
    cli.run()


if __name__ == "__main__":
    interactive_mode()
