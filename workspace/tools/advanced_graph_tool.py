"""Advanced Graphing Tool - Create professional charts with matplotlib."""

import os
import json
from typing import Literal, Optional, Dict, List, Any, Union
from pydantic import BaseModel, Field
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


class AdvancedGraphArgs(BaseModel):
    chart_type: Literal["bar", "line", "scatter", "pie", "histogram", "heatmap", "box", "area", "horizontal_bar"] = Field(
        ..., description="Type of chart to create"
    )
    data: Union[Dict[str, List], List] = Field(
        ..., 
        description="Data for the chart. For bar/line/scatter: {'labels': [...], 'values': [...]} or "
                    "{'x': [...], 'y': [...], 'y2': [...]}. For pie: {'labels': [...], 'values': [...]}. "
                    "For histogram: [values...]. For heatmap: 2D array."
    )
    title: str = Field(..., description="Title of the chart")
    xlabel: Optional[str] = Field(None, description="Label for x-axis")
    ylabel: Optional[str] = Field(None, description="Label for y-axis")
    output_path: Optional[str] = Field(None, description="Output file path (default: chart_<timestamp>.png)")
    theme: Literal["default", "dark", "minimal", "colorful"] = Field(
        "default", description="Chart theme/style"
    )
    figsize: Optional[List[int]] = Field(None, description="Figure size as [width, height] in inches")
    colors: Optional[List[str]] = Field(None, description="Custom colors for the chart")
    legend_labels: Optional[List[str]] = Field(None, description="Labels for legend (multi-series)")
    show_values: bool = Field(False, description="Show values on bars/points")
    grid: bool = Field(True, description="Show grid lines")


class AdvancedGraphTool:
    """
    Advanced charting tool using matplotlib.
    
    Supports multiple chart types:
    - bar: Vertical bar chart
    - horizontal_bar: Horizontal bar chart
    - line: Line chart (supports multiple series)
    - scatter: Scatter plot
    - pie: Pie chart with percentages
    - histogram: Frequency distribution
    - heatmap: 2D heatmap
    - box: Box plot
    - area: Stacked area chart
    
    Features:
    - Multiple themes (default, dark, minimal, colorful)
    - Custom colors
    - Multi-series support
    - Value labels
    - Grid control
    - Auto-saving to PNG
    """
    
    name = "advanced_graph"
    description = """Create professional charts and graphs. Supports bar, line, scatter, pie, histogram, 
    heatmap, box, and area charts. Customize with themes, colors, and labels. Saves as PNG."""
    args_schema = AdvancedGraphArgs
    
    # Color palettes for different themes
    PALETTES = {
        "default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"],
        "dark": ["#00d9ff", "#ff6b6b", "#69db7c", "#ffd93d", "#a855f7", "#fb923c", "#38bdf8", "#f472b6"],
        "minimal": ["#374151", "#6b7280", "#9ca3af", "#d1d5db", "#4b5563", "#1f2937", "#111827", "#e5e7eb"],
        "colorful": ["#ef4444", "#f97316", "#eab308", "#22c55e", "#06b6d4", "#3b82f6", "#8b5cf6", "#ec4899"],
    }

    def __init__(self, output_dir: str = "/output"):
        self.output_dir = output_dir
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError:
            self.output_dir = "/tmp"  # Fallback to /tmp if can't create

    def _normalize_data(self, data: Union[Dict, List, str]) -> Union[Dict[str, List], List]:
        """
        Normalize various data formats into the expected structure.
        Handles: JSON strings, GitHub trending output, arbitrary dicts, etc.
        """
        # If it's a string, try to parse as JSON
        if isinstance(data, str):
            try:
                import json
                data = json.loads(data)
            except (json.JSONDecodeError, TypeError):
                # If not JSON, try to parse as simple "Label1 Value1, Label2 Value2" format
                parts = [p.strip() for p in data.replace(',', ' ').split() if p.strip()]
                labels = []
                values = []
                for i, part in enumerate(parts):
                    try:
                        val = float(part)
                        if labels and len(values) < len(labels):
                            values.append(val)
                    except ValueError:
                        labels.append(part)
                if labels and values:
                    return {"labels": labels[:len(values)], "values": values}
                return {"labels": ["No data"], "values": [0]}
        
        # If already in expected format, return as-is
        if isinstance(data, dict):
            if "labels" in data and "values" in data:
                return data
            if "x" in data and "y" in data:
                return data
            
            # Handle comparison/time-series format:
            # {"comparison": {"dates": [...], "AAPL": [...], "GOOGL": [...], ...}}
            # or {"dates": [...], "AAPL": [...], "GOOGL": [...], ...}
            comparison = data.get("comparison", None)
            if comparison and isinstance(comparison, dict) and "dates" in comparison:
                dates = comparison["dates"]
                series_keys = [k for k in comparison if k != "dates"]
                if dates and series_keys:
                    result = {"x": dates}
                    for i, key in enumerate(series_keys):
                        y_key = "y" if i == 0 else f"y{i + 1}"
                        result[y_key] = comparison[key]
                        result[f"_label_{y_key}"] = key  # Store series name for legend
                    return result
            
            # Handle flat time-series: {"dates": [...], "AAPL": [...], "MSFT": [...]}
            if "dates" in data:
                dates = data["dates"]
                series_keys = [k for k in data if k != "dates" and isinstance(data[k], list)
                               and data[k] and isinstance(data[k][0], (int, float))]
                if dates and series_keys:
                    result = {"x": dates}
                    for i, key in enumerate(series_keys):
                        y_key = "y" if i == 0 else f"y{i + 1}"
                        result[y_key] = data[key]
                        result[f"_label_{y_key}"] = key
                    return result
            
            # Handle nested stock data format:
            # {"data": {"AAPL": {"dates": [...], "close": [...]}, ...}}
            if "data" in data and isinstance(data["data"], dict):
                nested = data["data"]
                first_key = next(iter(nested), None)
                if first_key and isinstance(nested[first_key], dict) and "dates" in nested[first_key]:
                    result = {"x": nested[first_key]["dates"]}
                    for i, (symbol, info) in enumerate(nested.items()):
                        if isinstance(info, dict) and "close" in info:
                            y_key = "y" if i == 0 else f"y{i + 1}"
                            result[y_key] = info["close"]
                            result[f"_label_{y_key}"] = symbol
                    if len([k for k in result if k.startswith("y")]) > 0:
                        return result
            
            # Handle GitHub trending format: {"repos": [{"name": ..., "stars": ...}]}
            if "repos" in data and isinstance(data["repos"], list):
                repos = data["repos"][:10]  # Limit to 10
                return {
                    "labels": [r.get("name", r.get("full_name", "unknown")) for r in repos],
                    "values": [r.get("stars", r.get("stargazers_count", 0)) for r in repos]
                }
            
            # Handle generic list of dicts - try to extract name/value pairs
            for key in data:
                if isinstance(data[key], list) and data[key]:
                    items = data[key][:10]
                    if isinstance(items[0], dict):
                        # Find a name-like field and a numeric field
                        name_fields = ["name", "title", "label", "id", "username", "full_name"]
                        value_fields = ["stars", "count", "value", "score", "forks", "followers", "size"]
                        
                        name_key = next((f for f in name_fields if f in items[0]), None)
                        value_key = next((f for f in value_fields if f in items[0]), None)
                        
                        if name_key and value_key:
                            return {
                                "labels": [str(item.get(name_key, "")) for item in items],
                                "values": [item.get(value_key, 0) for item in items]
                            }
        
        # If it's a list of numbers, use indices as labels
        if isinstance(data, list) and data and isinstance(data[0], (int, float)):
            return data  # For histogram
        
        # Handle list of dicts (e.g., from CSV parsing) — aggregate by first string column
        if isinstance(data, list) and data and isinstance(data[0], dict):
            items = data
            # Identify string (label) columns and numeric (value) columns
            str_cols = [k for k, v in items[0].items() if isinstance(v, str)]
            num_cols = [k for k, v in items[0].items() if isinstance(v, (int, float))]
            
            if str_cols and num_cols:
                # Use first string column as grouping key and first numeric column as values
                label_col = str_cols[0]
                value_col = num_cols[0]
                
                # Aggregate: sum values by label
                aggregated = {}
                for row in items:
                    key = str(row.get(label_col, ""))
                    val = row.get(value_col, 0)
                    if isinstance(val, (int, float)):
                        aggregated[key] = aggregated.get(key, 0) + val
                
                return {
                    "labels": list(aggregated.keys()),
                    "values": list(aggregated.values())
                }
            elif num_cols:
                # No string cols — use first numeric column as values
                return [row.get(num_cols[0], 0) for row in items]
        
        # Fallback: return as-is and let chart methods handle it
        return data
    
    def run(
        self,
        chart_type: str,
        data: Union[Dict[str, List], List],
        title: str,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        output_path: Optional[str] = None,
        theme: str = "default",
        figsize: Optional[List[int]] = None,
        colors: Optional[List[str]] = None,
        legend_labels: Optional[List[str]] = None,
        show_values: bool = False,
        grid: bool = True
    ) -> Dict[str, Any]:
        """Create a chart and save it as PNG."""
        
        # Normalize input data to expected format
        data = self._normalize_data(data)
        
        try:
            # Set theme
            self._apply_theme(theme)
            
            # Create figure
            fig_size = tuple(figsize) if figsize else (10, 6)
            fig, ax = plt.subplots(figsize=fig_size)
            
            # Get colors
            palette = colors or self.PALETTES.get(theme, self.PALETTES["default"])
            
            # Create chart based on type
            if chart_type == "bar":
                self._create_bar_chart(ax, data, palette, show_values)
            elif chart_type == "horizontal_bar":
                self._create_horizontal_bar(ax, data, palette, show_values)
            elif chart_type == "line":
                self._create_line_chart(ax, data, palette, legend_labels)
            elif chart_type == "scatter":
                self._create_scatter_plot(ax, data, palette)
            elif chart_type == "pie":
                self._create_pie_chart(ax, data, palette)
            elif chart_type == "histogram":
                self._create_histogram(ax, data, palette)
            elif chart_type == "heatmap":
                self._create_heatmap(ax, data, fig)
            elif chart_type == "box":
                self._create_box_plot(ax, data, palette)
            elif chart_type == "area":
                self._create_area_chart(ax, data, palette, legend_labels)
            else:
                return {"error": f"Unknown chart type: {chart_type}"}
            
            # Set labels and title
            ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
            if xlabel:
                ax.set_xlabel(xlabel, fontsize=11)
            if ylabel:
                ax.set_ylabel(ylabel, fontsize=11)
            
            # Grid
            if grid and chart_type not in ["pie", "heatmap"]:
                ax.grid(True, alpha=0.3, linestyle="--")
            
            # Layout
            plt.tight_layout()
            
            # Save
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(self.output_dir, f"chart_{chart_type}_{timestamp}.png")
            elif not output_path.startswith(("/output", self.output_dir)):
                # Force output to /output (mounted volume) to persist after container stops
                filename = os.path.basename(output_path)
                output_path = os.path.join(self.output_dir, filename)
            
            plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            
            return {
                "success": True,
                "chart_type": chart_type,
                "output_path": output_path,
                "message": f"Chart saved to {output_path}"
            }
            
        except Exception as e:
            plt.close("all")
            return {"error": f"Failed to create chart: {str(e)}"}
    
    def _apply_theme(self, theme: str):
        """Apply matplotlib style based on theme."""
        plt.style.use("default")
        
        if theme == "dark":
            plt.rcParams.update({
                "figure.facecolor": "#1a1a2e",
                "axes.facecolor": "#16213e",
                "axes.edgecolor": "#4a5568",
                "axes.labelcolor": "#e2e8f0",
                "text.color": "#e2e8f0",
                "xtick.color": "#a0aec0",
                "ytick.color": "#a0aec0",
                "grid.color": "#4a5568",
            })
        elif theme == "minimal":
            plt.rcParams.update({
                "figure.facecolor": "#ffffff",
                "axes.facecolor": "#ffffff",
                "axes.edgecolor": "#e5e7eb",
                "axes.spines.top": False,
                "axes.spines.right": False,
            })
    
    def _create_bar_chart(self, ax, data: Dict, colors: List[str], show_values: bool):
        """Create vertical bar chart."""
        labels = data.get("labels", data.get("x", []))
        values = data.get("values", data.get("y", []))
        
        x = np.arange(len(labels))
        bars = ax.bar(x, values, color=colors[:len(values)], edgecolor="white", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        
        if show_values:
            for bar, val in zip(bars, values):
                ax.annotate(f"{val:.1f}" if isinstance(val, float) else str(val),
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           ha="center", va="bottom", fontsize=9)
    
    def _create_horizontal_bar(self, ax, data: Dict, colors: List[str], show_values: bool):
        """Create horizontal bar chart."""
        labels = data.get("labels", data.get("x", []))
        values = data.get("values", data.get("y", []))
        
        y = np.arange(len(labels))
        bars = ax.barh(y, values, color=colors[:len(values)], edgecolor="white", linewidth=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        
        if show_values:
            for bar, val in zip(bars, values):
                ax.annotate(f"{val:.1f}" if isinstance(val, float) else str(val),
                           xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                           ha="left", va="center", fontsize=9)
    
    def _create_line_chart(self, ax, data: Dict, colors: List[str], legend_labels: Optional[List[str]]):
        """Create line chart with optional multiple series."""
        x = data.get("x", data.get("labels", list(range(len(data.get("y", []))))))
        
        # Find all y series (y, y1, y2, y3, etc.) - exclude internal _label_ keys
        y_keys = [k for k in data.keys() if k.startswith("y") and not k.startswith("_")]
        if not y_keys:
            y_keys = ["values"] if "values" in data else []
        
        # Determine marker style based on data density
        num_points = len(x) if isinstance(x, list) else 0
        marker = "o" if num_points <= 30 else None
        markersize = 4 if num_points > 15 else 5
        
        for i, key in enumerate(sorted(y_keys)):
            y = data[key]
            # Use embedded _label_ metadata, then legend_labels, then key as fallback
            label_key = f"_label_{key}"
            if label_key in data:
                label = data[label_key]
            elif legend_labels and i < len(legend_labels):
                label = legend_labels[i]
            else:
                label = key
            ax.plot(x, y, marker=marker, color=colors[i % len(colors)], label=label,
                    linewidth=2, markersize=markersize)
        
        # Rotate x-axis labels if they look like dates or are numerous
        if isinstance(x, list) and len(x) > 5 and isinstance(x[0], str):
            ax.tick_params(axis='x', rotation=45)
            for label in ax.get_xticklabels():
                label.set_ha('right')
            # Show fewer ticks if too many dates
            if len(x) > 15:
                step = max(1, len(x) // 8)
                ax.set_xticks(range(0, len(x), step))
                ax.set_xticklabels([x[i] for i in range(0, len(x), step)])
        
        if len(y_keys) > 1 or legend_labels:
            ax.legend()
    
    def _create_scatter_plot(self, ax, data: Dict, colors: List[str]):
        """Create scatter plot."""
        x = data.get("x", [])
        y = data.get("y", [])
        sizes = data.get("sizes", [50] * len(x))
        
        ax.scatter(x, y, c=colors[0], s=sizes, alpha=0.7, edgecolors="white", linewidth=0.5)
    
    def _create_pie_chart(self, ax, data: Dict, colors: List[str]):
        """Create pie chart with percentages."""
        labels = data.get("labels", [])
        values = data.get("values", [])
        
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, colors=colors[:len(values)],
            autopct="%1.1f%%", startangle=90, pctdistance=0.85
        )
        
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_fontweight("bold")
        
        ax.axis("equal")
    
    def _create_histogram(self, ax, data: Union[Dict, List], colors: List[str]):
        """Create histogram."""
        values = data if isinstance(data, list) else data.get("values", [])
        bins = data.get("bins", 20) if isinstance(data, dict) else 20
        
        ax.hist(values, bins=bins, color=colors[0], edgecolor="white", linewidth=0.7, alpha=0.8)
    
    def _create_heatmap(self, ax, data: Union[Dict, List], fig):
        """Create heatmap."""
        matrix = data if isinstance(data, list) else data.get("matrix", [])
        matrix = np.array(matrix)
        
        im = ax.imshow(matrix, cmap="viridis", aspect="auto")
        fig.colorbar(im, ax=ax)
        
        # Add text annotations
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", 
                       color="white" if matrix[i, j] < matrix.max() / 2 else "black", fontsize=8)
    
    def _create_box_plot(self, ax, data: Union[Dict, List], colors: List[str]):
        """Create box plot."""
        if isinstance(data, dict):
            plot_data = [data[k] for k in data.keys() if isinstance(data[k], list)]
            labels = list(data.keys())
        else:
            plot_data = data
            labels = None
        
        bp = ax.boxplot(plot_data, patch_artist=True, labels=labels)
        
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    def _create_area_chart(self, ax, data: Dict, colors: List[str], legend_labels: Optional[List[str]]):
        """Create stacked area chart."""
        x = data.get("x", data.get("labels", []))
        
        y_keys = sorted([k for k in data.keys() if k.startswith("y")])
        y_data = [data[k] for k in y_keys]
        
        labels = legend_labels or y_keys
        ax.stackplot(x, *y_data, labels=labels, colors=colors[:len(y_data)], alpha=0.8)
        
        if len(y_data) > 1:
            ax.legend(loc="upper left")


# For direct testing
if __name__ == "__main__":
    tool = AdvancedGraphTool()
    
    # Test bar chart
    result = tool.run(
        chart_type="bar",
        data={"labels": ["Python", "JavaScript", "Go", "Rust", "Java"], "values": [45, 35, 20, 15, 30]},
        title="Programming Language Popularity",
        xlabel="Language",
        ylabel="Score",
        theme="colorful",
        show_values=True
    )
    print(result)
