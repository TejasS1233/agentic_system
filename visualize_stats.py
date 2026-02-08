"""Visualize IASCIS accomplished task stats from the accomplished_tasks file."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import re
import os

# --- Parse the accomplished_tasks file ---
with open("docs/accomplished_tasks", "r", encoding="utf-8") as f:
    content = f.read()

# Extract tasks
task_pattern = re.compile(
    r"### (\d+)\. (.+?)\n\n"
    r"- \*\*Query:\*\* (.+?)\n"
    r"- \*\*Time:\*\* ([\d.]+)s \| Steps: (\d+)",
    re.DOTALL
)

tasks = []
for m in task_pattern.finditer(content):
    num, name, query, time_s, steps = m.groups()
    # Extract tools if present
    tools_match = re.search(r"Tools: (.+?)(?:\n|$)", m.group(0))
    tools = [t.strip() for t in tools_match.group(1).split(",")] if tools_match else []
    
    # Extract perf grades
    grades = re.findall(r"\((\w+)\)", m.group(0)[m.group(0).find("Perf:"):]) if "Perf:" in m.group(0) else []
    
    tasks.append({
        "num": int(num),
        "name": name.strip(),
        "query": query.strip(),
        "time_s": float(time_s),
        "steps": int(steps),
        "tools": tools,
        "grades": grades,
    })

# Removed tasks to exclude
removed = {
    "Chess Gender Ratio Plot", "Census Population Plot", "Twitter Trends",
    "Cat Image", "Brawl Stars Brawler Stats", "Brawl Stars Pie Chart"
}
tasks = [t for t in tasks if t["name"] not in removed]

print(f"Parsed {len(tasks)} tasks for visualization")

# --- Color palette ---
BG = "#0d1117"
CARD_BG = "#161b22"
TEXT = "#c9d1d9"
ACCENT = "#58a6ff"
GREEN = "#3fb950"
YELLOW = "#d29922"
RED = "#f85149"
ORANGE = "#d18616"
PURPLE = "#bc8cff"
CYAN = "#39d353"
GRID = "#21262d"

# Grade colors
GRADE_COLORS = {"fast": GREEN, "moderate": YELLOW, "slow": ORANGE, "critical": RED}

# --- Create figure ---
fig = plt.figure(figsize=(24, 20), facecolor=BG)
fig.suptitle("IASCIS - Task Performance Dashboard", fontsize=28, fontweight="bold",
             color=TEXT, y=0.98)
fig.text(0.5, 0.955, f"{len(tasks)} Successful Tasks | Avg: {np.mean([t['time_s'] for t in tasks]):.1f}s | "
         f"Min: {min(t['time_s'] for t in tasks):.1f}s | Max: {max(t['time_s'] for t in tasks):.1f}s",
         ha="center", fontsize=13, color="#8b949e")

gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.3, left=0.06, right=0.97, top=0.93, bottom=0.04)

# ============================================================
# 1. HORIZONTAL BAR: Task execution times (sorted)
# ============================================================
ax1 = fig.add_subplot(gs[0, :2])
ax1.set_facecolor(CARD_BG)

sorted_tasks = sorted(tasks, key=lambda t: t["time_s"])
names = [t["name"][:35] for t in sorted_tasks]
times = [t["time_s"] for t in sorted_tasks]
colors = [GREEN if t < 8 else YELLOW if t < 15 else ORANGE if t < 25 else RED for t in times]

bars = ax1.barh(range(len(names)), times, color=colors, height=0.7, edgecolor="none", alpha=0.9)
ax1.set_yticks(range(len(names)))
ax1.set_yticklabels(names, fontsize=7, color=TEXT)
ax1.set_xlabel("Time (seconds)", color=TEXT, fontsize=10)
ax1.set_title("Task Execution Times", color=TEXT, fontsize=14, fontweight="bold", pad=10)
ax1.tick_params(colors=TEXT, labelsize=8)
ax1.set_xlim(0, max(times) * 1.15)
ax1.spines[:].set_color(GRID)
ax1.xaxis.grid(True, color=GRID, alpha=0.5, linestyle="--")

# Add time labels
for bar, t in zip(bars, times):
    ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
             f"{t:.1f}s", va="center", fontsize=6.5, color="#8b949e")

# ============================================================
# 2. PIE: Pipeline complexity (1-step, 2-step, 3-step)
# ============================================================
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor(CARD_BG)

step_counts = {}
for t in tasks:
    step_counts[t["steps"]] = step_counts.get(t["steps"], 0) + 1

labels = [f"{k}-step" for k in sorted(step_counts.keys())]
sizes = [step_counts[k] for k in sorted(step_counts.keys())]
pie_colors = [ACCENT, PURPLE, CYAN][:len(labels)]

wedges, texts, autotexts = ax2.pie(
    sizes, labels=labels, autopct=lambda p: f"{int(p*sum(sizes)/100)}",
    colors=pie_colors, startangle=90, textprops={"color": TEXT, "fontsize": 11},
    pctdistance=0.6, wedgeprops={"edgecolor": BG, "linewidth": 2}
)
for t in autotexts:
    t.set_fontsize(14)
    t.set_fontweight("bold")
ax2.set_title("Pipeline Complexity", color=TEXT, fontsize=14, fontweight="bold", pad=10)

# ============================================================
# 3. TOP TOOLS BAR CHART
# ============================================================
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor(CARD_BG)

tool_counts = {}
for t in tasks:
    for tool in t["tools"]:
        tool = tool.strip()
        if tool:
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

# Top 12 tools
top_tools = sorted(tool_counts.items(), key=lambda x: -x[1])[:12]
tool_names = [t[0][:25] for t in top_tools]
tool_vals = [t[1] for t in top_tools]

bars3 = ax3.barh(range(len(tool_names)), tool_vals, color=ACCENT, height=0.65, alpha=0.85)
ax3.set_yticks(range(len(tool_names)))
ax3.set_yticklabels(tool_names, fontsize=8, color=TEXT)
ax3.set_xlabel("Tasks", color=TEXT, fontsize=10)
ax3.set_title("Most Used Tools (Top 12)", color=TEXT, fontsize=14, fontweight="bold", pad=10)
ax3.tick_params(colors=TEXT, labelsize=8)
ax3.spines[:].set_color(GRID)
ax3.invert_yaxis()

for bar, v in zip(bars3, tool_vals):
    ax3.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
             str(v), va="center", fontsize=9, color=TEXT, fontweight="bold")

# ============================================================
# 4. TIME DISTRIBUTION HISTOGRAM
# ============================================================
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor(CARD_BG)

all_times = [t["time_s"] for t in tasks]
bins = [0, 5, 8, 12, 16, 20, 30, 100]
hist_colors = [GREEN, GREEN, YELLOW, YELLOW, ORANGE, RED, RED]

n, bin_edges, patches = ax4.hist(all_times, bins=bins, color=ACCENT, edgecolor=BG, alpha=0.85, rwidth=0.85)
for patch, color in zip(patches, hist_colors):
    patch.set_facecolor(color)

ax4.set_xlabel("Time (seconds)", color=TEXT, fontsize=10)
ax4.set_ylabel("Number of Tasks", color=TEXT, fontsize=10)
ax4.set_title("Execution Time Distribution", color=TEXT, fontsize=14, fontweight="bold", pad=10)
ax4.tick_params(colors=TEXT, labelsize=9)
ax4.spines[:].set_color(GRID)

# Add count labels on bars
for i, (count, x) in enumerate(zip(n, bin_edges[:-1])):
    if count > 0:
        width = bin_edges[i+1] - bin_edges[i]
        ax4.text(x + width/2, count + 0.2, str(int(count)),
                ha="center", fontsize=11, color=TEXT, fontweight="bold")

# ============================================================
# 5. STEPS vs TIME SCATTER
# ============================================================
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_facecolor(CARD_BG)

for t in tasks:
    color = GREEN if t["time_s"] < 8 else YELLOW if t["time_s"] < 15 else ORANGE if t["time_s"] < 25 else RED
    ax5.scatter(t["steps"], t["time_s"], c=color, s=80, alpha=0.8, edgecolors=TEXT, linewidth=0.5, zorder=3)

ax5.set_xlabel("Number of Steps", color=TEXT, fontsize=10)
ax5.set_ylabel("Time (seconds)", color=TEXT, fontsize=10)
ax5.set_title("Steps vs Execution Time", color=TEXT, fontsize=14, fontweight="bold", pad=10)
ax5.set_xticks([1, 2, 3])
ax5.tick_params(colors=TEXT, labelsize=9)
ax5.spines[:].set_color(GRID)
ax5.yaxis.grid(True, color=GRID, alpha=0.5, linestyle="--")

# Add mean lines
for step in [1, 2, 3]:
    step_times = [t["time_s"] for t in tasks if t["steps"] == step]
    if step_times:
        mean_t = np.mean(step_times)
        ax5.plot([step - 0.3, step + 0.3], [mean_t, mean_t], color=ACCENT, linewidth=2, zorder=4)
        ax5.text(step + 0.35, mean_t, f"avg {mean_t:.1f}s", fontsize=8, color=ACCENT)

# ============================================================
# 6. CATEGORY BREAKDOWN (bottom left)
# ============================================================
ax6 = fig.add_subplot(gs[2, 0])
ax6.set_facecolor(CARD_BG)

categories = {
    "Search / Web": ["Google Search", "Search:", "Toolformer", "Python tutorials"],
    "Data & Charts": ["Chart", "Plot", "Bar chart", "Pie", "Graph", "Revenue", "Statistical", "Stock"],
    "Social Media": ["Reddit", "Dev.to", "Product Hunt", "GitHub"],
    "Utility": ["Weather", "Currency", "Route", "Image", "Dog"],
    "Research": ["Citations", "Table Extraction", "Paper", "Bibliography"],
    "Gaming/Fun": ["Royal Rumble", "Brawl Stars", "Games", "Fibonacci"],
    "Data Analysis": ["Executive Summary", "Sales", "Placement"],
}

cat_counts = {cat: 0 for cat in categories}
cat_times = {cat: [] for cat in categories}

for t in tasks:
    for cat, keywords in categories.items():
        if any(kw.lower() in t["name"].lower() or kw.lower() in t["query"].lower() for kw in keywords):
            cat_counts[cat] += 1
            cat_times[cat].append(t["time_s"])
            break

sorted_cats = sorted(cat_counts.items(), key=lambda x: -x[1])
cat_names = [c[0] for c in sorted_cats if c[1] > 0]
cat_vals = [c[1] for c in sorted_cats if c[1] > 0]
cat_avg = [np.mean(cat_times[c]) if cat_times[c] else 0 for c in cat_names]
cat_colors = [ACCENT, PURPLE, CYAN, GREEN, YELLOW, ORANGE, RED][:len(cat_names)]

bars6 = ax6.bar(range(len(cat_names)), cat_vals, color=cat_colors, alpha=0.85, edgecolor="none")
ax6.set_xticks(range(len(cat_names)))
ax6.set_xticklabels(cat_names, fontsize=8, color=TEXT, rotation=30, ha="right")
ax6.set_ylabel("Tasks", color=TEXT, fontsize=10)
ax6.set_title("Task Categories", color=TEXT, fontsize=14, fontweight="bold", pad=10)
ax6.tick_params(colors=TEXT, labelsize=9)
ax6.spines[:].set_color(GRID)

for bar, v, avg in zip(bars6, cat_vals, cat_avg):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
             f"{v}\n({avg:.0f}s avg)", ha="center", fontsize=8, color=TEXT)

# ============================================================
# 7. TASK TIMELINE (bottom middle + right)
# ============================================================
ax7 = fig.add_subplot(gs[2, 1:])
ax7.set_facecolor(CARD_BG)

# Sort by task number (chronological order in file)
chrono = sorted(tasks, key=lambda t: t["num"])
x_pos = range(len(chrono))
y_times = [t["time_s"] for t in chrono]
colors7 = [GREEN if t < 8 else YELLOW if t < 15 else ORANGE if t < 25 else RED for t in y_times]

ax7.bar(x_pos, y_times, color=colors7, alpha=0.85, width=0.8, edgecolor="none")
ax7.set_xticks(list(x_pos)[::2])
ax7.set_xticklabels([chrono[i]["name"][:18] for i in range(0, len(chrono), 2)],
                    fontsize=6.5, color=TEXT, rotation=45, ha="right")
ax7.set_ylabel("Time (seconds)", color=TEXT, fontsize=10)
ax7.set_title("All Tasks Timeline (Chronological)", color=TEXT, fontsize=14, fontweight="bold", pad=10)
ax7.tick_params(colors=TEXT, labelsize=8)
ax7.spines[:].set_color(GRID)
ax7.yaxis.grid(True, color=GRID, alpha=0.3, linestyle="--")

# Rolling average line
if len(y_times) >= 3:
    window = 3
    rolling_avg = [np.mean(y_times[max(0,i-window+1):i+1]) for i in range(len(y_times))]
    ax7.plot(x_pos, rolling_avg, color=ACCENT, linewidth=2, alpha=0.8, label=f"{window}-task rolling avg")
    ax7.legend(loc="upper right", fontsize=9, facecolor=CARD_BG, edgecolor=GRID, labelcolor=TEXT)

# Legend for colors
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=GREEN, label="< 8s (fast)"),
    Patch(facecolor=YELLOW, label="8-15s (moderate)"),
    Patch(facecolor=ORANGE, label="15-25s (slow)"),
    Patch(facecolor=RED, label="> 25s (critical)"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=10,
           facecolor=CARD_BG, edgecolor=GRID, labelcolor=TEXT,
           bbox_to_anchor=(0.5, 0.005))

# Save
output_path = os.path.join("docs", "task_performance_dashboard.png")
fig.savefig(output_path, dpi=150, facecolor=BG, bbox_inches="tight")
print(f"\nDashboard saved to: {output_path}")
plt.close()
