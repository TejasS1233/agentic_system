import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from tool_graph import load_registry, build_tool_graph


def visualize_graph(graph: nx.DiGraph, output_path: str = None):
    """Visualize the knowledge graph."""
    plt.figure(figsize=(12, 8))

    # Position nodes using spring layout
    pos = nx.spring_layout(graph, k=2, iterations=50)

    # Color nodes by type
    node_colors = []
    for node in graph.nodes():
        if graph.nodes[node].get("node_type") == "domain":
            node_colors.append("#FF6B6B")  # Red for domains
        else:
            node_colors.append("#4ECDC4")  # Teal for tools

    # Draw nodes
    nx.draw_networkx_nodes(
        graph, pos, node_color=node_colors, node_size=2000, alpha=0.9
    )
    nx.draw_networkx_labels(graph, pos, font_size=8, font_weight="bold")

    # Color edges by relation type
    edge_colors = []
    for u, v, data in graph.edges(data=True):
        relation = data.get("relation", "")
        if relation == "SIMILAR_TO":
            edge_colors.append("#45B7D1")  # Blue
        elif relation == "COMPOSES_WITH":
            edge_colors.append("#96CEB4")  # Green
        else:
            edge_colors.append("#DDA0DD")  # Purple for domain

    nx.draw_networkx_edges(
        graph,
        pos,
        edge_color=edge_colors,
        arrows=True,
        arrowsize=20,
        alpha=0.7,
        connectionstyle="arc3,rad=0.1",
    )

    # Legend
    plt.plot([], [], "o", color="#4ECDC4", label="Tool", markersize=10)
    plt.plot([], [], "o", color="#FF6B6B", label="Domain", markersize=10)
    plt.plot([], [], "-", color="#45B7D1", label="SIMILAR_TO", linewidth=2)
    plt.plot([], [], "-", color="#96CEB4", label="COMPOSES_WITH", linewidth=2)
    plt.plot([], [], "-", color="#DDA0DD", label="BELONGS_TO_DOMAIN", linewidth=2)
    plt.legend(loc="upper left")

    plt.title("Tool Knowledge Graph", fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Graph saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    registry_path = (
        Path(__file__).parent.parent / "workspace" / "tools" / "registry.json"
    )
    registry = load_registry(registry_path)
    graph = build_tool_graph(registry)

    output_path = Path(__file__).parent.parent / "docs" / "tool_graph.png"
    visualize_graph(graph, output_path=str(output_path))
