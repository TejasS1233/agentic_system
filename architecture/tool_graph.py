import json
import networkx as nx
from pathlib import Path


def load_registry(registry_path: str) -> dict:
    "Load tool registry from json file"
    with open(registry_path, "r") as f:
        return json.load(f)


def compute_jaccard_similarity(tags1: list, tags2: list) -> float:
    "compute similarity btw list of tools"
    set1, set2 = set(tags1), set(tags2)
    if not set1 or not set2:
        return 0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union


def can_compose(tool_a: dict, tool_b: dict) -> bool:
    """check if output of a can be fed into input of b"""
    outputs_a = set(tool_a.get("output_types", []))
    inputs_b = set(tool_b.get("input_types", []))
    return bool(outputs_a & inputs_b)


def build_tool_graph(registry: dict, similarity_threshold: float = 0.3) -> nx.DiGraph:
    """Build knowledge graph from tool registry with nodes as each tool and eges of three types
    1.similar to (jaccard>threshold)
    2.composes with
    3.belongs to domain
    """
    graph = nx.DiGraph()

    # init the tools as nodes
    for tool_name, tool_data in registry.items():
        graph.add_node(tool_name, node_type="tool", **tool_data)

    # init a new type of node as domain
    domains = set(data.get("domain", "unknown") for data in registry.values())
    for domain in domains:
        graph.add_node(f"domain:{domain}", node_type=domain)
    # idt need to recheck for unknown
    # adding edges for checking which domain constains which tools
    for tool_name, tool_data in registry.items():
        domain = tool_data.get("domain", "unknown")
        graph.add_edge(tool_name, f"domain:{domain}", relation="BELONGS_TO_DOMAIN")

    # use the jaccard fx to add edges for similar to
    tool_names = list(registry.keys())
    for i, name_a in enumerate(tool_names):
        for name_b in tool_names[i + 1 :]:
            tags_a = registry[name_a].get("tags", [])
            tags_b = registry[name_b].get("tags", [])
            similarity = compute_jaccard_similarity(tags_a, tags_b)
            if similarity >= similarity_threshold:
                graph.add_edge(name_a, name_b, relation="SIMILAR_TO", weight=similarity)
                graph.add_edge(name_b, name_a, relation="SIMILAR_TO", weight=similarity)

    for name_a, data_a in registry.items():
        for name_b, data_b in registry.items():
            if name_a != name_b and can_compose(data_a, data_b):
                graph.add_edge(name_a, name_b, relation="COMPOSES_WITH")
    return graph


def get_similar_tools(graph: nx.DiGraph, tool_name: str) -> list:
    """Get tools similar to the given tool"""
    similar = []
    for _, target, data in graph.out_edges(tool_name, data=True):
        if data.get("relation") == "SIMILAR_TO":
            similar.append((target, data.get("weight", 0)))
    # sorts based of jaccard weight in descending
    return sorted(similar, key=lambda x: x[1], reverse=True)


def get_composable_tools(graph: nx.DiGraph, tool_name: str) -> list:
    "get the chainable tools"
    composable = []
    for _, target, data in graph.out_edges(tool_name, data=True):
        if data.get("relation") == "COMPOSES_WITH":
            composable.append(target)
    return composable


if __name__ == "__main__":
    registry_path = (
        Path(__file__).parent.parent / "workspace" / "tools" / "registry.json"
    )
    registry = load_registry(registry_path)
    graph = build_tool_graph(registry)
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")
    print("\nAll edges:")
    for u, v, data in graph.edges(data=True):
        print(f"  {u} --[{data['relation']}]--> {v}")
