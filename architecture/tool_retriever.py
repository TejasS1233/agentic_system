from pathlib import Path
from architecture.tool_graph import (
    load_registry,
    build_tool_graph,
    get_similar_tools,
    get_composable_tools,
)
from architecture.tool_embedder import ToolEmbedder


class ToolRetriever:
    def __init__(self, registry_path: str):
        """init everything both graph and embedder"""
        self.registry = load_registry(registry_path)
        self.graph = build_tool_graph(self.registry)
        self.embedder = ToolEmbedder()
        self.embedder.build_index(self.registry)

    def get_tools_by_domain(self, domain: str) -> list:
        """Get all tools domainwise"""
        tools = []
        domain_node = f"domain:{domain}"
        if domain_node in self.graph:
            for node in self.graph.nodes():
                if self.graph.has_edge(node, domain_node):
                    tools.append(node)
        return tools

    def retrieve(self, query: str, top_k: int = 3, expand: bool = True) -> dict:
        """Semantic retrieval + composable tools checker"""
        semantic_results = self.embedder.search(query, top_k=top_k)

        # define a structure for the result
        result = {
            "query": query,
            "primary_tools": semantic_results,
            "similar_tools": [],
            "composable_tools": [],
        }
        if not expand:
            return result
        seen = set(r["tool"] for r in semantic_results)
        for r in semantic_results:
            tool_name = r["tool"]

            similar = get_similar_tools(self.graph, tool_name)
            for name, weight in similar:
                if name not in seen:
                    result["similar_tools"].append({"tool": name, "similarity": weight})
                    seen.add(name)

            composable = get_composable_tools(self.graph, tool_name)
            for name in composable:
                if name not in seen:
                    result["composable_tools"].append({"tool": name})
                    seen.add(name)
        return result

    def retrieve_with_scoring(self, query: str, top_k: int = 5) -> list:
        """
        Hybrid retrieval: semantic similarity + tag boost + domain boost.
        Returns candidates sorted by combined score.
        """
        import re
        
        # Get semantic results
        semantic_results = self.embedder.search(query, top_k=top_k)
        
        # Extract keywords from query (lowercase, remove common words)
        stop_words = {'the', 'a', 'an', 'for', 'in', 'on', 'to', 'from', 'with', 'and', 'or', 'of', 'my', 'me', 'i'}
        query_words = set(word.lower() for word in re.findall(r'\w+', query) if word.lower() not in stop_words)
        
        scored_results = []
        for r in semantic_results:
            tool_name = r["tool"]
            semantic_sim = r["similarity"]
            tool_data = self.registry.get(tool_name, {})
            
            # Tag matching boost
            tags = set(tag.lower() for tag in tool_data.get("tags", []))
            tag_matches = len(query_words & tags)
            tag_boost = tag_matches * 0.15  # 15% boost per matching tag
            
            # Domain boost
            domain = tool_data.get("domain", "").lower()
            domain_boost = 0.1 if domain in query.lower() else 0
            
            # Name keyword boost (if query contains part of tool name)
            name_lower = tool_name.lower()
            name_boost = 0.1 if any(word in name_lower for word in query_words if len(word) > 3) else 0
            
            combined_score = semantic_sim + tag_boost + domain_boost + name_boost
            
            scored_results.append({
                "tool": tool_name,
                "similarity": semantic_sim,
                "tag_matches": tag_matches,
                "combined_score": combined_score,
                "tags": list(tags)[:5],  # Include for debugging
                "description": tool_data.get("description", ""),
            })
        
        # Sort by combined score
        scored_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return scored_results

    def llm_select_tool(self, query: str, candidates: list, llm_manager=None) -> str:
        """
        Use LLM to select the best tool from candidates.
        Returns the tool name that best matches the query.
        """
        if not candidates or not llm_manager:
            return candidates[0]["tool"] if candidates else ""
        
        # Format candidates for LLM
        tool_list = "\n".join([
            f"- {c['tool']}: {c.get('description', 'No description')[:100]}"
            for c in candidates[:5]
        ])
        
        prompt = f"""Given this task: "{query}"

Which tool is the BEST match? Consider what the task is asking for and match it to the tool's purpose.

Available tools:
{tool_list}

Return ONLY the exact tool name, nothing else."""
        
        try:
            response = llm_manager.generate(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.0,
            )
            selected = response.get("content", "").strip()
            # Validate the selection is in our candidates
            valid_names = [c["tool"] for c in candidates]
            if selected in valid_names:
                return selected
        except Exception as e:
            pass  # Fall back to top candidate
        
        return candidates[0]["tool"] if candidates else ""


def print_result(result: dict):
    print(f"Query: '{result['query']}'")

    print("\nPrimary (semantic match):")
    for r in result["primary_tools"]:
        print(f"   • {r['tool']} (similarity: {r['similarity']:.3f})")

    if result["similar_tools"]:
        print("\nSimilar (graph expansion):")
        for r in result["similar_tools"]:
            print(f"   • {r['tool']} (similarity: {r['similarity']:.2f})")

    if result["composable_tools"]:
        print("\nComposable (can chain after):")
        for r in result["composable_tools"]:
            print(f"   • {r['tool']}")


if __name__ == "__main__":
    registry_path = (
        Path(__file__).parent.parent / "workspace" / "tools" / "registry.json"
    )

    retriever = ToolRetriever(str(registry_path))

    queries = [
        "analyze my social media profiles",
        "check if input is valid",
        "visualize my spotify data",
    ]

    for q in queries:
        result = retriever.retrieve(q, top_k=2, expand=True)
        print_result(result)
