from pathlib import Path
from architecture.tool_graph import load_registry, build_tool_graph, get_similar_tools, get_composable_tools
from architecture.tool_embedder import ToolEmbedder

class ToolRetriever:
    def __init__(self,registry_path:str):
        """init everything both graph and embedder"""
        self.registry = load_registry(registry_path)
        self.graph = build_tool_graph(self.registry)
        self.embedder = ToolEmbedder()
        self.embedder.build_index(self.registry)
    
    def get_tools_by_domain(self,domain:str)->list:
        """Get all tools domainwise"""
        tools=[]
        domain_node=f"domain:{domain}"
        if domain_node in self.graph:
            for node in self.graph.nodes():
                if self.graph.has_edge(node, domain_node):
                    tools.append(node)
        return tools

    def retrieve(self,query:str,top_k:int=3,expand:bool=True)->dict:
        """Semantic retrieval + composable tools checker"""
        semantic_results = self.embedder.search(query, top_k=top_k)

        # define a structure for the result
        result = {
            'query': query,
            'primary_tools': semantic_results,
            'similar_tools': [],
            'composable_tools': []
        }
        if not expand:
            return result
        seen = set(r['tool'] for r in semantic_results)
        for r in semantic_results:
            tool_name = r['tool']
            
            similar = get_similar_tools(self.graph, tool_name)
            for name, weight in similar:
                if name not in seen:
                    result['similar_tools'].append({'tool': name, 'similarity': weight})
                    seen.add(name)

            composable = get_composable_tools(self.graph, tool_name)
            for name in composable:
                if name not in seen:
                    result['composable_tools'].append({'tool': name})
                    seen.add(name)
        return result

def print_result(result:dict):
    print(f"Query: '{result['query']}'")

    print("\nPrimary (semantic match):")
    for r in result['primary_tools']:
        print(f"   • {r['tool']} (distance: {r['distance']:.3f})")

    if result['similar_tools']:
        print("\nSimilar (graph expansion):")
        for r in result['similar_tools']:
            print(f"   • {r['tool']} (similarity: {r['similarity']:.2f})")

    if result['composable_tools']:
        print("\nComposable (can chain after):")
        for r in result['composable_tools']:
            print(f"   • {r['tool']}")
        
if __name__ == '__main__':
    registry_path = Path(__file__).parent.parent / 'workspace' / 'tools' / 'registry.json'
    
    retriever = ToolRetriever(str(registry_path))
    
    queries = [
        "analyze my social media profiles",
        "check if input is valid",
        "visualize my spotify data"
    ]
    
    for q in queries:
        result = retriever.retrieve(q, top_k=2, expand=True)
        print_result(result)