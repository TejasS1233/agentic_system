import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss


def load_registry(registry_path: str) -> dict:
    """load tool registry"""
    with open(registry_path, "r") as f:
        return json.load(f)


def build_embedding_text(tool_name: str, tool_data: dict) -> str:
    """convert text to a clean form to embed"""
    name = tool_name
    description = tool_data.get("description", "")
    tags = " ".join(tool_data.get("tags", []))
    domain = tool_data.get("domain", "")
    inputs = " ".join(tool_data.get("domain", ""))
    inputs = " ".join(tool_data.get("input_types", []))
    outputs = " ".join(tool_data.get("output_types", []))
    # combine all
    text = f"name:{name} description:{description} tags:{tags} domain:{domain} inputs:{inputs} outputs:{outputs}"
    return text


class ToolEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        "use the transformer model"
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.tool_names = []

    def build_index(self, registry: dict):
        """Build FAISS index from tool registry"""
        if not registry:
            print("No tools in registry, skipping index build")
            return

        self.tool_names = list(registry.keys())
        texts = [build_embedding_text(name, registry[name]) for name in self.tool_names]

        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.index.add(embeddings.astype("float32"))
        print(f"Built index with {len(self.tool_names)} tools, dimension {dimension}")

    def search(self, query: str, top_k: int = 3) -> list:
        """searches for top 3 tools"""
        if self.index is None:
            return []  # No tools available yet

        query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        similarities, indices = self.index.search(query_embedding.astype("float32"), top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.tool_names):
                results.append(
                    {"tool": self.tool_names[idx], "similarity": float(similarities[0][i])}
                )
        return results

    def save_index(self, path: str):
        """Save faiss index"""
        faiss.write_index(self.index, path)
        with open(path + ".names.json", "w") as f:
            json.dump(self.tool_names, f)

    def load_index(self, path: str):
        """Load FAISS index from disk."""
        self.index = faiss.read_index(path)
        with open(path + ".names.json", "r") as f:
            self.tool_names = json.load(f)


if __name__ == "__main__":
    registry_path = (
        Path(__file__).parent.parent / "workspace" / "tools" / "registry.json"
    )
    registry = load_registry(registry_path)

    embedder = ToolEmbedder()
    embedder.build_index(registry)

    # Test queries
    test_queries = [
        "find my interests from social media",
        "validate if something is a number",
        "plot my music listening history",
    ]

    print("\n--- Search Results ---")
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = embedder.search(query, top_k=2)
        for r in results:
            print(f"  -> {r['tool']} (similarity: {r['similarity']:.3f})")
