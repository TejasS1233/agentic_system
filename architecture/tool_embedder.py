import json
import logging
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def load_registry(registry_path: str) -> dict:
    """Load tool registry from JSON file."""
    with open(registry_path, "r") as f:
        return json.load(f)


def build_embedding_text(tool_name: str, tool_data: dict) -> str:
    """Build embedding text from tool metadata."""
    description = tool_data.get("description", "")
    tags = " ".join(tool_data.get("tags", []))
    domain = tool_data.get("domain", "")
    inputs = " ".join(tool_data.get("input_types", []))
    outputs = " ".join(tool_data.get("output_types", []))
    return f"name:{tool_name} description:{description} tags:{tags} domain:{domain} inputs:{inputs} outputs:{outputs}"


class ToolEmbedder:
    """Embeds and searches tools using FAISS index."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with sentence transformer model."""
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.tool_names = []

    def build_index(self, registry: dict):
        """Build FAISS index from tool registry."""
        if not registry:
            logger.warning("No tools in registry, skipping index build")
            return

        self.tool_names = list(registry.keys())
        texts = [build_embedding_text(name, registry[name]) for name in self.tool_names]
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype("float32"))
        logger.info(
            f"Built index with {len(self.tool_names)} tools, dimension {dimension}"
        )

    def search(self, query: str, top_k: int = 3) -> list:
        """Search for top-k matching tools."""
        if self.index is None:
            return []

        query_embedding = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )
        similarities, indices = self.index.search(
            query_embedding.astype("float32"), top_k
        )

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.tool_names):
                results.append(
                    {
                        "tool": self.tool_names[idx],
                        "similarity": float(similarities[0][i]),
                    }
                )
        return results

    def save_index(self, path: str):
        """Save FAISS index and tool names to disk."""
        faiss.write_index(self.index, path)
        with open(path + ".names.json", "w") as f:
            json.dump(self.tool_names, f)

    def load_index(self, path: str):
        """Load FAISS index and tool names from disk."""
        self.index = faiss.read_index(path)
        with open(path + ".names.json", "r") as f:
            self.tool_names = json.load(f)


def main():
    logging.basicConfig(level=logging.INFO)

    registry_path = (
        Path(__file__).parent.parent / "workspace" / "tools" / "registry.json"
    )
    registry = load_registry(registry_path)

    embedder = ToolEmbedder()
    embedder.build_index(registry)

    test_queries = [
        "find my interests from social media",
        "validate if something is a number",
        "plot my music listening history",
    ]

    logger.info("--- Search Results ---")
    for query in test_queries:
        logger.info(f"Query: '{query}'")
        results = embedder.search(query, top_k=2)
        for r in results:
            logger.info(f"  -> {r['tool']} (similarity: {r['similarity']:.3f})")


if __name__ == "__main__":
    main()
