import networkx as nx
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field


class MarkdownMindmapArgs(BaseModel):
    markdown_string: str = Field(..., description="Markdown string to generate mindmap from")


class MarkdownMindmapTool:
    name = "markdown_mindmap"
    description = "Generate mindmap from Markdown section structure"
    args_schema = MarkdownMindmapArgs

    def run(self, markdown_string: str) -> str:
        # Parse markdown string to extract sections
        sections = markdown_string.split("\n--- ")
        graph = nx.DiGraph()
        graph.add_node("Root")
        for section in sections[1:]:
            lines = section.split("\n")
            section_title = lines[0].strip()
            graph.add_node(section_title)
            graph.add_edge("Root", section_title)
            for line in lines[1:]:
                if line.strip():
                    graph.add_node(line.strip())
                    graph.add_edge(section_title, line.strip())
        # Generate mindmap
        pos = nx.spring_layout(graph)
        nx.draw_networkx(graph, pos, with_labels=True, node_color='lightblue')
        plt.savefig("/output/mindmap.png")
        return "/output/mindmap.png"


def test_tool():
    tool = MarkdownMindmapTool()
    markdown_string = "\n--- FILE: research_paper.pdf ---\nProvided proper attribution is provided, Google hereby grants permission to reproduce the tables and figures in this paper solely for use in journalistic or scholarly works.\nAttention Is All You Need\nAshish Vaswani\nGoogle Brain\navaswani@google.com\nNoam Shazeer\nGoogle Brain\nnoam@google.com\nNiki Parmar\nGoogle Research\nnikip@g"
    output = tool.run(markdown_string)
    print(output)

if __name__ == "__main__":
    test_tool()