class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def add_node(self, node_id):
        if node_id not in self.nodes:
            self.nodes[node_id] = []
            self.edges[node_id] = {}

    def add_edge(self, node1, node2, weight=1):
        if node1 in self.nodes and node2 in self.nodes:
            self.nodes[node1].append(node2)
            self.edges[node1][node2] = weight

    def dfs(self, start_node):
        visited = set()
        self._dfs_helper(start_node, visited)

    def _dfs_helper(self, node, visited):
        visited.add(node)
        print(node)
        for neighbor in self.nodes[node]:
            if neighbor not in visited:
                self._dfs_helper(neighbor, visited)

    def bfs(self, start_node):
        visited = set()
        queue = [start_node]
        visited.add(start_node)

        while queue:
            node = queue.pop(0)
            print(node)
            for neighbor in self.nodes[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.add(neighbor)

# Example usage:
graph = Graph()
graph.add_node(1)
graph.add_node(2)
graph.add_node(3)
graph.add_edge(1, 2)
graph.add_edge(2, 3)
graph.add_edge(1, 3)

print("DFS Traversal:")
graph.dfs(1)

print("BFS Traversal:")
graph.bfs(1)