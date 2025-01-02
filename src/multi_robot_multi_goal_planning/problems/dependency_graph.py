# generated by claude
class DependencyGraph:
    def __init__(self):
        # Store dependencies as adjacency lists
        self.dependencies = {}
        # Store reverse dependencies for efficient lookup
        self.reverse_deps = {}
        
    def add_node(self, node):
        """Add a node to the graph if it doesn't exist."""
        if node not in self.dependencies:
            self.dependencies[node] = set()
            self.reverse_deps[node] = set()
    
    def add_dependency(self, dependent, dependency):
        """Add a dependency: dependent depends on dependency."""
        # Add both nodes if they don't exist
        self.add_node(dependent)
        self.add_node(dependency)
        
        # Add the dependency relationship
        self.dependencies[dependent].add(dependency)
        self.reverse_deps[dependency].add(dependent)
        
        # Check for cycles
        if self._has_cycle():
            # Remove the added dependency
            self.dependencies[dependent].remove(dependency)
            self.reverse_deps[dependency].remove(dependent)
            raise ValueError(f"Adding {dependent} -> {dependency} would create a cycle")
    
    def _has_cycle(self):
        """Check if the graph has any cycles using DFS."""
        visited = set()
        path = set()
        
        def dfs(node):
            visited.add(node)
            path.add(node)
            
            for neighbor in self.dependencies[node]:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in path:
                    return True
                    
            path.remove(node)
            return False
        
        for node in self.dependencies:
            if node not in visited:
                if dfs(node):
                    return True
        return False
    
    def get_all_dependencies(self, node):
        """Get all dependencies of a node (direct and indirect)."""
        if node not in self.dependencies:
            raise ValueError(f"Node {node} not in graph")
            
        all_deps = set()
        def collect_deps(n):
            for dep in self.dependencies[n]:
                if dep not in all_deps:
                    all_deps.add(dep)
                    collect_deps(dep)
                    
        collect_deps(node)
        return all_deps

    def get_direct_dependencies(self, node):
        return self.dependencies[node]
    
    def get_dependents(self, node):
        """Get all nodes that depend on this node."""
        if node not in self.reverse_deps:
            raise ValueError(f"Node {node} not in graph")
        return self.reverse_deps[node]
    
    def get_build_order(self):
        """Return a valid build order (topological sort)."""
        result = []
        visited = set()

        def dfs(node):
            if node in visited:
                return 
            visited.add(node)
            
            for dep in self.dependencies[node]:
                dfs(dep)
            result.append(node)

        # Visit all nodes
        for node in self.dependencies:
            dfs(node)

        return result  # No need to reverse since order is correct
    
    def get_root_nodes(self):
        """Get nodes that don't depend on anything."""
        return {node for node in self.dependencies 
                if not self.dependencies[node]}
    
    def get_leaf_nodes(self):
        """Get nodes that nothing depends on."""
        return {node for node in self.dependencies 
                if not self.reverse_deps[node]}
    
    def __str__(self):
        """String representation showing all dependencies."""
        return '\n'.join(f"{node} -> {deps}" 
                        for node, deps in self.dependencies.items()
                        if deps)