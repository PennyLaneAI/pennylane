import operator
import networkx as nx
import functools
import matplotlib.pyplot as plt

class MeasurementValue:
    def __init__(self, value=None):
        """Initializes a MeasurementValue."""
        self.value = value
        self.graph = nx.DiGraph()

    def __hash__(self):
        """
        Returns a unique hash for each instance based on its memory address.
        """
        return id(self)
    
    def get_root_node(self):
        """
        Returns the root node of the computational graph.
        The root node is the one with an out-degree of zero.
        """
        # Find all nodes with an out-degree of 0
        nodes_with_zero_out_degree = [
            node for node, out_degree in self.graph.out_degree() if out_degree == 0
        ]
        # In this graph structure, there should be only one such node
        return nodes_with_zero_out_degree[0] if nodes_with_zero_out_degree else None


    def _create_lazy_mv(self, op, other=None):
        """Helper to create a lazy expression with a new operator."""
        new_mv = MeasurementValue()
        
        # Add a new node for the operation
        op_node = f"op_{op.__name__}_{hash(new_mv)}_{hash(self)}_{hash(other)}"
        print(op_node)
        new_mv.graph.add_node(op_node, op=op)

        # Add operands to the graph
        if self.value is not None:
            # If self is a concrete value, add it as a node
            self_value = f"val_{self.value}_{hash(self)}"
            new_mv.graph.add_node(self_value, value=self.value)
            new_mv.graph.add_edge(self_value, op_node)
        else:
            # If self is a lazy expression, merge its graph
            new_mv.graph = nx.compose(new_mv.graph, self.graph)
            # Add edge from the root of the original graph to the new op node
            root_node = self.get_root_node()
            new_mv.graph.add_edge(root_node, op_node)
                
        # Do the same for the other operand
        if isinstance(other, MeasurementValue):
            if other.value is not None:
                other_value = f"val_{other.value}_{hash(other)}"
                new_mv.graph.add_node(other_value, value=other.value)
                new_mv.graph.add_edge(other_value, op_node)
            else:
                #TODOs s
                new_mv.graph = nx.compose(new_mv.graph, other.graph)
                other_root = other.get_root_node()
                new_mv.graph.add_edge(other_root, op_node)
        #else:
        elif other is not None:
            # Handle non-MeasurementValue operands (e.g., integers)
            other_node = f"val_{other}"
            new_mv.graph.add_node(other_node, value=other)
            new_mv.graph.add_edge(other_node, op_node)
            
        return new_mv
    
    def __xor__(self, other):
        """Overrides the XOR operator to build the graph lazily."""
        print("++")
        return self._create_lazy_mv(operator.xor, other)
    
    def __add__(self, other):
        return self._create_lazy_mv(operator.add, other)
    
    def __mod__(self, other):
        return self._create_lazy_mv(operator.mod, other)
    
    def __truediv__(self, other):
        return self._create_lazy_mv(operator.truediv, other)

    def __invert__(self):
        return self._create_lazy_mv(operator.invert)
    

    def concretize(self):
        """Evaluates the expression graph."""
        if self.value is not None:
            return self.value
        nx.draw(self.graph, with_labels=True, node_color='skyblue', node_size=1500, edge_color='gray', font_size=12, arrows=True)
        plt.title("Simple NetworkX Graph")
        plt.show()
        # Use topological sort to determine the evaluation order
        order = list(nx.topological_sort(self.graph))
        results = {}
        
        for node_id in order:
            node_data = self.graph.nodes[node_id]
            if "op" in node_data:
                # This is an operation node
                op = node_data["op"]
                for predecessor in self.graph.predecessors(node_id):
                    print(op, predecessor)
                operands_values = [
                    results[predecessor] 
                    for predecessor in self.graph.predecessors(node_id)
                ]
                results[node_id] = op(*operands_values)
            elif "value" in node_data:
                # This is a value node (operand)
                results[node_id] = node_data["value"]
        
        # The result of the final operation is the value of the root node
        self.value = results[order[-1]]
        return self.value

# Example Usage
a = MeasurementValue(value=5)
b = MeasurementValue(value=3)
c = MeasurementValue(value=2)

# This builds the graph for (5 ^ 3) ^ 2 lazily
lazy_expression = ~(a ^ b)^c

# Now, concretize the value, which triggers evaluation
print("The result is:", lazy_expression.concretize())