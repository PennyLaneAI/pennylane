import operator
import networkx as nx
import functools

class MeasurementValue:
    def __init__(self, value=None, graph=None, root_node=None):
        self.value = value
        self.graph = graph if graph is not None else nx.DiGraph()
        self.root_node = root_node

        if self.value is not None and not self.graph.nodes:
            node_name = f"val_{self.value}"
            self.graph.add_node(node_name, value=self.value)
            self.root_node = node_name

    def _create_lazy_mv(self, op, other=None):
        new_mv = MeasurementValue()
        op_node_name = f"op_{op.__name__}_{hash(new_mv)}"
        new_mv.graph.add_node(op_node_name, op=op)
        new_mv.root_node = op_node_name

        # Handle 'self' operand
        new_mv.graph = nx.compose(new_mv.graph, self.graph)
        new_mv.graph.add_edge(self.root_node, op_node_name)

        # Handle 'other' operand for binary ops
        if other is not None:
            if isinstance(other, MeasurementValue):
                new_mv.graph = nx.compose(new_mv.graph, other.graph)
                new_mv.graph.add_edge(other.root_node, op_node_name)
            else:
                other_node_name = f"val_{other}"
                new_mv.graph.add_node(other_node_name, value=other)
                new_mv.graph.add_edge(other_node_name, op_node_name)
        
        return new_mv

    def __xor__(self, other):
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
        if self.value is not None:
            return self.value
        
        try:
            order = list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            raise ValueError("Expression graph contains a cycle, cannot be evaluated.")
        
        results = {}
        for node_id in order:
            node_data = self.graph.nodes[node_id]
            if "op" in node_data:
                op = node_data["op"]
                operands = [results[pred] for pred in self.graph.predecessors(node_id)]
                results[node_id] = op(*operands)
            elif "value" in node_data:
                results[node_id] = node_data["value"]
        
        self.value = results[self.root_node]
        return self.value

# Example Usage:
a = MeasurementValue(value=5)
b = MeasurementValue(value=3)
c = 2

# This now works correctly
lazy_expression = (a ^ b) / c
print(f"The graph nodes are: {lazy_expression.graph.nodes}")
print(f"The result of (5 ^ 3) / 2 is: {lazy_expression.concretize()}")

lazy_expression_2 = ~(a + b) + c
print(f"The result of ~(5 + 3) is: {lazy_expression_2.concretize()}")