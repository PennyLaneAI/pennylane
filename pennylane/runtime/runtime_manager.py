import networkx as nx
from enum import IntEnum, auto
from dataclasses import dataclass
from typing import Any, Callable, Union


class RuntimeManager:
    class NodeType(IntEnum):
        "Internal structure to define node function type"
        HEAD = auto()
        TAIL = auto()
        BODY = auto()

    @dataclass
    class Node:
        """
        Internal data node for handling storage and processing of data in execution pipeline.
        """

        data: Any
        proc_fn: Union[None, Callable] = None
        node_type: NodeType

        def __call__(self):
            if self.proc_fn:
                return self.proc_fn(self.data)
            return self.data

        @property
        def label(self):
            return f"data:={self.data}, proc_fn:={self.proc_fn}"

    def __init__(self):
        self.exec_dag = nx.DiGraph()

    def compile_dag(self):
        self.exec_dag.add_node(NodeType.HEAD)
        self.exec_dag.add_node(NodeType.TAIL)

    def push_task_end(self):
        pass
